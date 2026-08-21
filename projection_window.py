import torch
import torch.nn as nn
import torch.nn.functional as F

def get_dist_indices(flat_input, embed_weight):
    dist = -(
        flat_input.pow(2).sum(1, keepdim=True)
        - 2 * flat_input @ embed_weight
        + embed_weight.pow(2).sum(0, keepdim=True)
    )
    return dist.max(dim=-1).indices

def bin_times_to_frames(bin_times, frame_hz):
    return (torch.tensor(bin_times) * frame_hz).long().tolist()

class ProjectionWindow():
    def __init__(self, 
                 bin_sec=[1.4, 0.6, 0.2, 0.4, 0.6, 0.8], 
                 num_hist_bins=2, 
                 frame_hz=25, 
                 threshold=0.5, 
                 mode="role_based", 
                 cut_center_frame=False):
        super().__init__()
        self.bin_sec = bin_sec
        self.frame_hz = frame_hz
        self.threshold = threshold
        self.mode = mode.lower()
        self.cut_center_frame = cut_center_frame
        self.num_hist_bins = num_hist_bins
        
        self.bin_frames = bin_times_to_frames(self.bin_sec, frame_hz)
        self.n_bins = len(self.bin_frames)
        
        self.hist_frames = sum(self.bin_frames[:num_hist_bins])
        self.fut_frames = sum(self.bin_frames[num_hist_bins:])
        
        if self.mode in ["role_based", "speaker_based"]:
            code_vecs = self._create_code_vectors((self.n_bins-self.num_hist_bins) * 2)
        elif self.mode == "independent":
            code_vecs = self._create_code_vectors(self.n_bins)
        else:
            raise ValueError(f"Invalid mode: {self.mode}.")
             
        if code_vecs is not None:
            self.n_classes = code_vecs.shape[0]
            self.emb = nn.Embedding(self.n_classes, code_vecs.shape[1])
            self.emb.weight.data = code_vecs
            self.emb.weight.requires_grad_(False)
            self.emb = self.emb.to('cpu')
        else:
            self.n_classes = 0

    def __repr__(self):
        lines = [
            f"{self.__class__.__name__}(",
            f"  mode='{self.mode}',",
            f"  bin_sec={self.bin_sec},",
            f"  bin_frames={self.bin_frames},",
            f"  num_hist_bins={self.num_hist_bins} (total hist_frames: {self.hist_frames}),",
            f"  num_fut_bins={self.n_bins - self.num_hist_bins} (total fut_frames: {self.fut_frames}),",
            f"  frame_hz={self.frame_hz},",
            f"  threshold={self.threshold},",
            f"  cut_center_frame={self.cut_center_frame},",
            f"  n_classes={self.n_classes}",
            f")"
        ]
        return "\n".join(lines)

    def to(self, *args, **kwargs):
        return self  

    def _create_code_vectors(self, n_bins):
        code_list = [[(i >> j) & 1 for j in reversed(range(n_bins))] for i in range(2**n_bins)]
        return torch.tensor(code_list, dtype=torch.float32)

    def extract_bins(self, va):
        pre, post = self.hist_frames, self.fut_frames
        unfold_size = pre + post + (1 if self.cut_center_frame else 0)

        pad_pre = max(0, pre - (0 if self.cut_center_frame else 1)) 
        
        win = F.pad(va, (0, 0, pad_pre, post)).unfold(1, unfold_size, 1)
        
        if pre == 0 and not self.cut_center_frame:
            win = win[:, :va.size(1), ...] 

        bins, start = [], 0
        for b in self.bin_frames:
            val = win[..., start:start + b].mean(dim=-1)
            bins.append((val >= self.threshold).float())
            start += b
            
        return torch.stack(bins, dim=-1)

    def _encode_base(self, flat_tensor, original_shape):
        embed = self.emb.weight.T.to(flat_tensor)
        dist = -(
            flat_tensor.pow(2).sum(1, keepdim=True) 
            - 2 * flat_tensor @ embed 
            + embed.pow(2).sum(0, keepdim=True)
        )
        return dist.max(dim=-1).indices.view(original_shape)

    def independent_projection(self, x):
        B, N, C, nb = x.shape
        return self._encode_base(x.view(B * N * C, nb), (B, N, C))
    
    def speaker_based_projection(self, x):
        B, N, C, nb = x.shape
        return self._encode_base(x.view(B * N, C * nb), (B, N))

    def role_based_projection(self, x):
        """
        x: [B, N, C, nB]
        """
        B, N, C, _ = x.shape
        
        v_h1 = x[..., 0]
        v_h2 = x[..., 1]

        v_f1 = (x[..., 2] + x[..., 3]).clamp(0, 1) 
        v_f2 = (x[..., 4] + x[..., 5]).clamp(0, 1)

        vals_hist = v_h1 * 2 + v_h2 * 1
        vals_fut  = v_f1 * 1 + v_f2 * 2
        
        score_h = vals_hist * 10 + vals_fut
        score_f = vals_fut * 10 + vals_hist

        top2_h = torch.topk(score_h, k=2, dim=2).indices
        idx_wh = top2_h[..., 0:1]
        
        is_tied = (vals_hist.gather(2, idx_wh) == vals_hist.gather(2, top2_h[..., 1:2]))

        score_f.scatter_(2, idx_wh, -float('inf'))
        idx_wf_untied = score_f.max(dim=2, keepdim=True).indices
        idx_wf_tied = top2_h[..., 1:2]

        idx_wf = torch.where(is_tied, idx_wf_tied, idx_wf_untied)
        top_indices = torch.cat([idx_wh, idx_wf], dim=2)

        future_bins = x[..., 2:]
        
        pairs = torch.gather(future_bins, 2, 
                             top_indices.unsqueeze(-1).expand(-1, -1, -1, 4))

        return self._encode_base(pairs.view(B * N, 8), (B, N))

    @torch.no_grad()
    def get_labels(self, va):
        """va: [B, C, T]
        """
        B, C, T = va.shape
        va = va.transpose(1, 2)
        bins = self.extract_bins(va).type(va.dtype)

        if self.mode == "role_based": return self.role_based_projection(bins)
        elif self.mode == "speaker_based": return self.speaker_based_projection(bins)
        elif self.mode == "independent": return bins.view(B, T, self.n_bins)

        else: raise ValueError("Invalid mode.")

    def decode(self, idx):
        v = self.emb(idx)
        if self.mode == "independent":
            return v 
        else:
            return v.view(*idx.shape, 2, self.n_bins-self.num_hist_bins)

    # Due to class imbalance, shift_scale 2.0 should be used for shift pred
    def probs_agg(self, probs, from_bin, to_bin, shift_scale=1.0):
        idx = torch.arange(self.n_classes, device=probs.device)
        states = self.decode(idx)
        
        if self.mode == "independent":
            abp = states[:, from_bin:to_bin+1].sum(-1)
            p_all = torch.einsum("...d,d->...", probs, abp)
        else:
            abp = states[:, :, from_bin:to_bin+1].sum(-1)
            p_all = torch.einsum("...d,dc->...c", probs, abp)
            p_all[..., 1] = p_all[..., 1] * shift_scale
            
        p_all = p_all / (p_all.sum(-1, keepdim=True) + 1e-8)
        return p_all

    def get_shift_hold(self, logits, shift_scale=2.0):
        return self.get_probs(logits, 2, 3, shift_scale=shift_scale)

    def get_probs(self, logits, start_bin, end_bin, shift_scale=1.0):
        probs = logits.softmax(dim=-1)
        probs_agg = self.probs_agg(probs, start_bin, end_bin, shift_scale=shift_scale)

        shift_prob = probs_agg[..., 1] if self.mode != "independent" else probs_agg
        pred = (shift_prob > 0.5).long()

        return {
            "probs": probs,
            "pred": pred,
            "p_shift": shift_prob,
        }