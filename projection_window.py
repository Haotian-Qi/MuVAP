import torch
import torch.nn.functional as F


def bin_times_to_frames(bin_times, frame_hz):
    frames = [round(seconds * frame_hz) for seconds in bin_times]
    if any(frame <= 0 for frame in frames):
        raise ValueError("Projection bins must contain at least one frame")
    return frames


class ProjectionWindow:
    def __init__(
        self,
        bin_sec=(1.4, 0.6, 0.6, 1.4),
        num_hist_bins=2,
        frame_hz=25,
        threshold=0.5,
        mode="role_relative",
        cut_center_frame=False,
    ):
        self.bin_sec = list(bin_sec)
        self.frame_hz = frame_hz
        self.threshold = threshold
        self.mode = mode.lower()
        self.cut_center_frame = cut_center_frame
        self.num_hist_bins = num_hist_bins
        self.bin_frames = bin_times_to_frames(self.bin_sec, frame_hz)
        self.n_bins = len(self.bin_frames)
        self.hist_frames = sum(self.bin_frames[:num_hist_bins])
        self.fut_frames = sum(self.bin_frames[num_hist_bins:])

        if not 0 <= num_hist_bins <= self.n_bins:
            raise ValueError("num_hist_bins is outside the configured bins")
        if self.mode not in {
            "role_relative",
            "role_future",
            "speaker_based",
            "independent",
        }:
            raise ValueError(f"Unknown projection mode: {mode}")
        if self.mode == "role_relative" and (self.n_bins != 4 or num_hist_bins != 2):
            raise ValueError(
                "role_relative mode requires two history and two future bins"
            )
        if self.mode == "role_future" and (self.n_bins != 6 or num_hist_bins != 2):
            raise ValueError(
                "role_future mode requires two history and four future bins"
            )

        if self.mode in {"role_relative", "role_future"}:
            self.codebook, self.pair_indices = self._unordered_pair_codebook()
            self.n_classes = len(self.codebook)
        elif self.mode == "speaker_based":
            encoded_bits = 2 * (self.n_bins - num_hist_bins)
            self.n_classes = 2**encoded_bits
            self.codebook = self._codebook(encoded_bits)
            self.pair_indices = None
        else:
            self.n_classes = 0
            self.codebook = None
            self.pair_indices = None

    def __repr__(self):
        return (
            f"ProjectionWindow(mode={self.mode!r}, frame_hz={self.frame_hz}, "
            f"bin_frames={self.bin_frames}, n_classes={self.n_classes})"
        )

    @staticmethod
    def _codebook(bits):
        values = torch.arange(2**bits)
        shifts = torch.arange(bits - 1, -1, -1)
        return values.unsqueeze(1).bitwise_right_shift(shifts).bitwise_and(1).float()

    @classmethod
    def _unordered_pair_codebook(cls):
        patterns = cls._codebook(4)
        pairs = []
        pair_indices = torch.empty(16, 16, dtype=torch.long)
        for first in range(16):
            for second in range(first, 16):
                index = len(pairs)
                pairs.append(torch.cat([patterns[first], patterns[second]]))
                pair_indices[first, second] = index
                pair_indices[second, first] = index
        return torch.stack(pairs), pair_indices

    @staticmethod
    def _binary_index(values):
        bits = values.shape[-1]
        weights = 2 ** torch.arange(bits - 1, -1, -1, device=values.device)
        return (values.long() * weights).sum(dim=-1)

    def extract_bins(self, activity):
        pre = self.hist_frames
        post = self.fut_frames
        center = 1 if self.cut_center_frame else 0
        window_size = pre + post + center
        pad_pre = max(0, pre - (0 if self.cut_center_frame else 1))
        windows = F.pad(activity, (0, 0, pad_pre, post)).unfold(1, window_size, 1)
        windows = windows[:, : activity.shape[1]]

        bins = []
        start = 0
        for width in self.bin_frames:
            value = windows[..., start : start + width].mean(dim=-1)
            bins.append((value >= self.threshold).to(activity.dtype))
            start += width
        return torch.stack(bins, dim=-1)

    def role_future_projection(self, bins):
        if bins.shape[2] < 2:
            raise ValueError("role_future projection requires at least two speakers")

        history_1, history_2 = bins[..., 0], bins[..., 1]
        future_1 = (bins[..., 2] + bins[..., 3]).clamp(max=1)
        future_2 = (bins[..., 4] + bins[..., 5]).clamp(max=1)
        history = history_1 * 2 + history_2
        future = future_1 + future_2 * 2

        history_score = history * 10 + future
        future_score = future * 10 + history
        top_history = torch.topk(history_score, k=2, dim=2).indices
        current = top_history[..., :1]
        tied = history.gather(2, current) == history.gather(2, top_history[..., 1:2])

        future_score = future_score.clone()
        future_score.scatter_(2, current, -torch.inf)
        next_untied = future_score.argmax(dim=2, keepdim=True)
        next_speaker = torch.where(tied, top_history[..., 1:2], next_untied)
        speakers = torch.cat([current, next_speaker], dim=2)
        future_bins = bins[..., self.num_hist_bins :]
        selected = torch.gather(
            future_bins,
            2,
            speakers.unsqueeze(-1).expand(-1, -1, -1, future_bins.shape[-1]),
        )
        patterns = self._binary_index(selected)
        return self.pair_indices.to(patterns.device)[patterns[..., 0], patterns[..., 1]]

    def role_relative_projection(self, bins):
        if bins.shape[2] < 2:
            raise ValueError("role_relative projection requires at least two speakers")

        history = bins[..., :2].sum(dim=-1)
        future = bins[..., 2:].sum(dim=-1)
        current_score = history * 10 + future
        current = current_score.argmax(dim=2, keepdim=True)

        next_score = future * 10 + history
        next_score = next_score.clone()
        next_score.scatter_(2, current, -torch.inf)
        next_speaker = next_score.argmax(dim=2, keepdim=True)
        speakers = torch.cat([current, next_speaker], dim=2)
        selected = torch.gather(
            bins,
            2,
            speakers.unsqueeze(-1).expand(-1, -1, -1, self.n_bins),
        )
        patterns = self._binary_index(selected)
        return self.pair_indices.to(patterns.device)[patterns[..., 0], patterns[..., 1]]

    def speaker_based_projection(self, bins):
        if bins.shape[2] != 2:
            raise ValueError("speaker_based projection requires exactly two speakers")
        future = bins[..., self.num_hist_bins :]
        return self._binary_index(future.flatten(start_dim=2))

    @torch.no_grad()
    def get_labels(self, activity):
        if activity.ndim != 3:
            raise ValueError("activity must have shape [batch, speakers, frames]")

        batch, speakers, frames = activity.shape
        bins = self.extract_bins(activity.transpose(1, 2))
        if self.mode == "role_relative":
            return self.role_relative_projection(bins)
        if self.mode == "role_future":
            return self.role_future_projection(bins)
        if self.mode == "speaker_based":
            return self.speaker_based_projection(bins)
        if self.mode == "independent":
            if speakers != 1:
                raise ValueError(
                    f"{self.mode} projection expects one speaker per batch item"
                )
            return bins.reshape(batch, frames, self.n_bins)
        raise RuntimeError("unreachable projection mode")

    def decode(self, indices):
        if self.codebook is None:
            raise ValueError(f"{self.mode} mode does not use a discrete codebook")
        values = self.codebook.to(indices.device)[indices]
        return values.view(*indices.shape, 2, self.n_future_bins)

    @property
    def n_future_bins(self):
        """Bins the codebook actually encodes per speaker."""
        if self.mode in {"role_relative", "role_future"}:
            return 4
        return self.n_bins - self.num_hist_bins

    def aggregate_probs(self, probs, start_bin, end_bin):
        """Collapse class probabilities onto two speaker slots.

        `role_relative` classes name roles, so the two slots are already
        (hold, shift). `speaker_based` classes name channels, so the slots are
        (channel 0, channel 1) and the caller must say which channel held the
        floor to read hold and shift off them.
        """
        if self.mode == "role_future":
            raise ValueError(
                "role_future classes do not preserve current-speaker identity; use an embedding probe"
            )
        indices = torch.arange(self.n_classes, device=probs.device)
        states = self.decode(indices)
        if self.mode == "role_relative":
            history = states[:, :, : self.num_hist_bins].sum(dim=-1)
            future = states[:, :, start_bin : end_bin + 1].sum(dim=-1)
            current = (history * 10 + future).argmax(dim=-1, keepdim=True)
            hold = future.gather(1, current).squeeze(1)
            shift = future.sum(dim=1) - hold
            activity = torch.stack([hold, shift], dim=-1)
        else:
            activity = states[:, :, start_bin : end_bin + 1].sum(dim=-1)
        aggregated = torch.einsum("...d,dc->...c", probs, activity)
        return aggregated / aggregated.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _default_future_window(self):
        if self.mode == "role_relative":
            return self.num_hist_bins, self.n_bins - 1
        # Original VAP reads shift off the last future bin alone.
        return self.n_future_bins - 1, self.n_future_bins - 1

    def get_probs(self, logits, start_bin=None, end_bin=None, shift_scale=1.0, prev_speaker=None):
        """Shift probability and its 0.5-threshold prediction for one setup."""
        default_start, default_end = self._default_future_window()
        start_bin = default_start if start_bin is None else start_bin
        end_bin = default_end if end_bin is None else end_bin

        probs = logits.softmax(dim=-1)
        aggregated = self.aggregate_probs(probs, start_bin, end_bin)

        if self.mode == "speaker_based":
            if prev_speaker is None:
                raise ValueError(
                    "speaker_based classes name channels, not roles; pass prev_speaker "
                    "so hold and shift can be read off the right channel"
                )
            prev = prev_speaker.to(device=aggregated.device).long().unsqueeze(-1)
            hold = aggregated.gather(-1, prev)
            shift = aggregated.gather(-1, 1 - prev)
            aggregated = torch.cat([hold, shift], dim=-1)

        scaled = aggregated.clone()
        scaled[..., 1] = scaled[..., 1] * shift_scale
        scaled = scaled / scaled.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        shift_prob = scaled[..., 1]
        return {
            "probs": probs,
            "pred": (shift_prob > 0.5).long(),
            "p_shift": shift_prob,
        }

    def get_shift_hold(self, logits, shift_scale=None, prev_speaker=None):
        if shift_scale is None:
            shift_scale = 2.0 if self.mode == "role_relative" else 1.0
        return self.get_probs(logits, shift_scale=shift_scale, prev_speaker=prev_speaker)
