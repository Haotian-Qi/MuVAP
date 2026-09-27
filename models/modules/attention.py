"""Transformer blocks for the VAP, ASD, and MuVAP stacks.

The temporal stacks are causal: frame `t` attends to frames `<= t` only, which
is what makes a trained model usable as a streaming turn-taking predictor. How
position reaches attention is configurable through `pos_encoding`; the
visibility rule is not.

`FrameTemporalEncoder` is the multiparty half. It alternates two views of the
same tensor: attention *across speakers* inside one frame, where the speakers
carry no order and no position, and attention *along time* on the fused frame
sequence, where both still apply.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import (
    POS_ENCODINGS,
    RotaryEmbedding,
    alibi_slopes,
    as_additive_mask,
    build_input_position_embedding,
    causal_alibi_bias,
    causal_float_mask,
)


class TransformerModel(nn.Module):
    """Stack of pre-norm transformer layers over a `[batch, frames, dim]` sequence."""

    #: A frame-wise stack attends across speakers, whose order is arbitrary, so
    #: it builds neither a rotary embedding nor an input position table.
    uses_position = True

    def __init__(self, cfg, cross_attention=False):
        super().__init__()
        self.dim = cfg["d_model"]
        self.dff = cfg["m_ff"] * self.dim
        self.num_layers = cfg["n_layers"]
        self.num_heads = cfg["n_heads"]
        self.activation = cfg["activation"]
        self.dropout = cfg["dropout"]
        self.cross_attention = cross_attention
        self.causal = bool(cfg.get("causal", True))
        self.pos_encoding = str(cfg.get("pos_encoding", "alibi")).lower()
        self.ffn_type = str(cfg.get("ffn", "mlp")).lower()
        self.norm_type = str(cfg.get("norm", "layer")).lower()
        self.max_frames = int(cfg.get("max_frames", 4096))
        if self.pos_encoding not in POS_ENCODINGS:
            raise ValueError(
                f"pos_encoding must be one of {POS_ENCODINGS}, got {self.pos_encoding!r}"
            )

        self.rope = None
        self.input_pos = None
        if self.uses_position:
            if self.pos_encoding == "rope":
                self.rope = RotaryEmbedding(
                    self.dim // self.num_heads, max_frames=self.max_frames
                )
            self.input_pos = build_input_position_embedding(
                self.pos_encoding, self.dim, self.max_frames
            )

        self._build_layers()
        self.apply(self._init_weights)

    def _build_layers(self):
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                    cross_attention=self.cross_attention,
                    pos_encoding=self.pos_encoding,
                    causal=self.causal,
                    rope=self.rope,
                    ffn_type=self.ffn_type,
                    norm_type=self.norm_type,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, src=None, mask=None):
        if self.input_pos is not None:
            x = self.input_pos(x)
        for layer in self.layers:
            x = layer(x, src, mask=mask)
        return x


class FrameTemporalEncoder(nn.Module):
    """Fuse one global stream with a variable number of per-speaker streams.

    The two streams are told apart by a learned type embedding rather than by
    position, because they are not two points on one timeline: `vap` is the
    conversation heard as a whole, `asd` is what each visible speaker is doing.
    The embedding starts at zero, so an untrained encoder sees exactly the
    features its two frozen sources produced.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ft = FrameTemporalBlock(cfg["frame"], cfg["temporal"])

        model_dim = cfg["temporal"]["d_model"]
        self.type_embed = nn.Embedding(2, model_dim)
        nn.init.zeros_(self.type_embed.weight)

    def forward(self, vap, asd, speaker_mask=None):
        device = asd.device
        vap = vap + self.type_embed(torch.tensor(0, device=device))
        asd = asd + self.type_embed(torch.tensor(1, device=device))

        vap, asd = self.ft(vap, asd, speaker_mask=speaker_mask)

        return vap, asd


#: Folded frames handled per frame-attention call. Attention inside a frame is
#: independent of every other frame, so the split changes nothing numerically -
#: it only stops `batch x frames` from becoming one enormous attention problem.
#: A 32-minute validation segment folds to 48k frames, which overruns the CUDA
#: grid and then memory; the loop keeps each call the same size whatever arrives.
FRAME_CHUNK = 8192


class FrameTemporalBlock(nn.Module):
    """One frame-wise read across speakers, then one causal pass along time.

    Doing it in that order is what keeps the model usable on a conversation
    with any number of visible speakers: the speaker axis is consumed inside a
    frame, where it is a set, and only the fused `[batch, frames, dim]` stream
    reaches the temporal stack, where order and causality mean something.
    """

    def __init__(self, f_cfg, t_cfg):
        super().__init__()
        self.vap_frame = FrameTransformerModel(f_cfg)
        self.vap_temporal = TransformerModel(t_cfg)

    def forward(self, vap, asd, speaker_mask=None):
        B, S, T, D = asd.shape

        # Every frame of every item becomes its own one-query attention problem,
        # which is what lets one batch hold conversations of differing length
        # without the speaker axis leaking across frames.
        q_vap = vap.permute(0, 2, 1, 3).reshape(B * T, 1, D)
        kv_asd = asd.permute(0, 2, 1, 3).reshape(B * T, S, D)

        mask = None
        if speaker_mask is not None:
            if speaker_mask.shape != (B, S, T):
                raise ValueError(
                    "speaker_mask must have shape [batch, speakers, frames]"
                )
            mask = speaker_mask.permute(0, 2, 1).reshape(B * T, 1, 1, S)
            # A frame where no face is visible would otherwise attend to nothing
            # and come back NaN. Opening every key there is the honest fallback:
            # the padded rows are the model's own zeros, not another speaker.
            empty = ~mask.any(dim=-1, keepdim=True)
            mask = mask | empty

        folded = []
        for start in range(0, B * T, FRAME_CHUNK):
            stop = start + FRAME_CHUNK
            folded.append(
                self.vap_frame(
                    x=q_vap[start:stop],
                    src=kv_asd[start:stop],
                    mask=None if mask is None else mask[start:stop],
                )
            )
        vap_in = (folded[0] if len(folded) == 1 else torch.cat(folded)).reshape(B, T, D)
        vap_out = self.vap_temporal(vap_in).reshape(B, 1, T, D)

        return vap_out, asd


class FrameTransformerModel(nn.Module):
    """A `FrameTransformer` that always reads a source stream."""

    def __init__(self, cfg):
        super().__init__()
        self.transformer = FrameTransformer(cfg)

    def forward(self, x, src, mask=None):
        return self.transformer(x=x, src=src, mask=mask)


class FrameTransformer(TransformerModel):
    """Attends across speakers inside one frame, where there is no position."""

    uses_position = False

    def _build_layers(self):
        self.layers = nn.ModuleList(
            [
                FrameLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                    norm_type=self.norm_type,
                )
                for _ in range(self.num_layers)
            ]
        )


class FrameLayer(nn.Module):
    """Cross-attention over speakers within one frame; order carries no meaning.

    Unlike `TransformerLayer` this has no self-attention over the query: a
    frame holds one query and the speakers are the source, so attending the
    query to itself would add nothing.
    """

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        ffn_activation="GELU",
        dropout=0.1,
        norm_type="layer",
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln_self_attn = build_norm(norm_type, dim)
        self.ln_kv = build_norm(norm_type, dim)
        self.ln_ffnetwork = build_norm(norm_type, dim)

        self.mha = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=False,
            pos_encoding="none",
        )
        self.ffnetwork = ffn_block(
            dim, ffn_dim, activation=ffn_activation, dropout=dropout
        )

    def forward(self, x, src=None, mask=None):
        z = self.ln_self_attn(x)
        src_norm = self.ln_kv(src)

        x = x + self.dropout(self.mha(Q=z, K=src_norm, V=src_norm, mask=mask))
        x = x + self.dropout(self.ffnetwork(self.ln_ffnetwork(x)))
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim=256,
        ffn_dim=768,
        num_heads=4,
        ffn_activation="GELU",
        dropout=0.1,
        cross_attention=False,
        pos_encoding="alibi",
        causal=True,
        rope=None,
        ffn_type="mlp",
        norm_type="layer",
    ):
        super().__init__()
        self.cross_attention = cross_attention
        self.dropout = nn.Dropout(dropout)

        self.ln_self_attn = build_norm(norm_type, dim)
        self.mha = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
            pos_encoding=pos_encoding,
            rope=rope,
        )

        if cross_attention:
            self.ln_cross_query = build_norm(norm_type, dim)
            self.ln_cross_source = build_norm(norm_type, dim)
            self.mha_cross = MultiHeadAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                causal=causal,
                pos_encoding=pos_encoding,
                rope=rope,
            )

        self.ln_ffnetwork = build_norm(norm_type, dim)
        self.ffnetwork = ffn_block(
            dim, ffn_dim, activation=ffn_activation, dropout=dropout, ffn_type=ffn_type
        )

    def forward(self, x, src=None, mask=None):
        z = self.ln_self_attn(x)
        x = x + self.dropout(self.mha(Q=z, K=z, V=z, mask=mask))

        if self.cross_attention and src is not None:
            query = self.ln_cross_query(x)
            source = self.ln_cross_source(src)
            x = x + self.dropout(self.mha_cross(Q=query, K=source, V=source, mask=mask))

        x = x + self.dropout(self.ffnetwork(self.ln_ffnetwork(x)))
        return x


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with a selectable positional encoding.

    `alibi` adds a per-head linear distance penalty to the logits, `rope`
    rotates the query and key instead, and every other setting leaves position
    to the input embedding. Causality is applied the same way in all cases.
    """

    def __init__(
        self,
        dim,
        num_heads,
        dropout,
        bias=False,
        causal=False,
        pos_encoding="none",
        rope=None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.pos_encoding = pos_encoding
        self.dropout_p = dropout
        self.rope = rope if pos_encoding == "rope" else None

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

        self.resid_drop = nn.Dropout(dropout)

        if pos_encoding == "alibi":
            self.register_buffer(
                "slopes", torch.tensor(alibi_slopes(num_heads)), persistent=False
            )
            self.register_buffer("alibi_bias", torch.empty(0), persistent=False)

    def _alibi_mask(self, num_frames, device, dtype):
        stale = (
            self.alibi_bias.numel() == 0
            or self.alibi_bias.shape[-1] < num_frames
            or self.alibi_bias.device != device
            or self.alibi_bias.dtype != dtype
        )
        if stale:
            self.alibi_bias = causal_alibi_bias(num_frames, self.slopes, device, dtype)
        return self.alibi_bias[..., :num_frames, :num_frames]

    def forward(self, Q, K, V, mask=None):
        B, T, C = Q.size()
        Tk = K.size(1)

        q = self.q_proj(Q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(K).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(V).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            # Query t sits at absolute frame Tk - T + t, so a query shorter than
            # its key stream (streaming inference) still lines up with history.
            q, k = self.rope(q, k, offset=Tk - T)

        # The causal mask is built on EVERY call when this layer is causal, and a
        # caller's mask is *added* to it -- never substituted for it. Building it
        # only when `mask is None` (as this did) meant that supplying any mask
        # dropped both the causal bias and `is_causal`, leaving a silently
        # bidirectional layer that still trained and still reported numbers. No
        # caller passes a mask into a temporal stack today, so the defect was
        # latent; it goes live the first time anyone adds padded batches, which
        # is exactly how the sibling ASD codebase found it.
        if self.causal and self.pos_encoding == "alibi" and T == Tk:
            causal_bias = self._alibi_mask(T, q.device, q.dtype)
        elif self.causal and mask is not None:
            # ALiBi's helper only covers T == Tk; fall back to plain causality.
            causal_bias = causal_float_mask(T, Tk, q.device, q.dtype)
        else:
            causal_bias = None

        if causal_bias is not None:
            mask = (
                causal_bias
                if mask is None
                else causal_bias + as_additive_mask(mask, q.dtype)
            )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.causal and mask is None,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


def build_norm(norm_type, dim):
    if norm_type == "rms":
        return RMSNorm(dim)
    if norm_type == "layer":
        return nn.LayerNorm(dim)
    raise ValueError(f"norm must be 'layer' or 'rms', got {norm_type!r}")


class SwiGLU(nn.Module):
    def __init__(self, din, dff, dropout=0.0, bias=False):
        super().__init__()
        self.gate = nn.Linear(din, dff, bias=bias)
        self.up = nn.Linear(din, dff, bias=bias)
        self.down = nn.Linear(dff, din, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


def ffn_block(din, dff, activation="GELU", dropout=0.0, bias=False, ffn_type="mlp"):
    if ffn_type == "swiglu":
        return SwiGLU(din, dff, dropout=dropout, bias=bias)
    if ffn_type != "mlp":
        raise ValueError(f"ffn must be 'mlp' or 'swiglu', got {ffn_type!r}")
    return nn.Sequential(
        nn.Linear(din, dff, bias=bias),
        getattr(nn, activation)(),
        nn.Dropout(dropout),
        nn.Linear(dff, din, bias=bias),
    )
