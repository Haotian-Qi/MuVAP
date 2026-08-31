"""Positional encodings shared by every MuVAP+ transformer stack.

VAP is a causal model: a frame may only attend to frames at or before it.
Every encoding here is defined for that regime, so switching between them
changes how position is represented and never what is visible.
"""

import math

import torch
import torch.nn as nn

#: A learned absolute table is deliberately absent. It is defined only up to
#: `max_frames` and cannot extrapolate past it, and these models are trained on
#: fixed-length clips but deployed on unbounded streams -- the position it would
#: need at inference is one it never saw a gradient for. ALiBi's distance bias
#: and RoPE's rotation are both defined at any offset; sinusoidal is at least
#: closed-form rather than a lookup.
POS_ENCODINGS = ("alibi", "rope", "sinusoidal", "none")


def alibi_slopes(num_heads):
    """Per-head decay rates from the ALiBi paper (Press et al., 2022)."""

    def power_of_two_slopes(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return power_of_two_slopes(num_heads)
    closest = 2 ** math.floor(math.log2(num_heads))
    extra = alibi_slopes(2 * closest)[0::2][: num_heads - closest]
    return power_of_two_slopes(closest) + extra


def causal_alibi_bias(num_frames, slopes, device, dtype):
    """Additive `[1, heads, T, T]` mask holding both ALiBi decay and causality."""
    positions = torch.arange(num_frames, device=device, dtype=dtype)
    distance = positions.view(1, -1) - positions.view(-1, 1)
    bias = distance.view(1, 1, num_frames, num_frames) * slopes.to(
        device=device, dtype=dtype
    ).view(1, -1, 1, 1)
    causal = torch.ones(num_frames, num_frames, device=device, dtype=torch.bool).tril()
    return bias.masked_fill(~causal.view(1, 1, num_frames, num_frames), float("-inf"))


def causal_float_mask(num_queries, num_keys, device, dtype):
    """Additive `[1, 1, Tq, Tk]` mask: 0 where attendable, `-inf` ahead of the query.

    Query `t` sits at absolute frame `Tk - Tq + t` -- a query stream shorter than
    its key stream is streaming inference over a longer history -- so it may
    attend keys `<= Tk - Tq + t`. Additive rather than boolean on purpose:
    `F.scaled_dot_product_attention` reads a bool mask as "True = attend" and a
    float one as a pre-softmax bias, and mixing the two is silent.
    """
    offset = num_keys - num_queries
    queries = torch.arange(num_queries, device=device).view(-1, 1) + offset
    keys = torch.arange(num_keys, device=device).view(1, -1)
    mask = torch.zeros(num_queries, num_keys, device=device, dtype=dtype)
    return mask.masked_fill(keys > queries, float("-inf")).view(
        1, 1, num_queries, num_keys
    )


def as_additive_mask(mask, dtype):
    """A caller's mask as an additive float bias, whichever convention it used."""
    if mask.dtype == torch.bool:
        return torch.zeros_like(mask, dtype=dtype).masked_fill_(~mask, float("-inf"))
    return mask.to(dtype)


class RotaryEmbedding(nn.Module):
    """Causal rotary position embedding (Su et al., 2024).

    Rotates the query and key of every head by an angle proportional to the
    absolute frame index, which makes the attention logit depend only on the
    distance between two frames. Under a causal mask that distance is always
    non-negative, so the model sees a strictly backward-looking clock and no
    future frame can leak through the encoding.
    """

    def __init__(self, head_dim, base=10_000.0, max_frames=4096):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"rotary head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self._build_cache(max_frames, inv_freq.device, torch.float32)

    def _build_cache(self, num_frames, device, dtype):
        positions = torch.arange(num_frames, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.to(device=device))
        angles = torch.cat([angles, angles], dim=-1)
        self.cos = angles.cos().to(dtype)
        self.sin = angles.sin().to(dtype)

    def _cached(self, num_frames, device, dtype):
        stale = (
            self.cos.numel() == 0
            or self.cos.shape[0] < num_frames
            or self.cos.device != device
            or self.cos.dtype != dtype
        )
        if stale:
            self._build_cache(max(num_frames, 1), device, dtype)
        return self.cos[:num_frames], self.sin[:num_frames]

    @staticmethod
    def _rotate_half(x):
        first, second = x.chunk(2, dim=-1)
        return torch.cat([-second, first], dim=-1)

    def forward(self, query, key, offset=0):
        """Rotate `[batch, heads, frames, head_dim]` query and key in place of no-op."""
        frames = query.shape[-2] + offset
        cos, sin = self._cached(frames, query.device, query.dtype)
        q_cos, q_sin = cos[offset:], sin[offset:]
        k_cos, k_sin = cos[: key.shape[-2]], sin[: key.shape[-2]]
        query = query * q_cos + self._rotate_half(query) * q_sin
        key = key * k_cos + self._rotate_half(key) * k_sin
        return query, key


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed additive sinusoidal embedding, kept for ablation against RoPE."""

    def __init__(self, dim, max_frames=4096):
        super().__init__()
        position = torch.arange(max_frames, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / dim)
        )
        table = torch.zeros(max_frames, dim)
        table[:, 0::2] = torch.sin(position * div)
        table[:, 1::2] = torch.cos(position * div)
        self.register_buffer("table", table, persistent=False)

    def forward(self, x):
        if x.shape[1] > self.table.shape[0]:
            raise ValueError(
                f"sequence of {x.shape[1]} frames exceeds the {self.table.shape[0]}-frame table"
            )
        return x + self.table[: x.shape[1]].to(dtype=x.dtype)


def build_input_position_embedding(pos_encoding, dim, max_frames=4096):
    """Additive input-side encoding, or None when position lives in attention."""
    if pos_encoding == "sinusoidal":
        return SinusoidalPositionEmbedding(dim, max_frames)
    return None
