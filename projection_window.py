"""Voice activity projection windows: VAD frames in, one class index per frame.

A class is an ordered pair of `row_bins`-bit rows. What a row is, per mode:

* `speaker_based` - one row per channel, in channel order. The original VAP:
  2 speakers x 4 future bins = 256 classes. The rows name channels, not roles,
  so hold/shift needs the event's floor holder.
* `role_relative` - N speakers ranked down to a pair on the 1.4, 0.6, 0.6, 1.4
  window and encoded `Scurr` first, then folded onto `16 * 17 / 2 = 136`
  classes. Row 1 is the next speaker, so its `p_future` is the shift.
* `independent` - no codebook, the raw per-speaker bins.
"""

import torch
import torch.nn.functional as F

P_FUTURE_SEC = 1.4   # hold/shift is decided over the trailing 1.4s
MODES = ("role_relative", "speaker_based", "independent")


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
        mode = str(mode).lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "role_relative" and (len(bin_sec) != 4 or num_hist_bins != 2):
            raise ValueError("role_relative needs two history and two future bins")

        self.mode = mode
        self.bin_sec = list(bin_sec)
        self.frame_hz = frame_hz
        self.threshold = threshold
        self.num_hist_bins = num_hist_bins
        self.cut_center_frame = cut_center_frame
        self.bin_frames = bin_times_to_frames(self.bin_sec, frame_hz)
        self.n_bins = len(self.bin_frames)
        self.hist_frames = sum(self.bin_frames[:num_hist_bins])
        self.fut_frames = sum(self.bin_frames[num_hist_bins:])

        keeps_history = mode == "role_relative"
        self.row_bins = self.n_bins if keeps_history else self.n_bins - num_hist_bins
        self.row_hist_bins = num_hist_bins if keeps_history else 0
        self.row_sec = self.bin_sec[self.n_bins - self.row_bins :]
        self._row_sec = torch.tensor(self.row_sec)

        if self.is_role_based:
            self.codebook, self.fold = self._role_codebook()
        elif mode == "speaker_based":
            self.codebook, self.fold = self._codebook(2 * self.row_bins), None
        else:
            self.codebook, self.fold = None, None
        self.n_classes = 0 if self.codebook is None else len(self.codebook)

        self._lead = 0 if cut_center_frame or self.hist_frames == 0 else 1
        self._skip = 1 if cut_center_frame else 0
        self._offsets = self._bin_offsets(self.bin_frames)
        self._hist_sec = torch.tensor(self.bin_sec[:num_hist_bins])
        self._fut_sec = torch.tensor(self.bin_sec[num_hist_bins:])
        self._weights = 2 ** torch.arange(self.row_bins - 1, -1, -1)
        self._pair_weights = 2 ** torch.arange(2 * self.row_bins - 1, -1, -1)
        self._slots = {}

    def __repr__(self):
        return (
            f"ProjectionWindow(mode={self.mode!r}, frame_hz={self.frame_hz}, "
            f"bin_frames={self.bin_frames}, n_classes={self.n_classes})"
        )

    @property
    def is_role_based(self):
        return self.mode == "role_relative"

    # ---------------------------------------------------------------- codebook

    @staticmethod
    def _codebook(bits):
        values = torch.arange(2**bits)
        return values.unsqueeze(1).bitwise_right_shift(
            torch.arange(bits - 1, -1, -1)
        ).bitwise_and(1).float()

    def _role_codebook(self):
        """`(a, b)` and `(b, a)` share a class: 256 ordered codes -> 136.

        Indices follow the paper's enumeration, by pattern value; the pair each
        index *stores* is the ranked one, so `decode` returns `(Scurr, Snext)`
        with nothing to recompute. That works because the ranking is a function
        of the rows - each role keeps its own history bins - so a pair whose top
        row ranks below its bottom row never occurs: exactly 136 of the 256
        ordered states are reachable and the fold is a bijection.
        """
        patterns = self._codebook(self.row_bins)
        size = len(patterns)
        split = self.row_hist_bins
        rank = self._activity(patterns[:, :split], self._row_sec[:split]) * 10
        rank += self._activity(patterns[:, split:], self._row_sec[split:])

        fold = torch.empty(size * size, dtype=torch.long)
        codebook = torch.empty(size * (size + 1) // 2, 2 * self.row_bins)
        for low in range(size):
            for high in range(low, size):
                index = low * size - low * (low - 1) // 2 + high - low
                fold[low * size + high] = fold[high * size + low] = index
                top, bottom = (high, low) if rank[high] >= rank[low] else (low, high)
                codebook[index] = torch.cat([patterns[top], patterns[bottom]])
        return codebook, fold

    # ------------------------------------------------------------------ labels

    def _bin_offsets(self, bin_frames):
        offsets, start = [], 0
        for width in bin_frames:
            shift = self._lead + (self._skip if start >= self.hist_frames else 0)
            offsets.append((start + shift, start + shift + width, width))
            start += width
        return offsets

    def _totals(self, activity):
        """Running frame count; the extra left pad makes the sum exclusive."""
        return F.pad(
            activity,
            (0, 0, self.hist_frames + 1, self.fut_frames + self._lead + self._skip),
        ).cumsum(dim=1)

    def _bins(self, totals, offsets, frames, dtype):
        """A bin is two slices of the running count subtracted, any width."""
        return torch.stack(
            [
                (totals[:, end : end + frames] - totals[:, start : start + frames])
                >= self.threshold * width
                for start, end, width in offsets
            ],
            dim=-1,
        ).to(dtype)

    def extract_bins(self, activity):
        """Thresholded `[B, T, S, n_bins]` from `[B, T, S]` activity."""
        return self._bins(
            self._totals(activity), self._offsets, activity.shape[1], activity.dtype
        )

    def _activity(self, bins, seconds):
        """Seconds of activity: the 2, 1 / 1, 2 weighting of the original code."""
        return (bins * seconds.to(bins.device)).sum(dim=-1)

    def _rank_roles(self, bins):
        """History picks `Scurr`; the remaining speakers' future picks `Snext`.

        When the top two tie on history outright, stage 1's runner-up takes the
        second slot rather than the speaker with the most future.
        """
        history = self._activity(bins[..., : self.num_hist_bins], self._hist_sec)
        future = self._activity(bins[..., self.num_hist_bins :], self._fut_sec)
        by_history = history * 10 + future

        if bins.shape[2] == 2:                       # stage 2 has one candidate left
            current = (by_history[..., 1:] > by_history[..., :1]).long()
            return torch.cat([current, 1 - current], dim=2)

        by_future = future * 10 + history
        pair = torch.topk(by_history, k=2, dim=2).indices
        current, runner_up = pair[..., :1], pair[..., 1:2]
        tied = history.gather(2, current) == history.gather(2, runner_up)
        by_future.scatter_(2, current, -torch.inf)
        following = torch.where(tied, runner_up, by_future.argmax(dim=2, keepdim=True))
        return torch.cat([current, following], dim=2)

    @torch.no_grad()
    def get_labels(self, activity):
        """`[B, S, frames]` voice activity to `[B, frames]` class indices."""
        activity = activity.transpose(1, 2)
        frames = activity.shape[1]
        totals = self._totals(activity)
        bins = self._bins(totals, self._offsets, frames, activity.dtype)

        if self.mode == "independent":
            if bins.shape[2] != 1:
                raise ValueError("independent projection expects one speaker per item")
            return bins.squeeze(2)
        if bins.shape[2] < 2:
            raise ValueError(f"{self.mode} projection requires at least two speakers")

        if self.mode == "speaker_based":                 # rows are channels, in order
            pair = bins[..., self.num_hist_bins :].flatten(start_dim=2)
            return (pair * self._pair_weights.to(pair.dtype)).sum(dim=-1).long()

        # Index every row first: gathering two integers beats gathering two rows.
        patterns = (bins * self._weights.to(bins.dtype)).sum(dim=-1).long()
        speakers = self._rank_roles(bins)
        ordered = patterns.gather(2, speakers[..., :1]) * 2**self.row_bins
        ordered += patterns.gather(2, speakers[..., 1:])
        return self.fold.to(ordered.device)[ordered.squeeze(-1)]

    # ----------------------------------------------------------------- readout

    def decode(self, indices):
        """`(Scurr, Snext)` for the role modes, `(channel 1, channel 2)` otherwise."""
        values = self.codebook.to(indices.device)[indices]
        return values.view(*indices.shape, 2, self.row_bins)

    def _default_future_window(self):
        """The `p_future` bins: the trailing bins covering the last 1.4s."""
        span, start = 0.0, self.row_bins
        while start > self.row_hist_bins and span < P_FUTURE_SEC - 1e-6:
            start -= 1
            span += self.row_sec[start]
        return start, self.row_bins - 1

    def _default_now_window(self):
        """The `p_now` bins: the rest of the future, ahead of `p_future`."""
        start = self._default_future_window()[0]
        return (self.row_hist_bins, start - 1) if start > self.row_hist_bins else None

    def _rows_over(self, probs, window):
        """Class probabilities collapsed onto the two rows over one bin window."""
        if window not in self._slots:
            states = self.decode(torch.arange(self.n_classes))
            self._slots[window] = states[:, :, window[0] : window[1] + 1].sum(dim=-1)
        rows = torch.einsum("...d,dc->...c", probs, self._slots[window].to(probs.device))
        return rows / rows.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def get_probs(self, logits):
        """Row probabilities over `p_now` and `p_future`.

        The two entries are the codebook's two rows: `(Scurr, Snext)` for the
        role modes, `(speaker 1, speaker 2)` for `speaker_based`. This is the
        plain readout - turning it into hold and shift is `get_shift_hold`.
        """
        probs = logits.softmax(dim=-1)
        now = self._default_now_window()
        return {
            "probs": probs,
            "p_now": None if now is None else self._rows_over(probs, now),
            "p_future": self._rows_over(probs, self._default_future_window()),
        }

    @staticmethod
    def as_hold_shift(rows, prev_speaker):
        """Channel probabilities to (hold, shift), given who held the floor."""
        prev = prev_speaker.to(device=rows.device).long().unsqueeze(-1)
        return torch.cat([rows.gather(-1, prev), rows.gather(-1, 1 - prev)], dim=-1)

    def get_shift_hold(self, logits, shift_scale=None, prev_speaker=None):
        """Shift probability over `p_future`, and its 0.5-threshold prediction.

        The role modes rank into `(Scurr, Snext)` at encode time, so their two
        rows already are (hold, shift). `speaker_based` rows are the channels,
        and which one is the hold is a property of the event, so the caller
        supplies the floor holder.
        """
        output = self.get_probs(logits)
        rows = output["p_future"]

        if self.mode == "speaker_based":
            if prev_speaker is None:
                raise ValueError(
                    "speaker_based rows are speaker 1 and speaker 2, not hold and "
                    "shift; pass prev_speaker to say which one held the floor"
                )
            rows = self.as_hold_shift(rows, prev_speaker)

        if shift_scale is None:
            shift_scale = 2.0 if self.is_role_based else 1.0
        scaled = rows.clone()
        scaled[..., 1] *= shift_scale
        shift = scaled[..., 1] / scaled.sum(dim=-1).clamp_min(1e-8)
        return {**output, "pred": (shift > 0.5).long(), "p_shift": shift}
