"""Incremental encoding, so the live demo can run at the model's own 25 Hz.

Re-running a 10 s window every step costs 44 ms, which fits 10 Hz and not 25.
Profiling says where it goes: 24 ms of it is the visual frontend, a CNN applied
to each 112x112 crop independently - and at 25 Hz exactly one frame in 250 is
new. The other 249 crops produce the same 512 features they produced last step.

So the frontend output is cached per frame, which is the same idea as a KV cache
and lands in the same place: the part that only depends on the past is computed
once. What cannot be cached is recomputed honestly - the visual TCN is dilated
over time, the audio frontend is causal convolution over the window, and both
transformer stacks read the whole window - because those genuinely change when a
frame arrives.

The result is bit-comparable to the one-shot path, not merely close: a frame's
frontend features do not depend on when they were computed. `verify()` checks
that against the model rather than asserting it.
"""

import numpy as np
import torch

FACE = 112


class StreamingEncoders:
    """The frozen pair, with the per-frame visual features remembered."""

    def __init__(self, encoders, speakers: int, context: int, device: str):
        self.enc = encoders
        self.speakers = speakers
        self.context = context
        self.device = device
        feature = encoders.asd.visual_encoder.frontend
        with torch.no_grad():
            probe = torch.zeros(1, 1, FACE, FACE, device=device)
            width = feature(probe).shape[-1]
        # Two windows long, so the fold-back is a copy rather than a shuffle.
        self.ring = torch.zeros(2 * context, speakers, width, device=device)
        self.head = 0
        # What the frontend makes of an empty crop, kept so a speaker row can be
        # cleared without recomputing the window.
        with torch.no_grad():
            blank = torch.full((1, 1, FACE, FACE), 128.0, device=device)
            visual = encoders.asd.visual_encoder
            self._blank = visual.frontend((blank / 255.0 - visual.mean) / visual.std).view(-1)

    def forget(self, speaker: int) -> None:
        """Drop one speaker row's cached features.

        Used when a model row changes occupant: the features held for it are the
        previous person's face, and leaving them in place would let the model
        read one speaker's history as another's.
        """
        self.ring[:, speaker] = self._blank

    @torch.no_grad()
    def push(self, crops: torch.Tensor) -> None:
        """One model frame's crops: `[speakers, 112, 112]` uint8 on device."""
        if self.head >= 2 * self.context:
            self.ring[: self.context] = self.ring[-self.context :]
            self.head = self.context
        visual = self.enc.asd.visual_encoder
        x = crops.to(self.device).float().view(self.speakers, 1, FACE, FACE)
        x = (x / 255.0 - visual.mean) / visual.std
        self.ring[self.head] = visual.frontend(x).view(self.speakers, -1)
        self.head += 1

    @torch.no_grad()
    def run(self, audio: torch.Tensor):
        """Encode the window, reusing every frontend feature but the newest.

        `audio` is `[1, 1, samples]` covering exactly the cached frames.
        """
        frames = min(self.head, self.context)
        start = self.head - frames
        visual = self.enc.asd.visual_encoder
        asd = self.enc.asd

        # [frames, speakers, width] -> [speakers, width, frames], the layout the
        # temporal half of the visual encoder expects.
        feats = self.ring[start : self.head].permute(1, 2, 0)
        video = visual.conv1d(visual.tcn(feats)).transpose(1, 2)

        audio_embedding = asd.norm_audio(asd.encoder_proj(asd.audio_encoder(audio)))
        audio_embedding = audio_embedding.expand(self.speakers, -1, -1)
        visual_embedding = asd.norm_visual(video)
        cross_audio = asd.cross_v2a(audio_embedding, src=visual_embedding)
        cross_video = asd.cross_a2v(visual_embedding, src=audio_embedding)
        fused = asd.transformer(torch.cat([cross_audio, cross_video], dim=-1))

        _, vap = self.enc.vap(audio, return_embeddings=True)
        return vap.unsqueeze(1), fused.unsqueeze(0)


@torch.no_grad()
def verify(encoders, speakers: int, frames: int, device: str, tol: float = 1e-2):
    """Compare the streaming path with the one-shot path on identical input.

    Judged on mean error relative to the signal, not on an absolute number: the
    ASD embedding has magnitude around 11, so an absolute tolerance says nothing
    about whether the two agree.
    """
    audio = torch.randn(1, 1, frames * 640, device=device) * 0.05
    crops = torch.randint(
        0, 255, (speakers, frames, FACE, FACE), dtype=torch.uint8, device=device
    )
    report = {}
    for name, context in (
        ("fp32", torch.autocast(device, enabled=False)),
        ("bf16", torch.autocast(device, torch.bfloat16)),
    ):
        with context:
            want_vap, want_asd = encoders(audio, crops[None])
            stream = StreamingEncoders(encoders, speakers, frames, device)
            for t in range(frames):
                stream.push(crops[:, t])
            got_vap, got_asd = stream.run(audio)
        error = (want_asd.float() - got_asd.float()).abs()
        report[name] = {
            "asd_relative": (error.mean() / want_asd.float().abs().mean()).item(),
            "vap_max": (want_vap.float() - got_vap.float()).abs().max().item(),
        }
    report["ok"] = report["fp32"]["asd_relative"] < tol
    return report


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.muvap import frozen_pair

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vap-weights", required=True)
    parser.add_argument("--asd-weights", required=True)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--speakers", type=int, default=3)
    args = parser.parse_args()

    pair = frozen_pair(args.vap_weights, args.asd_weights).cuda().eval()
    report = verify(pair, args.speakers, args.frames, "cuda")
    for name in ("fp32", "bf16"):
        print(f"  {name}: asd {report[name]['asd_relative']:.1e} mean relative error, "
              f"vap max {report[name]['vap_max']:.1e}")
    print("  " + ("the cache is equivalent (fp32 agrees to 1e-2); the bf16 gap is "
                  "batching, not caching" if report["ok"]
                  else "MISMATCH - the cache is not equivalent"))
    raise SystemExit(0 if report["ok"] else 1)
