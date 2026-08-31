"""Validate packed samples and report per-dataset signal statistics."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from data.alignment import model_frame_count
from data.dataloaders.packed import PackedASDDataset
from data.media import WaveformStats, normalization_gain
from preprocess.asd.schema import MODEL_FPS
from preprocess.asd.writer import native_audio_samples


def percentile(values: list[float], points=(0, 1, 50, 95, 99, 100)) -> dict:
    return {str(point): float(np.percentile(values, point)) for point in points}


def validate(roots: Path | Sequence[Path]) -> None:
    dataset = PackedASDDataset(roots)
    grouped = defaultdict(lambda: defaultdict(list))

    for index in range(len(dataset)):
        audio, visual, vad, _, metadata = dataset[index]
        sample_id = metadata["sample_id"]
        expected = model_frame_count(
            metadata["source_frames"], metadata["source_fps"], MODEL_FPS
        )
        frames = visual.shape[0]
        if frames == 0:
            raise ValueError(f"{sample_id}: empty sample")
        if frames != expected:
            raise ValueError(
                f"{sample_id}: {metadata['source_frames']} source frames at "
                f"{metadata['source_fps']} fps imply {expected} model frames, "
                f"loader produced {frames}"
            )
        if vad.shape[0] != frames or audio.shape[-1] != frames * 640:
            raise ValueError(f"{sample_id}: stream-length mismatch")
        if visual.shape[1:] != (112, 112) or visual.min() < 0 or visual.max() > 255:
            raise ValueError(f"{sample_id}: invalid visual tensor")
        if not np.isfinite(metadata["audio_peak"]):
            raise ValueError(f"{sample_id}: invalid audio statistics")
        if metadata["audio_samples"] != native_audio_samples(
            metadata["source_frames"], metadata["source_fps"]
        ):
            raise ValueError(f"{sample_id}: stored audio does not span the faces")

        stats = grouped[metadata["dataset"]]
        stats["source_audio_peak"].append(metadata["audio_peak"])
        stats["source_audio_rms"].append(metadata["audio_rms"])
        stats["source_audio_mean"].append(metadata["audio_mean"])
        stats["speaking_ratio"].append(metadata["speaking_ratio"])
        stats["normalized_rms"].append(
            audio.square().mean().sqrt().item()
        )
        stats["normalized_peak"].append(audio.abs().max().item())
        gain = normalization_gain(
            WaveformStats(
                mean=metadata["audio_mean"],
                rms=metadata["audio_rms"],
                peak=metadata["audio_peak"],
            )
        )
        stats["gain_db"].append(20 * math.log10(max(gain, 1e-8)))

    report = {"samples": len(dataset), "datasets": {}}
    for name, stats in grouped.items():
        report["datasets"][name] = {
            "samples": len(stats["source_audio_peak"]),
            "source_audio_peak_percentiles": percentile(stats["source_audio_peak"]),
            "source_audio_rms_percentiles": percentile(stats["source_audio_rms"]),
            "normalized_rms_percentiles": percentile(stats["normalized_rms"]),
            "normalized_peak_percentiles": percentile(stats["normalized_peak"]),
            "gain_db_percentiles": percentile(stats["gain_db"]),
            "max_abs_source_mean": max(abs(v) for v in stats["source_audio_mean"]),
            "speaking_ratio_percentiles": percentile(stats["speaking_ratio"]),
        }
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    validate(parser.parse_args().roots)


if __name__ == "__main__":
    main()
