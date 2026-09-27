"""Validate packed conversations and report per-corpus statistics.

The invariants a conversation pack has to hold beyond the ASD ones are about
the speaker axis: every stream has to agree on how many speakers there are and
how long they last, and a turn event has to name speakers the segment actually
rosters. An embedding pack is checked the same way, against the frame count its
own index claims.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from data.dataloaders.conversation import PackedConversationDataset
from data.media import WaveformStats, normalization_gain
from preprocess.muvap.schema import EVENT_KIND, MEDIA_SOURCE
from preprocess.muvap.writer import native_audio_samples


def percentile(values, points=(0, 1, 50, 95, 99, 100)) -> dict:
    return {str(point): float(np.percentile(values, point)) for point in points}


def pack_kind(root: Path) -> str:
    with (Path(root) / "dataset.json").open() as handle:
        return json.load(handle)["kind"]


def validate(roots: Path | Sequence[Path]) -> None:
    roots = [roots] if isinstance(roots, (str, Path)) else list(roots)
    kinds = {pack_kind(root) for root in roots}
    if len(kinds) > 1:
        raise ValueError(f"cannot validate {sorted(kinds)} packs together")
    kind = kinds.pop()
    dataset = PackedConversationDataset(roots, kind=kind)
    grouped = defaultdict(lambda: defaultdict(list))

    for index in range(len(dataset)):
        item = dataset[index]
        metadata = item["metadata"]
        sample_id = metadata["sample_id"]
        speakers = metadata["speakers"]
        frames = item["mask"].shape[0]
        if frames == 0:
            raise ValueError(f"{sample_id}: empty sample")
        if speakers != len(metadata["speaker_ids"]):
            raise ValueError(
                f"{sample_id}: {speakers} speakers but "
                f"{len(metadata['speaker_ids'])} speaker ids"
            )
        if len(set(metadata["speaker_ids"])) != speakers:
            raise ValueError(f"{sample_id}: duplicate speaker ids")

        for name in ("vad", "visual_mask"):
            if tuple(item[name].shape) != (speakers, frames):
                raise ValueError(
                    f"{sample_id}: {name} is {tuple(item[name].shape)}, expected "
                    f"{(speakers, frames)}"
                )
        if tuple(item["bbox"].shape) != (speakers, frames, 4):
            raise ValueError(f"{sample_id}: bbox is {tuple(item['bbox'].shape)}")
        if item["bbox"].min() < 0 or item["bbox"].max() > 1:
            raise ValueError(f"{sample_id}: bbox is not normalized to the frame")

        if dataset.source == MEDIA_SOURCE:
            visual, audio = item["visual"], item["audio"]
            if tuple(visual.shape) != (speakers, frames, 112, 112):
                raise ValueError(f"{sample_id}: visual is {tuple(visual.shape)}")
            if visual.min() < 0 or visual.max() > 255:
                raise ValueError(f"{sample_id}: invalid visual tensor")
            if audio.shape[-1] != frames * 640:
                raise ValueError(f"{sample_id}: audio does not span the faces")
            if not np.isfinite(metadata["audio_peak"]):
                raise ValueError(f"{sample_id}: invalid audio statistics")
            if metadata["audio_samples"] != native_audio_samples(
                metadata["source_frames"], metadata["source_fps"]
            ):
                raise ValueError(f"{sample_id}: stored audio does not span the faces")
        else:
            if item["vap_emb"].shape[1] != frames:
                raise ValueError(f"{sample_id}: vap stream is the wrong length")
            if tuple(item["asd_emb"].shape[:2]) != (speakers, frames):
                raise ValueError(f"{sample_id}: asd stream is the wrong shape")

        if kind == EVENT_KIND:
            event = metadata["event"]
            unknown = {event["previous"], event["following"]}.difference(
                metadata["speaker_ids"]
            )
            if unknown:
                raise ValueError(f"{sample_id}: event names unrostered {sorted(unknown)}")
            hold = event["previous"] == event["following"]
            if hold != ("HOLD" in str(event["label"]).upper()):
                raise ValueError(
                    f"{sample_id}: label {event['label']!r} disagrees with its "
                    "previous and following speakers"
                )

        stats = grouped[metadata["dataset"]]
        stats["speakers"].append(speakers)
        stats["frames"].append(frames)
        stats["speaking_ratio"].append(metadata["speaking_ratio"])
        stats["visible_ratio"].append(metadata["visible_ratio"])
        if dataset.source == MEDIA_SOURCE:
            stats["source_audio_peak"].append(metadata["audio_peak"])
            stats["source_audio_rms"].append(metadata["audio_rms"])
            stats["normalized_rms"].append(item["audio"].square().mean().sqrt().item())
            stats["normalized_peak"].append(item["audio"].abs().max().item())
            gain = normalization_gain(
                WaveformStats(
                    mean=metadata["audio_mean"],
                    rms=metadata["audio_rms"],
                    peak=metadata["audio_peak"],
                )
            )
            stats["gain_db"].append(20 * math.log10(max(gain, 1e-8)))

    report = {
        "samples": len(dataset),
        "kind": kind,
        "source": dataset.source,
        "corpora": {},
    }
    for name, stats in grouped.items():
        entry = {
            "samples": len(stats["frames"]),
            "speakers": dict(
                sorted(
                    (int(count), stats["speakers"].count(count))
                    for count in set(stats["speakers"])
                )
            ),
            "model_frames_percentiles": percentile(stats["frames"]),
            "speaking_ratio_percentiles": percentile(stats["speaking_ratio"]),
            "visible_ratio_percentiles": percentile(stats["visible_ratio"]),
        }
        if stats["normalized_rms"]:
            entry |= {
                "source_audio_peak_percentiles": percentile(stats["source_audio_peak"]),
                "source_audio_rms_percentiles": percentile(stats["source_audio_rms"]),
                "normalized_rms_percentiles": percentile(stats["normalized_rms"]),
                "normalized_peak_percentiles": percentile(stats["normalized_peak"]),
                "gain_db_percentiles": percentile(stats["gain_db"]),
            }
        report["corpora"][name] = entry
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    validate(parser.parse_args().roots)


if __name__ == "__main__":
    main()
