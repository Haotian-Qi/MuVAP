import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from config.setup import init_yaml_config
from data.utils import load_audio
from preprocess.vap.utils import (
    EVENT_PADDING_SEC,
    EVENT_SEG_LEN_SEC,
    SAMPLE_RATE,
    TEST_GROUPS,
    VAL_GROUPS,
    ProcessConfig,
    event_crop,
    get_dataset_split,
    print_dataset_summary,
)

SEG_LEN_SEC: int = EVENT_SEG_LEN_SEC
PADDING_SEC: float = EVENT_PADDING_SEC


AUDIO_DIRS = {1: "audio", 2: "audio_stereo"}


@dataclass(slots=True)
class TurnEvent:
    part: str
    group: str
    conv_id: str
    region_start: float
    region_end: float
    prev_spk: str
    next_spk: str
    timing: str
    gap_type: str
    label: str
    crop_start: float = 0.0
    crop_end: float = 0.0


def parse_line(line: str) -> TurnEvent:
    tokens = line.strip().split()
    if len(tokens) != 10:
        return None

    entry = TurnEvent(
        part=tokens[0],
        group=tokens[1],
        conv_id=tokens[2],
        region_start=float(tokens[3]),
        region_end=float(tokens[4]),
        prev_spk=tokens[5],
        next_spk=tokens[6],
        timing=tokens[7],
        gap_type=tokens[8],
        label=tokens[9],
    )

    crop = event_crop(entry.timing, entry.region_start, entry.region_end)
    if crop is None:
        return None
    entry.crop_start, entry.crop_end = crop

    return entry


def group_by_conversation(entries: List[TurnEvent]):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[(entry.part, entry.group, entry.conv_id)].append(entry)
    return dict(grouped)


def format_output_line(entry: TurnEvent):
    location = f"{entry.part} {entry.group} {entry.conv_id} {entry.region_start:.2f} {entry.region_end:.2f}"
    event = f"{entry.prev_spk} {entry.next_spk} {entry.timing} {entry.gap_type} {entry.label}"
    return f"{location} {event}"


def load_entries_from_disk(cfg: ProcessConfig):
    splits = {"train": [], "val": [], "test": []}
    path = os.path.join(cfg.fisher_path, cfg.input_file)

    with open(path, "r") as f:
        parsed_entries = (parse_line(line) for line in f)

        valid_entries = (e for e in parsed_entries if e is not None)

        for entry in valid_entries:
            split_name = get_dataset_split(entry.group, cfg)
            if split_name in cfg.active_splits:
                splits[split_name].append(entry)

    return splits


def save_segment(entry: TurnEvent, audio: np.ndarray, cfg: ProcessConfig, channels: int):
    seg_samples = int(SEG_LEN_SEC * SAMPLE_RATE)
    start_sample = int(entry.crop_start * SAMPLE_RATE)
    end_sample = start_sample + seg_samples

    if end_sample > audio.shape[1]:
        return False

    seg = audio[:, start_sample:end_sample]
    if seg.shape[1] != seg_samples:
        return False

    out_dir = os.path.join(
        cfg.fisher_path,
        entry.part,
        "tune",
        AUDIO_DIRS[channels],
        entry.group,
        entry.conv_id,
    )
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{entry.region_start:.2f}.npy")
    if not os.path.exists(out_path):
        pcm16 = (seg.clamp(-1.0, 1.0) * 32767).round().to(torch.int16)
        np.save(out_path, pcm16.numpy())

    return True


def process_split(
    split_name: str, entries: List[TurnEvent], cfg: ProcessConfig, channels: int
):
    if not entries:
        return [], []

    out_lines: List[str] = []
    successful_events: List[TurnEvent] = []
    grouped_convos = group_by_conversation(entries)

    for (part, group, cid), convo_entries in tqdm(
        grouped_convos.items(), desc=f"Processing {split_name}"
    ):
        audio_sub_dir = "p1" if part == "p1" else "p2"
        audio_path = os.path.join(
            cfg.fisher_path, audio_sub_dir, "audio", group, f"{cid}.wav"
        )

        if not os.path.exists(audio_path):
            continue

        audio, _ = load_audio(audio_path, mono=channels == 1)

        for entry in convo_entries:
            if save_segment(entry, audio, cfg, channels):
                out_lines.append(format_output_line(entry))
                successful_events.append(entry)

    return out_lines, successful_events


def run_pipeline(cfg: ProcessConfig, channels: int = 1):
    splits = load_entries_from_disk(cfg)
    all_saved_events = []

    for split_name in cfg.active_splits:
        entries = splits.get(split_name, [])
        if not entries:
            continue

        out_lines, success_events = process_split(split_name, entries, cfg, channels)
        all_saved_events.extend(success_events)

        # The manifest lists which events have audio, so it is written once per
        # channel layout and stays identical between them.
        if out_lines:
            out_path = os.path.join(cfg.fisher_path, f"{split_name}_events.txt")
            with open(out_path, "w") as f:
                f.write("\n".join(out_lines) + "\n")

    if all_saved_events:
        print_dataset_summary(all_saved_events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/yaml/vap.yaml")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["val", "test"],
        help="splits to cut clips for; training reads segments, not events, so "
        "train clips are only needed to enlarge the probe's tuning pool",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 2),
        default=1,
        help="1 writes tune/audio for the mono setups, 2 writes tune/audio_stereo for original VAP",
    )
    args = parser.parse_args()
    yaml_cfg = init_yaml_config(args.config)

    config = ProcessConfig(
        fisher_path=yaml_cfg["fisher_path"],
        input_file="turn_events.txt",
        val_groups=VAL_GROUPS,
        test_groups=TEST_GROUPS,
        active_splits=set(args.splits),
    )

    run_pipeline(config, channels=args.channels)
