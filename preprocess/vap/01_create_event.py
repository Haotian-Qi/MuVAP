import argparse
import os
from dataclasses import dataclass
from glob import glob
from typing import List, Set, Tuple

from tqdm import tqdm

from config.setup import init_yaml_config
from preprocess.vap.utils import print_dataset_summary


@dataclass
class LogicConfig:
    pause_threshold: float = 0.1
    long_pause_threshold: float = 1.0
    max_pause_length: float = 3.0
    active_window: float = 0.5
    purity_win: float = 1.0
    min_speech_density: float = 1.0


@dataclass
class SpeechSegment:
    start_t: float
    end_t: float
    speaker: str


@dataclass
class TurnEvent:
    part: str
    group: str
    conv_id: str
    region_start: float
    region_end: float
    prev_spk: str
    next_spk: str
    label: str
    timing: str
    gap_type: str


def parse_vad_file(file_path: str, speaker_id: str):
    segments = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                segments.append(
                    SpeechSegment(float(parts[0]), float(parts[1]), speaker_id)
                )
    return segments


def merge_into_ipus(segments: List[SpeechSegment], cfg: LogicConfig):
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.start_t)
    ipus = []

    first = sorted_segs[0]
    c_start, c_end, c_spk = first.start_t, first.end_t, first.speaker

    for seg in sorted_segs[1:]:
        if seg.speaker == c_spk and (seg.start_t - c_end) <= cfg.pause_threshold:
            c_end = max(c_end, seg.end_t)
        else:
            ipus.append(SpeechSegment(c_start, c_end, c_spk))
            c_start, c_end, c_spk = seg.start_t, seg.end_t, seg.speaker

    ipus.append(SpeechSegment(c_start, c_end, c_spk))
    return ipus


def get_pure_speaker(
    ipus: List[SpeechSegment], win_start: float, win_end: float, min_density: float
):
    speaker_time = {}

    for ipu in ipus:
        if ipu.end_t <= win_start:
            continue
        if ipu.start_t >= win_end:
            break

        o_start = max(ipu.start_t, win_start)
        o_end = min(ipu.end_t, win_end)

        if o_end > o_start:
            speaker_time[ipu.speaker] = speaker_time.get(ipu.speaker, 0.0) + (
                o_end - o_start
            )

    active_speakers = [spk for spk, duration in speaker_time.items() if duration > 0]

    if len(active_speakers) == 1:
        spk = active_speakers[0]
        if speaker_time[spk] >= min_density:
            return spk

    return None


def get_mutual_silences(ipus: List[SpeechSegment]):
    if not ipus:
        return []

    sorted_all = sorted(ipus, key=lambda x: x.start_t)
    merged_speech = []

    c_s, c_e = sorted_all[0].start_t, sorted_all[0].end_t
    for ipu in sorted_all[1:]:
        if ipu.start_t <= c_e:
            c_e = max(c_e, ipu.end_t)
        else:
            merged_speech.append((c_s, c_e))
            c_s, c_e = ipu.start_t, ipu.end_t
    merged_speech.append((c_s, c_e))

    gaps = []
    for i in range(len(merged_speech) - 1):
        gaps.append((merged_speech[i][1], merged_speech[i + 1][0]))

    return gaps


def get_overlaps(ipus: List[SpeechSegment]):
    overlaps = []
    sorted_ipus = sorted(ipus, key=lambda x: x.start_t)
    active = []

    for ipu in sorted_ipus:
        active = [a for a in active if a.end_t > ipu.start_t]

        for a in active:
            if a.speaker != ipu.speaker:
                o_start = ipu.start_t
                o_end = min(a.end_t, ipu.end_t)
                if o_end > o_start:
                    overlaps.append((o_start, o_end))
        active.append(ipu)

    return overlaps


def extract_events(
    ipus: List[SpeechSegment], part: str, group: str, conv_id: str, cfg: LogicConfig
):
    events = []
    sorted_ipus = sorted(ipus, key=lambda s: s.start_t)

    gaps = get_mutual_silences(sorted_ipus)

    for g_start, g_end in gaps:
        g_dur = g_end - g_start

        prev_spk = get_pure_speaker(
            sorted_ipus, g_start - cfg.purity_win, g_start, cfg.min_speech_density
        )
        next_spk = get_pure_speaker(
            sorted_ipus, g_end, g_end + cfg.purity_win, cfg.min_speech_density
        )

        if prev_spk is not None and next_spk is not None:
            label = "HOLD" if prev_spk == next_spk else "SHIFT"
            g_type = "LONG" if g_dur >= cfg.long_pause_threshold else "SHORT"

            if g_dur <= cfg.max_pause_length:
                events.append(
                    TurnEvent(
                        part,
                        group,
                        conv_id,
                        g_start,
                        g_end,
                        prev_spk,
                        next_spk,
                        label,
                        "SILENT",
                        g_type,
                    )
                )

            region_start = g_start - cfg.active_window
            region_end = g_start
            predicted_gap_len = g_dur + cfg.active_window

            if predicted_gap_len <= cfg.max_pause_length:
                events.append(
                    TurnEvent(
                        part,
                        group,
                        conv_id,
                        region_start,
                        region_end,
                        prev_spk,
                        next_spk,
                        label,
                        "ACTIVE",
                        g_type,
                    )
                )

    return events


def collect_vad(fisher_path: str, part: str):
    seen: Set[Tuple[str, str]] = set()
    conv = []
    pattern = os.path.join(fisher_path, part, "regions_vap", "*", "*_words.txt")

    for vad_file in glob(pattern):
        group = vad_file.split(os.sep)[-2]
        file_name = os.path.basename(vad_file)
        if not file_name.endswith("_words.txt"):
            continue

        conv_id = "_".join(file_name.split("_")[:3])
        key = (group, conv_id)
        if key not in seen:
            seen.add(key)
            conv.append(key)

    return conv


def run_pipeline(fisher_path: str):
    cfg = LogicConfig()
    lines = []
    all_events = []

    for partition in ["p1", "p2"]:
        conversations = collect_vad(fisher_path, partition)

        for group, conv_id in tqdm(conversations, desc=f"Gen labels {partition}"):
            vad_dir = os.path.join(fisher_path, partition, "regions_vap", group)
            vad_files = sorted(glob(os.path.join(vad_dir, f"{conv_id}_*_words.txt")))
            if not vad_files:
                continue

            all_ipus = []
            for i, file in enumerate(vad_files):
                raw_segments = parse_vad_file(file, str(i))
                all_ipus.extend(merge_into_ipus(raw_segments, cfg))

            events = extract_events(all_ipus, partition, group, conv_id, cfg)
            all_events.extend(events)

            for e in events:
                lines.append(
                    f"{e.part} {e.group} {e.conv_id} {e.region_start:.2f} {e.region_end:.2f} "
                    f"{e.prev_spk} {e.next_spk} {e.timing} {e.gap_type} {e.label}"
                )

    out_path = os.path.join(fisher_path, "turn_events.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print_dataset_summary(all_events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Fisher turn events")
    parser.add_argument("--config", default="config/yaml/vap.yaml")
    args = parser.parse_args()
    run_pipeline(init_yaml_config(args.config)["fisher_path"])
