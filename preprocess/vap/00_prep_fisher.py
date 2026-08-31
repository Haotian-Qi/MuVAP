"""Cut Fisher into fixed segments of stereo audio beside their raw VAD.

Class labels are *not* baked in here. Each segment stores the two-speaker VAD
padded with `--context-sec` of extra frames on both sides, which is everything
`ProjectionWindow` needs to produce role-based or original-VAP labels in the
dataloader. Switching projection setups therefore never requires re-running
this script.
"""

import argparse
import os
from dataclasses import dataclass
from glob import glob

import numpy as np
from tqdm import tqdm

from config.setup import init_yaml_config
from data.utils import load_audio, load_fisher_vad
from preprocess.vap.utils import (
    SAMPLE_RATE,
    TEST_GROUPS,
    VAL_GROUPS,
    ProcessConfig,
    get_dataset_split,
)

SEG_SEC: int = 30
OVERLAP: int = 10
VAD_HZ: int = 25
CONTEXT_SEC: float = 2.0

STRIDE_SEC: int = SEG_SEC - OVERLAP
SEG_SAMPLES: int = int(SEG_SEC * SAMPLE_RATE)
STRIDE_SAMPLES: int = int(STRIDE_SEC * SAMPLE_RATE)
SEG_VAD_FRAMES: int = int(SEG_SEC * VAD_HZ)


@dataclass()
class SegmentMeta:
    part: str
    group: str
    cid: str
    seg_id: int
    start_t: float
    end_t: float
    nframes: int


def format_meta(s: SegmentMeta) -> str:
    return f"{s.part} {s.group} {s.cid} {s.seg_id} {s.start_t:.2f} {s.end_t:.2f} {s.nframes}\n"


def extract_and_save_chunks(
    audio: np.ndarray,
    vad: np.ndarray,
    part: str,
    group: str,
    cid: str,
    cfg: ProcessConfig,
    context_frames: int,
):
    total_samples = audio.shape[-1]
    total_vad_frames = vad.shape[-1]

    audio_out = os.path.join(cfg.fisher_path, part, "seg", "audio", group, cid)
    vad_out = os.path.join(cfg.fisher_path, part, "seg", "vad", group, cid)
    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(vad_out, exist_ok=True)

    segments = []
    start_sample = 0
    seg_idx = 0

    while start_sample + SEG_SAMPLES <= total_samples:
        end_sample = start_sample + SEG_SAMPLES

        chunk_audio = audio[:, start_sample:end_sample]
        pcm16 = np.round(np.clip(chunk_audio, -1.0, 1.0) * 32767).astype(np.int16)
        audio_file = os.path.join(audio_out, f"{seg_idx}.npy")
        if not os.path.exists(audio_file):
            np.save(audio_file, pcm16)

        curr_vad_start = int((start_sample / SAMPLE_RATE) * VAD_HZ)
        curr_vad_end = curr_vad_start + SEG_VAD_FRAMES

        ext_start = max(0, curr_vad_start - context_frames)
        ext_end = min(total_vad_frames, curr_vad_end + context_frames)

        chunk_vad = vad[:, ext_start:ext_end]

        pad_left = max(0, context_frames - curr_vad_start)
        pad_right = max(0, (curr_vad_end + context_frames) - total_vad_frames)

        if pad_left > 0 or pad_right > 0:
            chunk_vad = np.pad(
                chunk_vad, ((0, 0), (pad_left, pad_right)), mode="constant"
            )

        vad_file = os.path.join(vad_out, f"{seg_idx}.npy")
        if not os.path.exists(vad_file):
            np.save(vad_file, chunk_vad.astype(np.float32))

        segments.append(
            SegmentMeta(
                part=part,
                group=group,
                cid=cid,
                seg_id=seg_idx,
                start_t=start_sample / SAMPLE_RATE,
                end_t=end_sample / SAMPLE_RATE,
                nframes=total_vad_frames,
            )
        )

        start_sample += STRIDE_SAMPLES
        seg_idx += 1

    return segments


def process_entry(
    part: str,
    group: str,
    cid: str,
    cfg: ProcessConfig,
    context_frames: int,
):
    audio_path = os.path.join(cfg.fisher_path, part, "audio", group, f"{cid}.wav")
    vad_dir = os.path.join(cfg.fisher_path, part, "regions_vap", group)

    if not os.path.exists(audio_path):
        return []

    # Both channels are kept: the original-VAP setup needs one per speaker and
    # the mono setups downmix in the dataloader.
    audio_tensor, duration = load_audio(audio_path, mono=False)
    audio = audio_tensor.numpy()

    nframes = int(round(duration * VAD_HZ))
    vad_tensor = load_fisher_vad(vad_dir, cid, nframes)
    vad = vad_tensor.numpy()

    return extract_and_save_chunks(audio, vad, part, group, cid, cfg, context_frames)


def run_pipeline(cfg: ProcessConfig, context_frames: int):
    pattern = os.path.join(cfg.fisher_path, "*", "regions_vap", "*", "*_words.txt")
    all_files = glob(pattern)

    entries = {"train": [], "val": [], "test": []}
    seen = set()

    for vad_file in tqdm(all_files, desc="Building Dataset"):
        parts = vad_file.split(os.sep)
        part, group = parts[-4], parts[-2]
        cid = "_".join(os.path.basename(vad_file).split("_")[:3])

        if (part, group, cid) in seen:
            continue
        seen.add((part, group, cid))

        split = get_dataset_split(group, cfg)
        if split not in cfg.active_splits:
            continue

        segments = process_entry(part, group, cid, cfg, context_frames)
        entries[split].extend(format_meta(s) for s in segments)

    for split in cfg.active_splits:
        lines = entries.get(split, [])
        if lines:
            out_file = os.path.join(cfg.fisher_path, f"{split}.txt")
            with open(out_file, "w") as f:
                f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/yaml/vap.yaml")
    parser.add_argument(
        "--context-sec",
        type=float,
        default=CONTEXT_SEC,
        help="VAD kept on each side of a segment; must cover the widest projection window",
    )
    args = parser.parse_args()
    yaml_cfg = init_yaml_config(args.config)

    config = ProcessConfig(
        fisher_path=yaml_cfg["fisher_path"],
        val_groups=VAL_GROUPS,
        test_groups=TEST_GROUPS,
        active_splits={"train", "val", "test"},
    )
    run_pipeline(config, int(round(args.context_sec * VAD_HZ)))
