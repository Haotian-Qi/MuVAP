import os
from glob import glob

import torch

from data.media import AudioSpec, load_waveform, slice_audio


def load_audio(
    audio_path,
    target_sr=16000,
    mono=True,
    start_time=None,
    end_time=None,
    target_samples=None,
):
    """Load canonical float PCM while preserving the clip's time origin.

    Short clips are padded on the right and long clips are cropped on the
    right. No peak, RMS, or dataset-specific volume normalization is applied.
    """
    spec = AudioSpec(sample_rate=target_sr, mono=mono)
    waveform, sr = load_waveform(audio_path, spec)
    waveform = slice_audio(
        waveform,
        sr,
        start_time=start_time,
        end_time=end_time,
        target_samples=target_samples,
    )

    duration = waveform.shape[1] / sr

    return waveform, duration


def load_fisher_vad(vad_path, conv_id, nframes, frame_hz=25):

    pattern = os.path.join(vad_path, f"{conv_id}_*_words.txt")
    vad_files = sorted(glob(pattern))

    vad_labels = []
    for vad_file in vad_files:
        labels = torch.zeros(nframes)
        with open(vad_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                start = float(parts[0])
                end = float(parts[1])
                start_idx = int(round(start * frame_hz))
                end_idx = int(round(end * frame_hz))
                start_idx = max(0, min(nframes, start_idx))
                end_idx = max(0, min(nframes, end_idx))
                labels[start_idx:end_idx] = 1
        vad_labels.append(labels)

    return torch.stack(vad_labels, dim=0)


def load_lit_state_dict(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {k.replace("model.", ""): v for k, v in state_dict.items()}
