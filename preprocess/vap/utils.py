from collections import defaultdict
from dataclasses import dataclass
from typing import List

SAMPLE_RATE: int = 16000
VAL_GROUPS = {"000", "016", "032", "048", "096", "112"}
TEST_GROUPS = {"064", "080"}


@dataclass(slots=True)
class ProcessConfig:
    fisher_path: str
    val_groups: set[str]
    test_groups: set[str]
    active_splits: set[str]
    input_file: str = ""


def get_dataset_split(group: str, cfg: ProcessConfig):
    if group in cfg.val_groups:
        return "val"
    if group in cfg.test_groups:
        return "test"
    return "train"


def print_dataset_summary(events: List, is_turn_event=True):
    stats = defaultdict(lambda: defaultdict(int))
    categories = set()

    for e in events:
        split = get_dataset_split(
            e.group, ProcessConfig("", VAL_GROUPS, TEST_GROUPS, set())
        )

        cat = f"{e.timing} {e.label}" if is_turn_event else e.label
        categories.add(cat)
        stats[split][cat] += 1
        stats["TOTAL"][cat] += 1

    sorted_cats = sorted(list(categories))

    header = f"{'SPLIT':<10}"
    for cat in sorted_cats:
        header += f" | {cat:<15}"
    header += f" | {'TOTAL'}"

    width = len(header)
    print(f"\n{'=' * width}")
    print(f"{'DATASET EVENT SUMMARY':^{width}}")
    print(f"{'=' * width}")
    print(header)
    print(f"{'-' * width}")

    for split in ["train", "val", "test", "TOTAL"]:
        row = f"{split.upper():<10}"
        total = 0
        for cat in sorted_cats:
            val = stats[split][cat]
            total += val
            row += f" | {val:<15}"
        row += f" | {total}"
        print(row)
    print(f"{'=' * width}\n")


EVENT_SEG_LEN_SEC: int = 10
EVENT_PADDING_SEC: float = 0.1


def event_crop(
    timing: str,
    region_start: float,
    region_end: float,
    seg_len_sec: float = EVENT_SEG_LEN_SEC,
    padding_sec: float = EVENT_PADDING_SEC,
):
    """Audio window an evaluation clip covers, or None if the event has none.

    The window ends just before the decision point: a hair into the silence for
    a SILENT event, a hair before the speech ends for an ACTIVE one. This is the
    single definition of that window - `02_prep_event.py` writes clips with it
    and the raw dataloader reads the same span straight from the source audio,
    so the two cannot drift apart.
    """
    if timing == "SILENT":
        crop_end = region_start + padding_sec
        if crop_end > region_end:
            return None
    elif timing == "ACTIVE":
        crop_end = region_end - padding_sec
    else:
        return None

    crop_start = crop_end - seg_len_sec
    if crop_start < 0:
        return None
    return crop_start, crop_end
