"""Leakage-safe train/val/test splitting."""
import random


def group_key(stem: str) -> str:
    """Group key for a label/image stem.

    Stems look like ``<gbifID>_<suffix>``; grouping by gbifID keeps
    near-duplicate photos of the same occurrence record in a single split.
    """
    return stem.split("_")[0]


def stratified_split(groups: dict, val_frac: float, test_frac: float, seed: int) -> dict:
    """Split one class's grouped items into train/val/test at the group level.

    ``groups`` maps group key -> list of items. Every item of a group lands in
    the same split. Very small pools skip val (<= 3 groups) and test
    (<= 5 groups) rather than leave train empty.
    """
    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_val = max(1, int(n * val_frac)) if n > 3 else 0
    n_test = max(1, int(n * test_frac)) if n > 5 else 0
    val_keys = set(keys[:n_val])
    test_keys = set(keys[n_val:n_val + n_test])

    out = {"train": [], "val": [], "test": []}
    for k in keys:
        split = "val" if k in val_keys else "test" if k in test_keys else "train"
        out[split].extend(groups[k])
    return out
