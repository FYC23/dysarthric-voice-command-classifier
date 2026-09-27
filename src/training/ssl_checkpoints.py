"""
Checkpoints of one SSL seed (src/training/ssl_finetune.py): written
atomically, read with an error that names the file, and checked before a
resumed seed reuses one in place of training that unit again.
"""

import os
import uuid
from pathlib import Path
from typing import Mapping

import torch

CONTROLS_CHECKPOINT = "controls.pt"
TMP_SUFFIX = ".tmp"
RUN_ID = "run_id"  # drawn when controls.pt is trained, copied into every fold trained from it

# What a reused checkpoint must share with the one this job would write. The
# rest (device, attn_implementation, num_workers) may differ: a resumed seed
# may run on another GPU.
IDENTITY_KEYS = ("backbone", "hf_id", "seed", "classes", "masking", "stages",
                 "train_speakers")


def new_run_id() -> str:
    return uuid.uuid4().hex


def fold_checkpoint(index: int, speaker: str) -> str:
    """fold<i>_<speaker>.pt: fold `index` (1-based), never trained on `speaker`."""
    return f"fold{index}_{speaker}.pt"


def save_checkpoint(checkpoint: Mapping, path: Path) -> None:
    """
    Written to <name>.pt.tmp and then renamed, so a crash mid-save never leaves
    a complete-looking <name>.pt. A leftover .tmp is simply overwritten.
    """
    path = Path(path)
    tmp = path.with_name(path.name + TMP_SUFFIX)
    torch.save(dict(checkpoint), tmp)
    os.replace(tmp, path)


def read_checkpoint(path: Path, mmap: bool = False) -> dict:
    """
    The saved dict. `mmap` leaves the tensors on disk (enough to read the
    metadata of a large checkpoint). A file that cannot be read is kept.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", mmap=mmap)
    except Exception as error:  # truncated or corrupt: torch raises several types
        raise RuntimeError(f"cannot read {path} ({error}); a crash may have cut it short. "
                           f"{_how_to_retrain(Path(path))}") from error
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{path} does not hold a checkpoint dict")
    return checkpoint


def _how_to_retrain(path: Path) -> str:
    if path.name == CONTROLS_CHECKPOINT:  # every fold started from it
        return f"Delete the seed directory ({path.parent}) to train the seed again"
    return "Delete it to train that unit again"


def check_reusable(path: Path, expected: Mapping) -> dict:
    """
    The checkpoint's metadata, once every IDENTITY_KEYS value matches `expected`
    (what this job would write); otherwise a ValueError naming the differences.
    """
    checkpoint = read_checkpoint(path, mmap=True)
    differing = [k for k in IDENTITY_KEYS if checkpoint.get(k) != expected[k]]
    if differing:
        details = "; ".join(f"{k}: saved {checkpoint.get(k)!r}, this job {expected[k]!r}"
                            for k in differing)
        raise ValueError(f"{path} was not written by this job ({', '.join(differing)} "
                         f"differ: {details}). Delete it, or the seed directory, to "
                         "train it again")
    return {k: v for k, v in checkpoint.items() if k != "model_state_dict"}


def check_lineage(out_dir: Path, kept: Mapping[str, Mapping]) -> None:
    """
    Every kept fold was trained from the kept controls.pt (the same run_id), so
    folds copied in from another seed directory are never mixed in. `kept`
    maps checkpoint names, controls.pt among them, to their metadata.
    """
    run_id = kept[CONTROLS_CHECKPOINT].get(RUN_ID)
    if not run_id:
        raise ValueError(f"{Path(out_dir) / CONTROLS_CHECKPOINT} has no {RUN_ID}, so its folds "
                         "cannot be traced to it; delete the seed directory to start over")
    for name, meta in kept.items():
        if meta.get(RUN_ID) != run_id:
            raise ValueError(f"{Path(out_dir) / name} was not trained from this "
                             f"{CONTROLS_CHECKPOINT} ({RUN_ID} {meta.get(RUN_ID)!r}, "
                             f"{CONTROLS_CHECKPOINT} has {run_id!r}); delete it to train "
                             "that fold again")
