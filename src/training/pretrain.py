"""
Stage 1 of BC-ResNet training: Speech Commands v0.02, 36 classes, the paper's
recipe at the 2 s window. Keeps the best-validation epoch, scores the test
split once, and resumes from the last completed epoch.
Design: docs/superpowers/specs/2026-09-26-bcresnet-training-design.md
"""

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.noise import NoiseBank
from src.data.speech_commands import (
    CLASSES, SpeechCommandsEvalSet, SpeechCommandsTrainSet, WordCache, ensure_word_cache,
    silence_count, silence_eval_windows,
)
from src.model.bcresnet import BCResNet
from src.model.frontend import LogMel, spec_augment_params
from src.training.bcresnet_loop import predict, train_epoch
from src.training.bcresnet_recipe import (
    PRETRAIN, SAMPLE_RATE, WINDOW_S, WINDOW_SAMPLES, SgdStage, make_optimizer, make_scheduler,
)
from src.training.loso import set_seed

BEST_CHECKPOINT = "pretrain.pt"
LAST_CHECKPOINT = "pretrain_last.pt"
METRICS_FILE = "pretrain_metrics.json"
EVAL_BATCH_SIZE = 256


@dataclass(frozen=True)
class PretrainJob:
    tau: float
    seed: int
    out_dir: Path
    sc_root: Path
    cache_dir: Path
    noise_dir: Path
    device: torch.device
    stage: SgdStage = PRETRAIN
    num_workers: int = 4
    resume: bool = False


def _identity(job: PretrainJob) -> dict:
    """What a resumed run must share with the run it continues."""
    return {"tau": float(job.tau), "seed": job.seed, "stage": dataclasses.asdict(job.stage)}


def _loaders(job: PretrainJob) -> Tuple[DataLoader, DataLoader, DataLoader]:
    noise = NoiseBank.from_dir(job.noise_dir, SAMPLE_RATE)
    words = {split: WordCache(ensure_word_cache(job.sc_root, split, job.cache_dir, SAMPLE_RATE))
             for split in ("train", "validation", "test")}
    train_set = SpeechCommandsTrainSet(words["train"], noise, WINDOW_SAMPLES,
                                       silence_count(len(words["train"])))

    def eval_loader(split: str, persistent: bool) -> DataLoader:
        silence = silence_eval_windows(job.sc_root, split, WINDOW_SAMPLES, SAMPLE_RATE)
        return DataLoader(SpeechCommandsEvalSet(words[split], silence, WINDOW_SAMPLES),
                          batch_size=EVAL_BATCH_SIZE, shuffle=False,
                          num_workers=job.num_workers,
                          persistent_workers=persistent and job.num_workers > 0)

    train_loader = DataLoader(train_set, batch_size=job.stage.batch_size, shuffle=True,
                              drop_last=job.stage.drop_last, num_workers=job.num_workers,
                              persistent_workers=job.num_workers > 0,
                              generator=torch.Generator().manual_seed(job.seed))
    # Validation runs every epoch, so its workers persist; the test split is scored once.
    return train_loader, eval_loader("validation", True), eval_loader("test", False)


def _accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(preds == labels))


def _fresh_progress() -> dict:
    return {"epoch": 0, "best": {"val_acc": -1.0, "epoch": 0, "state_dict": None},
            "history": []}


def _existing(job: PretrainJob, names: Tuple[str, ...]) -> list:
    return [Path(job.out_dir) / name for name in names if (Path(job.out_dir) / name).exists()]


def _refuse_overwrite(job: PretrainJob) -> None:
    """
    An existing run is never overwritten: without resume, any run (finished or
    not); with resume but no last checkpoint to resume from, a finished one.
    """
    if not job.resume:
        existing = _existing(job, (BEST_CHECKPOINT, LAST_CHECKPOINT, METRICS_FILE))
        if existing:
            raise FileExistsError(f"{existing[0]} exists; pass --resume to continue that run, "
                                  "or choose another --out-dir")
        return
    if (Path(job.out_dir) / LAST_CHECKPOINT).exists():
        return
    finished = _existing(job, (BEST_CHECKPOINT, METRICS_FILE))
    if finished:
        raise FileExistsError(f"{finished[0]} is from a finished run, but there is no "
                              f"{LAST_CHECKPOINT} to resume from (no resume point); choose "
                              "another --out-dir, or remove the finished run's files")


def _resume(job, model, optimizer, scheduler, generator) -> dict:
    path = Path(job.out_dir) / LAST_CHECKPOINT
    if not job.resume:
        return _fresh_progress()
    if not path.exists():
        print(f"No {path} to resume from; starting fresh")
        return _fresh_progress()
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["identity"] != _identity(job):
        raise ValueError(f"{path} was trained with {checkpoint['identity']}, "
                         f"not {_identity(job)}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    generator.set_state(checkpoint["specaug_rng"])
    print(f"Resuming {path} after epoch {checkpoint['epoch']}")
    return {"epoch": checkpoint["epoch"], "best": checkpoint["best"],
            "history": checkpoint["history"]}


def _save_last(job, model, optimizer, scheduler, generator, progress: dict) -> None:
    """Written to a temporary file first, so a crash mid-save never breaks resuming."""
    path = Path(job.out_dir) / LAST_CHECKPOINT
    tmp = path.with_suffix(".tmp")
    torch.save({"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "specaug_rng": generator.get_state(),
                "identity": _identity(job), **progress}, tmp)
    tmp.replace(path)


def _train_one(job, epoch, progress, model, optimizer, scheduler, generator, logmel, spec,
               train_loader, val_loader) -> dict:
    start = time.time()
    loss, acc = train_epoch(model, train_loader, logmel, spec, optimizer, scheduler,
                            job.device, generator)
    val_acc = _accuracy(*predict(model, val_loader, logmel, job.device))
    entry = {"epoch": epoch + 1, "train_loss": loss, "train_acc": acc, "val_acc": val_acc,
             "seconds": time.time() - start}
    print(f"epoch {epoch + 1}/{job.stage.epochs}: loss {loss:.4f}  train {acc:.4f}  "
          f"val {val_acc:.4f}  ({entry['seconds']:.0f} s)")
    best = progress["best"]
    if val_acc > best["val_acc"]:
        best = {"val_acc": val_acc, "epoch": epoch + 1,
                "state_dict": {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}}
    return {"epoch": epoch + 1, "best": best, "history": progress["history"] + [entry]}


def _finish(job, model, progress, logmel, test_loader) -> dict:
    best = progress["best"]
    if best["state_dict"] is None:
        raise RuntimeError("no pretraining epoch completed")
    model.load_state_dict(best["state_dict"])
    test_acc = _accuracy(*predict(model, test_loader, logmel, job.device))
    torch.save({"model_state_dict": best["state_dict"], "tau": float(job.tau), "seed": job.seed,
                "classes": list(CLASSES), "epoch": best["epoch"], "val_acc": best["val_acc"],
                "stage": dataclasses.asdict(job.stage)}, Path(job.out_dir) / BEST_CHECKPOINT)
    metrics = {"tau": float(job.tau), "seed": job.seed, "best_epoch": best["epoch"],
               "best_val_acc": best["val_acc"], "test_acc": test_acc,
               "history": progress["history"]}
    (Path(job.out_dir) / METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    return metrics


def run_pretraining(job: PretrainJob) -> dict:
    _refuse_overwrite(job)
    set_seed(job.seed)
    Path(job.out_dir).mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = _loaders(job)
    model = BCResNet(job.tau, len(CLASSES)).to(job.device)
    optimizer = make_optimizer(model, job.stage)
    scheduler = make_scheduler(optimizer, job.stage, len(train_loader))
    logmel = LogMel().to(job.device)
    spec = spec_augment_params(job.tau, WINDOW_S)
    generator = torch.Generator().manual_seed(job.seed)
    progress = _resume(job, model, optimizer, scheduler, generator)
    for epoch in range(progress["epoch"], job.stage.epochs):
        progress = _train_one(job, epoch, progress, model, optimizer, scheduler, generator,
                              logmel, spec, train_loader, val_loader)
        _save_last(job, model, optimizer, scheduler, generator, progress)
    return _finish(job, model, progress, logmel, test_loader)
