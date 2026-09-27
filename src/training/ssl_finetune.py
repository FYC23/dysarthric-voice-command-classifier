"""
One seed of an SSL backbone on TORGO (step 2): the control stage (head
warmup, then head + top layers) on all control speakers; that checkpoint
scored on every dysarthric speaker (the "<backbone>-controls" run); then the
dysarthric fine-tune once per LOSO fold, each starting from the control
stage. No model is trained on all 8 dysarthric speakers. Hyperparameters are
fixed (src/training/ssl_recipe.py); last epoch kept.

A seed resumes (SslJob.resume) at stage/fold granularity: the control stage
and each fold are units, done once their checkpoint is saved. A crash inside
a unit redoes only that unit. Every unit seeds itself (unit_seed), so a
resumed seed follows the same random streams as an uninterrupted one with the
same num_workers: bitwise the same weights on the CPU, while GPU kernels may
still differ slightly (they do between two uninterrupted runs too).
"""

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import TORGOCommandDataset, collate_fn
from src.data.noise import NoiseBank
from src.data.segments import evaluation_clips
from src.eval.constants import DYSARTHRIC_SPEAKERS
from src.eval.cost import COST_FILE, CostProfile, profile_model, save_cost
from src.eval.io import METADATA_FILE, load_run
from src.eval.schema import Run
from src.model.architecture import SSLCommandClassifier
from src.model.backbones import (
    MASKING_OVERRIDES, BackboneSpec, load_backbone, load_feature_extractor,
)
from src.training.device import attention_for
from src.training.loso import (
    ID2LABEL, TORGO_CLASSES, balanced_class_weights, fold_predictions, save_loso_run,
    set_seed, split_fold, with_label_ids,
)
from src.training.ssl_checkpoints import (
    CONTROLS_CHECKPOINT, RUN_ID, check_lineage, check_reusable, fold_checkpoint, new_run_id,
    read_checkpoint, save_checkpoint,
)
from src.training.ssl_loop import predict, run_stage
from src.training.ssl_recipe import (
    CONTROL_FINETUNE, CONTROL_HEAD_WARMUP, DYSARTHRIC_FINETUNE, SAMPLE_RATE, WINDOW_S,
    WINDOW_SAMPLES, AdamWStage, controls_run_name, run_name, seed_dir,
)

EVAL_DIR = "eval"
EVAL_BATCH_SIZE = 16
GB = 1e9
CONTROL_UNIT = 0  # folds are units 1..8


@dataclass(frozen=True)
class SslJob:
    backbone: BackboneSpec
    seed: int
    runs_dir: Path
    noise_dir: Path
    device: torch.device
    cache_dir: Optional[Path] = None
    control_stages: Tuple[AdamWStage, ...] = (CONTROL_HEAD_WARMUP, CONTROL_FINETUNE)
    dysarthric_stage: AdamWStage = DYSARTHRIC_FINETUNE
    num_workers: int = 2
    resume: bool = False  # keep controls.pt and finished folds; train only the rest

    @property
    def out_dir(self) -> Path:
        return seed_dir(self.runs_dir, run_name(self.backbone.name), self.seed)

    @property
    def controls_out_dir(self) -> Path:
        return seed_dir(self.runs_dir, controls_run_name(self.backbone.name), self.seed)


def build_model(job: SslJob) -> SSLCommandClassifier:
    """A fresh classifier on the pretrained backbone, on the CPU; attention suits job.device."""
    backbone = load_backbone(job.backbone, job.cache_dir,
                             attn_implementation=attention_for(job.device))
    return SSLCommandClassifier(backbone, len(TORGO_CLASSES))


def ssl_cost(model: SSLCommandClassifier) -> CostProfile:
    """Params and MACs of the whole model (CNN front end included) on one 2 s window."""
    device = next(model.parameters()).device
    return profile_model(model, torch.zeros(1, WINDOW_SAMPLES, device=device), WINDOW_S,
                         note="CNN front end, every transformer layer and the head")


def unit_seed(seed: int, unit: int) -> int:
    """The seed of one unit of a seed: CONTROL_UNIT for the control stage, i for fold i."""
    return seed * 1000 + unit


def _seed_unit(job: SslJob, unit: int) -> torch.Generator:
    """Seed Python, NumPy and torch for `unit`, whatever ran before it; its DataLoader generator."""
    seed = unit_seed(job.seed, unit)
    set_seed(seed)
    return torch.Generator().manual_seed(seed)


def _refuse_finished(job: SslJob) -> None:
    """
    The LOSO run is written last, after controls.pt, so the two together mark
    a finished seed. A run.json alone was left by the removed scripts/train.py
    (same runs/hubert-large/seed<k>/ path), never by this code.
    """
    marker = job.out_dir / EVAL_DIR / METADATA_FILE
    if not marker.exists():
        return
    if (job.out_dir / CONTROLS_CHECKPOINT).exists():
        raise FileExistsError(f"{job.out_dir} holds a finished run ({marker}); "
                              "delete that seed's directory to train it again")
    raise FileExistsError(f"{marker} exists without {CONTROLS_CHECKPOINT}: it looks like "
                          "output of the removed scripts/train.py; move or delete "
                          f"{job.out_dir} before training this seed")


def _saved_checkpoints(job: SslJob) -> Tuple[str, ...]:
    """controls.pt and fold checkpoints in the seed directory (a leftover .pt.tmp is not one)."""
    controls = (CONTROLS_CHECKPOINT,) if (job.out_dir / CONTROLS_CHECKPOINT).exists() else ()
    return (*controls, *sorted(p.name for p in job.out_dir.glob("fold*_*.pt")))


def _check_existing(job: SslJob) -> None:
    """
    Before anything is loaded or written: a finished seed is never overwritten,
    resume or not, and an unfinished one is continued only with resume.
    """
    _refuse_finished(job)
    saved = _saved_checkpoints(job)
    if saved and not job.resume:
        raise FileExistsError(f"{job.out_dir} holds an unfinished seed ({', '.join(saved)}); "
                              "pass --resume to continue it, or delete that seed's directory "
                              "to start over")
    if job.resume and not saved:
        print(f"{job.out_dir}: nothing to resume; starting fresh", flush=True)


def _identity(job: SslJob, stages: Sequence[AdamWStage],
              train_speakers: Iterable[str]) -> dict:
    """What produced a checkpoint, as saved; a reused one must match it (IDENTITY_KEYS)."""
    return {"backbone": job.backbone.name, "hf_id": job.backbone.hf_id, "seed": job.seed,
            "classes": list(TORGO_CLASSES), "masking": dict(MASKING_OVERRIDES),
            "stages": [asdict(s) for s in stages],
            "train_speakers": sorted(set(train_speakers))}


def _units(job: SslJob, controls: pd.DataFrame, dysarthric: pd.DataFrame) -> dict:
    """Checkpoint name -> the identity this job saves with it: controls.pt, then each fold."""
    control_speakers = sorted(controls["speaker_id"].unique())
    stages = (*job.control_stages, job.dysarthric_stage)
    folds = {fold_checkpoint(i, speaker): _identity(
                 job, stages, [*split_fold(dysarthric, speaker)[0]["speaker_id"],
                               *control_speakers])
             for i, speaker in enumerate(sorted(dysarthric["speaker_id"].unique()), start=1)}
    return {CONTROLS_CHECKPOINT: _identity(job, job.control_stages, control_speakers), **folds}


def _reusable(job: SslJob, units: dict) -> Dict[str, dict]:
    """
    The checkpoints a resumed seed keeps instead of training their unit again,
    with their metadata. Each must match what this job would save; folds also
    need the controls.pt they started from.
    """
    if not job.resume:
        return {}
    saved = {name: check_reusable(job.out_dir / name, identity)
             for name, identity in units.items() if (job.out_dir / name).exists()}
    if saved and CONTROLS_CHECKPOINT not in saved:
        raise ValueError(f"{job.out_dir} holds {', '.join(saved)} but not the "
                         f"{CONTROLS_CHECKPOINT} they were trained from; restore it, or "
                         "delete the seed directory to start over")
    if saved:
        check_lineage(job.out_dir, saved)
    workers = sorted({str(meta.get("num_workers")) for meta in saved.values()}
                     - {str(job.num_workers)})
    if workers:
        print(f"Note: the kept checkpoints were trained with num_workers {', '.join(workers)}, "
              f"this run uses {job.num_workers}: the units trained now draw different "
              "augmentation than the original run would have", flush=True)
    if saved:
        print(f"Resuming {job.out_dir}: keeping {', '.join(saved)}", flush=True)
    return saved


def state_dict_bytes(model: torch.nn.Module) -> int:
    """Bytes of the tensors one checkpoint of `model` holds (every _save is a full state dict)."""
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def _nearest_existing(path: Path) -> Path:
    path = Path(path).absolute()
    while not path.exists():
        path = path.parent
    return path


def _check_disk_space(runs_dir: Path, needed: int) -> None:
    """Fail now, not hours in, when a seed's checkpoints cannot fit under `runs_dir`."""
    free = shutil.disk_usage(_nearest_existing(runs_dir)).free
    if needed > free:
        raise OSError(f"this seed needs {needed / GB:.1f} GB of checkpoints under {runs_dir}, "
                      f"but only {free / GB:.1f} GB free; free space or move runs/")


def _save(model: SSLCommandClassifier, path: Path, job: SslJob,
          stages: Sequence[AdamWStage], train_speakers: Iterable[str], run_id: str) -> None:
    """Weights plus what produced them, written atomically."""
    save_checkpoint({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                     **_identity(job, stages, train_speakers), RUN_ID: run_id,
                     "attn_implementation": model.backbone.config._attn_implementation,
                     "device": str(job.device), "num_workers": job.num_workers}, path)


def _restore(model: SSLCommandClassifier, path: Path) -> None:
    model.load_state_dict(read_checkpoint(path)["model_state_dict"])


def _load(job: SslJob, path: Path) -> SSLCommandClassifier:
    model = build_model(job)
    _restore(model, path)
    return model.to(job.device)


def _train(model: SSLCommandClassifier, df: pd.DataFrame, stages: Sequence[AdamWStage],
           job: SslJob, feature_extractor, noise: NoiseBank,
           generator: torch.Generator) -> None:
    """Train in place on `df` through `stages`, with the TORGO augmentation recipe."""
    weights = balanced_class_weights(df["label_id"], len(TORGO_CLASSES)).to(job.device)
    dataset = TORGOCommandDataset(df, feature_extractor, Config, max_length=WINDOW_SAMPLES,
                                  target_sr=SAMPLE_RATE, augment=True, noise=noise)
    for stage in stages:
        loader = DataLoader(dataset, batch_size=stage.batch_size, shuffle=True,
                            num_workers=job.num_workers,
                            persistent_workers=job.num_workers > 0,
                            collate_fn=collate_fn, generator=generator)
        run_stage(model, stage, loader, job.device,
                  weights if stage.class_weighted else None)


def _predict(model: SSLCommandClassifier, test_df: pd.DataFrame, job: SslJob,
             feature_extractor) -> pd.DataFrame:
    """Eval-harness rows for `test_df`: real audio only, never augmented."""
    dataset = TORGOCommandDataset(test_df, feature_extractor, Config,
                                  max_length=WINDOW_SAMPLES, target_sr=SAMPLE_RATE,
                                  augment=False)
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                        num_workers=job.num_workers, collate_fn=collate_fn)
    preds, labels = predict(model, loader, job.device)
    return fold_predictions(test_df, preds, labels, ID2LABEL)


def _control_stage(job: SslJob, model: SSLCommandClassifier, controls: pd.DataFrame,
                   dysarthric: pd.DataFrame, feature_extractor, noise: NoiseBank,
                   generator: torch.Generator, reuse: bool, run_id: str) -> None:
    """Train on the controls and save controls.pt (or reuse it); save the controls-only run."""
    path = job.out_dir / CONTROLS_CHECKPOINT
    speakers = sorted(controls["speaker_id"].unique())
    if reuse:
        print(f"Control stage: reusing {path}", flush=True)
        _restore(model, path)
    else:
        print(f"Control stage: {len(speakers)} control speakers, {len(controls)} clips",
              flush=True)
        _train(model, controls, job.control_stages, job, feature_extractor, noise, generator)
        _save(model, path, job, job.control_stages, speakers, run_id)
    frame = _predict(model, evaluation_clips(dysarthric), job, feature_extractor)
    no_dysarthric_training = {s: frozenset() for s in dysarthric["speaker_id"].unique()}
    save_loso_run([frame], no_dysarthric_training, speakers, job.seed,
                  controls_run_name(job.backbone.name), job.controls_out_dir / EVAL_DIR)


def _fold_model(job: SslJob, index: int, speaker: str, train_df: pd.DataFrame,
                control_speakers: Sequence[str], feature_extractor, noise: NoiseBank,
                reuse: bool, run_id: str) -> SSLCommandClassifier:
    """Fold `index`'s model: its checkpoint if reused, else trained from controls.pt and saved."""
    path = job.out_dir / fold_checkpoint(index, speaker)
    if reuse:
        print(f"Fold {index} (hold out {speaker}): reusing {path}", flush=True)
        return _load(job, path)
    print(f"Fold {index} (hold out {speaker}): {len(train_df)} training clips", flush=True)
    model = _load(job, job.out_dir / CONTROLS_CHECKPOINT)
    generator = _seed_unit(job, index)
    _train(model, train_df, (job.dysarthric_stage,), job, feature_extractor, noise, generator)
    _save(model, path, job, (*job.control_stages, job.dysarthric_stage),
          [*train_df["speaker_id"], *control_speakers], run_id)
    return model


def _folds(job: SslJob, controls: pd.DataFrame, dysarthric: pd.DataFrame,
           feature_extractor, noise: NoiseBank, reused: Mapping[str, dict],
           run_id: str) -> None:
    """One dysarthric fine-tune per held-out speaker, each from controls.pt; save the LOSO run."""
    control_speakers = sorted(controls["speaker_id"].unique())
    frames, fold_train = [], {}
    for i, speaker in enumerate(sorted(dysarthric["speaker_id"].unique()), start=1):
        train_df, test_df = split_fold(dysarthric, speaker)
        model = _fold_model(job, i, speaker, train_df, control_speakers, feature_extractor,
                            noise, fold_checkpoint(i, speaker) in reused, run_id)
        frames.append(_predict(model, test_df, job, feature_extractor))
        fold_train[speaker] = frozenset(train_df["speaker_id"])
        del model  # before the next fold loads its own: one model in memory at a time
    save_loso_run(frames, fold_train, control_speakers, job.seed,
                  run_name(job.backbone.name), job.out_dir / EVAL_DIR)


def _check_speakers(controls: pd.DataFrame, dysarthric: pd.DataFrame) -> None:
    """Every dysarthric speaker is a fold, and the control stage needs a control speaker."""
    found = tuple(sorted(dysarthric["speaker_id"].unique()))
    if found != DYSARTHRIC_SPEAKERS:
        raise ValueError(f"expected the dysarthric speakers {DYSARTHRIC_SPEAKERS}, got {found}")
    if controls.empty:
        raise ValueError("no control speakers: the control stage has nothing to train on")


def run_ssl_finetuning(job: SslJob, samples: pd.DataFrame) -> Tuple[Run, Run]:
    """Train one seed, or with job.resume what it left unfinished. (LOSO run, controls-only run)."""
    _check_existing(job)
    df = with_label_ids(samples)
    controls, dysarthric = df[~df["is_dysarthric"]], df[df["is_dysarthric"]]
    _check_speakers(controls, dysarthric)
    units = _units(job, controls, dysarthric)
    reused = _reusable(job, units)
    kept_controls = reused.get(CONTROLS_CHECKPOINT)
    run_id = kept_controls[RUN_ID] if kept_controls else new_run_id()  # not from the seeded RNGs
    noise = NoiseBank.from_dir(job.noise_dir, SAMPLE_RATE)
    feature_extractor = load_feature_extractor(job.backbone, job.cache_dir)
    generator = _seed_unit(job, CONTROL_UNIT)  # the control unit's seed also draws the new head
    model = build_model(job)
    _check_disk_space(job.runs_dir, (len(units) - len(reused)) * state_dict_bytes(model))
    cost = ssl_cost(model)  # before training, so a failure here costs seconds
    for name in (run_name, controls_run_name):
        save_cost(cost, Path(job.runs_dir) / name(job.backbone.name) / COST_FILE)
    job.out_dir.mkdir(parents=True, exist_ok=True)
    _control_stage(job, model.to(job.device), controls, dysarthric, feature_extractor,
                   noise, generator, kept_controls is not None, run_id)
    del model
    _folds(job, controls, dysarthric, feature_extractor, noise, reused, run_id)
    return load_run(job.out_dir / EVAL_DIR), load_run(job.controls_out_dir / EVAL_DIR)
