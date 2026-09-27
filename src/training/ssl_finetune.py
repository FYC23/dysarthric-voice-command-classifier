"""
One seed of an SSL backbone on TORGO (step 2): the control stage (head
warmup, then head + top layers) on all control speakers; that checkpoint
scored on every dysarthric speaker (the "<backbone>-controls" run); then the
dysarthric fine-tune once per LOSO fold, each starting from the control
stage. No model is trained on all 8 dysarthric speakers. Hyperparameters are
fixed (src/training/ssl_recipe.py); last epoch kept.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import TORGOCommandDataset, collate_fn
from src.data.noise import NoiseBank
from src.data.segments import evaluation_clips
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
from src.training.ssl_loop import predict, run_stage
from src.training.ssl_recipe import (
    CONTROL_FINETUNE, CONTROL_HEAD_WARMUP, DYSARTHRIC_FINETUNE, SAMPLE_RATE, WINDOW_S,
    WINDOW_SAMPLES, AdamWStage, controls_run_name, run_name, seed_dir,
)

CONTROLS_CHECKPOINT = "controls.pt"
EVAL_DIR = "eval"
EVAL_BATCH_SIZE = 16


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


def _save(model: SSLCommandClassifier, path: Path, job: SslJob,
          stages: Sequence[AdamWStage], train_speakers: Iterable[str]) -> None:
    """Weights plus what produced them."""
    torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "backbone": job.backbone.name, "hf_id": job.backbone.hf_id,
                "seed": job.seed, "classes": list(TORGO_CLASSES),
                "masking": dict(MASKING_OVERRIDES),
                "attn_implementation": model.backbone.config._attn_implementation,
                "stages": [asdict(s) for s in stages],
                "train_speakers": sorted(set(train_speakers))}, path)


def _load(job: SslJob, path: Path) -> SSLCommandClassifier:
    model = build_model(job)
    model.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"])
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
                   generator: torch.Generator) -> None:
    """Train on the controls, save controls.pt, save the controls-only run."""
    speakers = sorted(controls["speaker_id"].unique())
    print(f"Control stage: {len(speakers)} control speakers, {len(controls)} clips", flush=True)
    _train(model, controls, job.control_stages, job, feature_extractor, noise, generator)
    _save(model, job.out_dir / CONTROLS_CHECKPOINT, job, job.control_stages, speakers)
    frame = _predict(model, evaluation_clips(dysarthric), job, feature_extractor)
    no_dysarthric_training = {s: frozenset() for s in dysarthric["speaker_id"].unique()}
    save_loso_run([frame], no_dysarthric_training, speakers, job.seed,
                  controls_run_name(job.backbone.name), job.controls_out_dir / EVAL_DIR)


def _folds(job: SslJob, controls: pd.DataFrame, dysarthric: pd.DataFrame,
           feature_extractor, noise: NoiseBank, generator: torch.Generator) -> None:
    """One dysarthric fine-tune per held-out speaker, each from controls.pt; save the LOSO run."""
    control_speakers = sorted(controls["speaker_id"].unique())
    stages = (*job.control_stages, job.dysarthric_stage)
    frames, fold_train = [], {}
    for i, speaker in enumerate(sorted(dysarthric["speaker_id"].unique()), start=1):
        train_df, test_df = split_fold(dysarthric, speaker)
        print(f"Fold {i} (hold out {speaker}): {len(train_df)} training clips", flush=True)
        model = _load(job, job.out_dir / CONTROLS_CHECKPOINT)
        _train(model, train_df, (job.dysarthric_stage,), job, feature_extractor, noise,
               generator)
        _save(model, job.out_dir / f"fold{i}_{speaker}.pt", job, stages,
              [*train_df["speaker_id"], *control_speakers])
        frames.append(_predict(model, test_df, job, feature_extractor))
        fold_train[speaker] = frozenset(train_df["speaker_id"])
    save_loso_run(frames, fold_train, control_speakers, job.seed,
                  run_name(job.backbone.name), job.out_dir / EVAL_DIR)


def run_ssl_finetuning(job: SslJob, samples: pd.DataFrame) -> Tuple[Run, Run]:
    """Train one seed. Returns (LOSO run, controls-only run)."""
    _refuse_finished(job)
    set_seed(job.seed)
    df = with_label_ids(samples)
    controls, dysarthric = df[~df["is_dysarthric"]], df[df["is_dysarthric"]]
    noise = NoiseBank.from_dir(job.noise_dir, SAMPLE_RATE)
    feature_extractor = load_feature_extractor(job.backbone, job.cache_dir)
    model = build_model(job)
    cost = ssl_cost(model)  # before training, so a failure here costs seconds
    for name in (run_name, controls_run_name):
        save_cost(cost, Path(job.runs_dir) / name(job.backbone.name) / COST_FILE)
    job.out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(job.seed)
    _control_stage(job, model.to(job.device), controls, dysarthric, feature_extractor,
                   noise, generator)
    del model
    _folds(job, controls, dysarthric, feature_extractor, noise, generator)
    return load_run(job.out_dir / EVAL_DIR), load_run(job.controls_out_dir / EVAL_DIR)
