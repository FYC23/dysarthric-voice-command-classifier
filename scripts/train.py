#!/usr/bin/env python
"""
Training script for dysarthric voice command classifier.

Implements a 3-phase curriculum learning approach:
- Phase A: Control speaker pretraining
- Phase B: Dysarthric speaker fine-tuning
- Phase C: LOSO evaluation on dysarthric speakers (each fold re-runs Phase B
  from the Phase A checkpoint without the held-out speaker)

Usage:
    python scripts/train.py
    python scripts/train.py --skip-phase-c
    python scripts/train.py --epochs-a 10 --epochs-b 10
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
from modelscope import snapshot_download

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.preprocessing import scan_torgo_dataset, create_label_mapping
from src.data.dataset import TORGOCommandDataset, collate_fn
from src.data.segments import apply_segment_labels, load_segment_labels, measure_kept_lengths
from src.model.architecture import HuBERTForCommandClassification
from src.training.trainer import train_epoch, validate, save_checkpoint


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class_weights(df: pd.DataFrame, label2id: dict, device: torch.device) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.
    
    Uses sklearn's "balanced" strategy: weight = n_samples / (n_classes * n_samples_for_class)
    """
    train_labels = df['label_id'].values
    classes_in_data = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes_in_data,
        y=train_labels
    )
    # Create full weight tensor (1.0 for missing classes)
    full_weights = np.ones(len(label2id))
    for i, cls in enumerate(classes_in_data):
        full_weights[cls] = weights[i]
    return torch.tensor(full_weights, dtype=torch.float32).to(device)


def phase_a_training(
    model_dir: str,
    control_df: pd.DataFrame,
    label2id: dict,
    feature_extractor: Wav2Vec2FeatureExtractor,
    device: torch.device,
    args: argparse.Namespace
) -> Path:
    """
    Phase A: Control Speaker Pretraining
    
    Train on control speakers (clean speech) to learn basic command discrimination.
    
    Returns:
        Path to saved Phase A checkpoint
    """
    print("\n" + "=" * 60)
    print("PHASE A: CONTROL SPEAKER PRETRAINING")
    print("=" * 60)
    
    control_speakers = sorted(control_df['speaker_id'].unique().tolist())
    print(f"\nTraining on {len(control_speakers)} control speakers: {control_speakers}")
    print(f"Total samples: {len(control_df)}")
    
    # Create control dataset
    control_train_dataset = TORGOCommandDataset(
        control_df, feature_extractor, config=config,
        max_length=config.MAX_AUDIO_SAMPLES, augment=True
    )
    
    batch_size = args.batch_size or config.BATCH_SIZE
    control_train_loader = DataLoader(
        control_train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # Compute class weights
    control_class_weights = None
    if config.USE_CLASS_WEIGHTS:
        control_class_weights = get_class_weights(control_df, label2id, device)
        print(f"\nClass weights computed for control data")
        print(f"  Classes present: {len(np.unique(control_df['label_id'].values))}/{len(label2id)}")
    
    # Initialize fresh model for Phase A
    print("\nInitializing model for Phase A...")
    phase_a_model = HuBERTForCommandClassification(
        model_path=model_dir,
        num_labels=len(label2id),
        hidden_size=config.HIDDEN_SIZE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
        freeze_encoder=True,
        freeze_feature_extractor=True
    ).to(device)
    
    # Training history
    phase_a_history = {'train_loss': [], 'train_acc': [], 'phase': []}
    
    # Phase A epochs
    total_epochs = args.epochs_a or config.CURRICULUM_CONTROL_EPOCHS
    warmup_epochs = min(5, total_epochs // 3)
    finetune_epochs = total_epochs - warmup_epochs
    
    # =========================================================================
    # Phase A.1: Warmup - Train classifier only
    # =========================================================================
    print(f"\nPhase A.1: Classifier warmup ({warmup_epochs} epochs)")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, phase_a_model.parameters()),
        lr=config.CURRICULUM_CONTROL_LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=warmup_epochs, eta_min=1e-6
    )
    
    for epoch in range(warmup_epochs):
        train_loss, train_acc = train_epoch(
            phase_a_model, control_train_loader, optimizer, device,
            epoch, warmup_epochs,
            max_grad_norm=config.MAX_GRAD_NORM, class_weights=control_class_weights
        )
        scheduler.step()
        phase_a_history['train_loss'].append(train_loss)
        phase_a_history['train_acc'].append(train_acc)
        phase_a_history['phase'].append('warmup')
        print(f"  Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}")
    
    # =========================================================================
    # Phase A.2: Fine-tuning - Unfreeze top layers
    # =========================================================================
    print(f"\nPhase A.2: Encoder fine-tuning ({finetune_epochs} epochs)")
    
    phase_a_model.unfreeze_encoder(num_layers=config.CURRICULUM_UNFREEZE_LAYERS)
    print(f"  Trainable params: {phase_a_model.get_trainable_params():,}")
    
    optimizer = torch.optim.AdamW([
        {'params': phase_a_model.classifier.parameters(), 'lr': config.CURRICULUM_CONTROL_LR_FINETUNE},
        {'params': phase_a_model.hubert.encoder.parameters(), 'lr': config.CURRICULUM_CONTROL_LR_FINETUNE * 0.1}
    ], weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs, eta_min=1e-7
    )
    
    best_train_acc_a = 0.0
    best_state_dict_a = None
    
    for epoch in range(finetune_epochs):
        train_loss, train_acc = train_epoch(
            phase_a_model, control_train_loader, optimizer, device,
            warmup_epochs + epoch, total_epochs,
            max_grad_norm=config.MAX_GRAD_NORM, class_weights=control_class_weights
        )
        scheduler.step()
        phase_a_history['train_loss'].append(train_loss)
        phase_a_history['train_acc'].append(train_acc)
        phase_a_history['phase'].append('finetune')
        print(f"  Epoch {warmup_epochs + epoch + 1}: loss={train_loss:.4f}, acc={train_acc:.4f}")
        
        if train_acc > best_train_acc_a:
            best_train_acc_a = train_acc
            best_state_dict_a = {k: v.cpu().clone() for k, v in phase_a_model.state_dict().items()}
    
    # Save Phase A checkpoint
    phase_a_checkpoint_path = config.RUNS_DIR / 'phase_a_control_pretrained.pt'
    torch.save({
        'model_state_dict': best_state_dict_a,
        'train_acc': best_train_acc_a,
        'history': phase_a_history,
        'control_speakers': control_speakers,
    }, phase_a_checkpoint_path)
    
    print(f"\n" + "=" * 60)
    print(f"PHASE A COMPLETE")
    print(f"=" * 60)
    print(f"Best training accuracy: {best_train_acc_a:.4f}")
    print(f"Checkpoint saved: {phase_a_checkpoint_path}")
    
    # Clean up
    del phase_a_model
    torch.cuda.empty_cache()
    
    return phase_a_checkpoint_path


def phase_b_training(
    model_dir: str,
    dysarthric_df: pd.DataFrame,
    label2id: dict,
    feature_extractor: Wav2Vec2FeatureExtractor,
    device: torch.device,
    phase_a_checkpoint_path: Path,
    args: argparse.Namespace,
    checkpoint_path: Optional[Path] = None
) -> Path:
    """
    Phase B: Dysarthric Speaker Fine-tuning

    Load Phase A checkpoint and fine-tune on the given dysarthric speakers only.
    Also reused by Phase C to train each LOSO fold from the same Phase A
    starting point, so the held-out speaker is never seen during training.

    Returns:
        Path to saved Phase B checkpoint
    """
    print("\n" + "=" * 60)
    print("PHASE B: DYSARTHRIC SPEAKER FINE-TUNING")
    print("=" * 60)
    
    dysarthric_speakers = sorted(dysarthric_df['speaker_id'].unique().tolist())
    print(f"\nFine-tuning on {len(dysarthric_speakers)} dysarthric speakers: {dysarthric_speakers}")
    print(f"Total samples: {len(dysarthric_df)}")
    
    # Create dysarthric dataset
    dysarthric_train_dataset = TORGOCommandDataset(
        dysarthric_df, feature_extractor, config=config,
        max_length=config.MAX_AUDIO_SAMPLES, augment=True
    )
    
    batch_size = args.batch_size or config.BATCH_SIZE
    dysarthric_train_loader = DataLoader(
        dysarthric_train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # Compute class weights for dysarthric data
    dysarthric_class_weights = None
    if config.USE_CLASS_WEIGHTS:
        dysarthric_class_weights = get_class_weights(dysarthric_df, label2id, device)
        print(f"\nClass weights computed for dysarthric data")
        print(f"  Classes present: {len(np.unique(dysarthric_df['label_id'].values))}/{len(label2id)}")
    
    # Load Phase A checkpoint
    print(f"\nLoading Phase A checkpoint from: {phase_a_checkpoint_path}")
    phase_a_checkpoint = torch.load(phase_a_checkpoint_path, map_location=device)
    print(f"  Phase A training accuracy: {phase_a_checkpoint['train_acc']:.4f}")
    
    # Initialize model and load Phase A weights
    phase_b_model = HuBERTForCommandClassification(
        model_path=model_dir,
        num_labels=len(label2id),
        hidden_size=config.HIDDEN_SIZE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
        freeze_encoder=False,
        freeze_feature_extractor=True
    ).to(device)
    
    # Load Phase A state dict
    phase_b_model.load_state_dict({k: v.to(device) for k, v in phase_a_checkpoint['model_state_dict'].items()})
    print("Phase A weights loaded successfully")
    
    # Unfreeze top layers for fine-tuning
    phase_b_model.unfreeze_encoder(num_layers=config.CURRICULUM_UNFREEZE_LAYERS)
    print(f"Trainable params: {phase_b_model.get_trainable_params():,}")
    
    # Training history
    phase_b_history = {'train_loss': [], 'train_acc': []}
    
    # Phase B epochs
    total_epochs = args.epochs_b or config.CURRICULUM_DYSARTHRIC_EPOCHS
    
    print(f"\nPhase B: Dysarthric fine-tuning ({total_epochs} epochs)")
    
    optimizer = torch.optim.AdamW([
        {'params': phase_b_model.classifier.parameters(), 'lr': config.CURRICULUM_DYSARTHRIC_LR},
        {'params': phase_b_model.hubert.encoder.parameters(), 'lr': config.CURRICULUM_DYSARTHRIC_LR_ENCODER}
    ], weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-7
    )
    
    best_train_acc_b = 0.0
    best_state_dict_b = None
    
    for epoch in range(total_epochs):
        train_loss, train_acc = train_epoch(
            phase_b_model, dysarthric_train_loader, optimizer, device,
            epoch, total_epochs,
            max_grad_norm=config.MAX_GRAD_NORM, class_weights=dysarthric_class_weights
        )
        scheduler.step()
        phase_b_history['train_loss'].append(train_loss)
        phase_b_history['train_acc'].append(train_acc)
        print(f"  Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}")
        
        if train_acc > best_train_acc_b:
            best_train_acc_b = train_acc
            best_state_dict_b = {k: v.cpu().clone() for k, v in phase_b_model.state_dict().items()}
    
    # Save Phase B checkpoint (curriculum-trained model)
    phase_b_checkpoint_path = checkpoint_path or config.RUNS_DIR / 'phase_b_curriculum_trained.pt'
    torch.save({
        'model_state_dict': best_state_dict_b,
        'train_acc': best_train_acc_b,
        'history': phase_b_history,
        'dysarthric_speakers': dysarthric_speakers,
    }, phase_b_checkpoint_path)
    
    print(f"\n" + "=" * 60)
    print(f"PHASE B COMPLETE")
    print(f"=" * 60)
    print(f"Best training accuracy on dysarthric data: {best_train_acc_b:.4f}")
    print(f"Checkpoint saved: {phase_b_checkpoint_path}")
    
    # Clean up
    del phase_b_model
    torch.cuda.empty_cache()
    
    return phase_b_checkpoint_path


def evaluate_checkpoint(
    model_dir: str,
    checkpoint_path: Path,
    eval_df: pd.DataFrame,
    label2id: dict,
    feature_extractor: Wav2Vec2FeatureExtractor,
    device: torch.device,
    batch_size: int
) -> tuple:
    """
    Evaluate a saved checkpoint once on held-out data (no augmentation).

    Loss is unweighted cross-entropy so it is comparable across folds.

    Returns:
        (loss, accuracy, preds, labels, speaker_ids)
    """
    eval_dataset = TORGOCommandDataset(
        eval_df, feature_extractor, config=config,
        max_length=config.MAX_AUDIO_SAMPLES, augment=False
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HuBERTForCommandClassification(
        model_path=model_dir,
        num_labels=len(label2id),
        hidden_size=config.HIDDEN_SIZE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
        freeze_encoder=True,
        freeze_feature_extractor=True
    ).to(device)
    model.load_state_dict({k: v.to(device) for k, v in checkpoint['model_state_dict'].items()})

    results = validate(model, eval_loader, device, class_weights=None)

    del model
    torch.cuda.empty_cache()
    return results


def phase_c_loso_evaluation(
    model_dir: str,
    dysarthric_df: pd.DataFrame,
    label2id: dict,
    feature_extractor: Wav2Vec2FeatureExtractor,
    device: torch.device,
    phase_a_checkpoint_path: Path,
    args: argparse.Namespace
) -> dict:
    """
    Phase C: LOSO Evaluation on Dysarthric Speakers
    
    For each dysarthric speaker, re-run Phase B from the Phase A checkpoint on
    the remaining dysarthric speakers, then evaluate once on the held-out one.

    The Phase B model trained on *all* dysarthric speakers must not be used as
    the starting point here: it has already seen the held-out speaker. Phase A
    only uses control speakers, so sharing it across folds does not leak.
    Model selection inside each fold uses training accuracy only (same rule as
    Phase B); the held-out speaker is never used to pick an epoch.
    
    Returns:
        Dictionary with cross-validation results
    """
    print("\n" + "=" * 60)
    print("PHASE C: LOSO EVALUATION ON DYSARTHRIC SPEAKERS")
    print("=" * 60)
    
    dysarthric_speakers = sorted(dysarthric_df['speaker_id'].unique().tolist())
    print(f"\nEvaluating on {len(dysarthric_speakers)} dysarthric speakers")
    print(f"Each fold: Phase A checkpoint -> Phase B on {len(dysarthric_speakers)-1} "
          f"dysarthric speakers -> evaluate on the held-out speaker")
    
    batch_size = args.batch_size or config.BATCH_SIZE
    
    # Storage for cross-validation results
    cv_results = {
        'fold': [],
        'test_speaker': [],
        'test_acc': [],
        'test_loss': [],
        'train_samples': [],
        'test_samples': [],
        'all_preds': [],
        'all_labels': [],
        'all_speaker_ids': []
    }
    
    for fold_idx, test_speaker in enumerate(dysarthric_speakers):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(dysarthric_speakers)}: Hold out speaker {test_speaker}")
        print(f"{'='*60}")
        
        # Split dysarthric data for this fold
        fold_train_df = dysarthric_df[dysarthric_df['speaker_id'] != test_speaker].copy()
        fold_test_df = dysarthric_df[dysarthric_df['speaker_id'] == test_speaker].copy()
        assert test_speaker not in set(fold_train_df['speaker_id']), \
            f"Held-out speaker {test_speaker} leaked into fold training data"
        
        print(f"  Training speakers: {sorted(fold_train_df['speaker_id'].unique().tolist())}")
        print(f"  Test speaker: {test_speaker}")
        print(f"  Train samples: {len(fold_train_df)}, Test samples: {len(fold_test_df)}")
        
        # Train this fold's model from Phase A, without the held-out speaker
        fold_model_path = config.RUNS_DIR / f'curriculum_fold{fold_idx + 1}_{test_speaker}.pt'
        phase_b_training(
            model_dir, fold_train_df, label2id, feature_extractor, device,
            phase_a_checkpoint_path, args, checkpoint_path=fold_model_path
        )
        
        # Single evaluation on the held-out speaker
        test_loss, test_acc, test_preds, test_labels, test_speakers = evaluate_checkpoint(
            model_dir, fold_model_path, fold_test_df, label2id,
            feature_extractor, device, batch_size
        )
        
        print(f"\n  Fold {fold_idx + 1} Results:")
        print(f"    Test accuracy: {test_acc:.4f}")
        print(f"    Samples correctly classified: {sum(p == l for p, l in zip(test_preds, test_labels))}/{len(test_labels)}")
        
        # Store results
        cv_results['fold'].append(fold_idx)
        cv_results['test_speaker'].append(test_speaker)
        cv_results['test_acc'].append(test_acc)
        cv_results['test_loss'].append(test_loss)
        cv_results['train_samples'].append(len(fold_train_df))
        cv_results['test_samples'].append(len(fold_test_df))
        cv_results['all_preds'].extend(test_preds)
        cv_results['all_labels'].extend(test_labels)
        cv_results['all_speaker_ids'].extend(test_speakers)
    
    print("\n" + "=" * 60)
    print("PHASE C: LOSO EVALUATION COMPLETE")
    print("=" * 60)
    
    return cv_results


def print_results_summary(cv_results: dict, phase_a_train_acc: float, phase_b_train_acc: float):
    """Print a summary of the curriculum learning results."""
    fold_accs = cv_results['test_acc']
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    # Pooled accuracy weights every utterance equally; the per-speaker mean
    # weights every speaker equally (folds range from ~8 to ~40 utterances).
    pooled_acc = float(np.mean(
        np.array(cv_results['all_preds']) == np.array(cv_results['all_labels'])
    ))
    
    print("\n" + "=" * 60)
    print("CURRICULUM LEARNING RESULTS BY FOLD")
    print("=" * 60)
    
    # Create results DataFrame
    cv_summary = pd.DataFrame({
        'Fold': [f + 1 for f in cv_results['fold']],
        'Test Speaker': cv_results['test_speaker'],
        'Test Accuracy': fold_accs,
        'Test Loss': cv_results['test_loss'],
        'Train Samples': cv_results['train_samples'],
        'Test Samples': cv_results['test_samples']
    })
    cv_summary['Speaker Type'] = 'Dysarthric'
    
    print(cv_summary.to_string(index=False))
    
    print(f"\n" + "=" * 60)
    print("OVERALL DYSARTHRIC SPEAKER METRICS (held-out LOSO)")
    print("=" * 60)
    print(f"Mean per-speaker accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Pooled accuracy:           {pooled_acc:.4f} "
          f"({len(cv_results['all_labels'])} utterances)")
    print(f"Min speaker accuracy:      {min(fold_accs):.4f}")
    print(f"Max speaker accuracy:      {max(fold_accs):.4f}")
    
    print(f"\nPhase A training accuracy (control, augmented):    {phase_a_train_acc:.4f}")
    print(f"Phase B training accuracy (dysarthric, augmented): {phase_b_train_acc:.4f}")
    print(f"Phase C held-out accuracy (LOSO, per-speaker mean): {mean_acc:.4f}")
    
    # Save CV results
    cv_summary.to_csv(config.OUTPUT_DIR / 'curriculum_cv_results.csv', index=False)
    print(f"\nResults saved to {config.OUTPUT_DIR / 'curriculum_cv_results.csv'}")
    
    # Save to JSON
    cv_results_json = {
        'evaluation': 'loso_phase_b_retrained_per_fold',
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'pooled_accuracy': pooled_acc,
        'fold_results': cv_summary.to_dict(orient='records'),
        'phase_a_train_accuracy': float(phase_a_train_acc),
        'phase_b_train_accuracy': float(phase_b_train_acc),
        'phase_c_mean_accuracy': float(mean_acc)
    }
    with open(config.OUTPUT_DIR / 'curriculum_cv_results.json', 'w') as f:
        json.dump(cv_results_json, f, indent=2)
    print(f"JSON results saved to {config.OUTPUT_DIR / 'curriculum_cv_results.json'}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train dysarthric voice command classifier')
    parser.add_argument('--skip-phase-c', action='store_true',
                        help='Skip Phase C (LOSO evaluation) for faster training')
    parser.add_argument('--epochs-a', type=int, default=None,
                        help=f'Phase A epochs (default: {config.CURRICULUM_CONTROL_EPOCHS})')
    parser.add_argument('--epochs-b', type=int, default=None,
                        help=f'Phase B epochs (default: {config.CURRICULUM_DYSARTHRIC_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help=f'Batch size (default: {config.BATCH_SIZE})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directories
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    
    print("\nScanning TORGO dataset...")
    df = scan_torgo_dataset(config.TORGO_ROOT, config.TARGET_COMMANDS,
                            config.MIC_TYPES, config.MIN_AUDIO_DURATION,
                            config.MAX_AUDIO_DURATION)

    print(f"\nFound {len(df)} samples matching target commands")
    print(f"Per mic: {df['mic'].value_counts().to_dict()}")
    print(f"Unique speakers: {df['speaker_id'].nunique()}")
    print(f"Unique labels: {df['label'].nunique()}")

    # Clips still longer than the window after trimming are cropped to
    # hand-labelled word segments; unlabelled ones are dropped (src/data/segments.py)
    labels = load_segment_labels(config.TORGO_SEGMENT_LABELS)
    result = apply_segment_labels(measure_kept_lengths(df), labels,
                                  config.TORGO_ROOT, config.MAX_AUDIO_LENGTH)
    df = result.samples
    print(f"\nSegment labels: {len(labels)} clips labelled, "
          f"{df['seg_start'].notna().sum()} samples cropped to a labelled word")
    if len(result.dropped):
        print(f"Dropped {len(result.dropped)} clips: {result.dropped['reason'].value_counts().to_dict()}")
        for _, r in result.dropped[result.dropped['reason'] == 'unlabeled_over_window'].iterrows():
            print(f"  needs a label: {Path(r['file_path']).relative_to(config.TORGO_ROOT)} "
                  f"({r['kept_s']:.2f} s after trimming)")
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.non_speech.to_csv(config.CACHE_DIR / 'torgo_non_speech_segments.csv', index=False)

    # Create label encoding
    label2id, id2label = create_label_mapping(df)
    df['label_id'] = df['label'].map(label2id)
    
    print(f"\nNumber of classes: {len(label2id)}")
    
    # Save label mapping
    label_mapping = {'label2id': label2id, 'id2label': {str(k): v for k, v in id2label.items()}}
    with open(config.OUTPUT_DIR / 'label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Label mapping saved to {config.OUTPUT_DIR / 'label_mapping.json'}")
    
    # Split data by speaker type
    control_df = df[~df['is_dysarthric']].copy()
    dysarthric_df = df[df['is_dysarthric']].copy()
    
    control_speakers = sorted(control_df['speaker_id'].unique().tolist())
    dysarthric_speakers = sorted(dysarthric_df['speaker_id'].unique().tolist())
    
    print(f"\nControl speakers: {control_speakers}")
    print(f"Control samples: {len(control_df)}")
    print(f"Dysarthric speakers: {dysarthric_speakers}")
    print(f"Dysarthric samples: {len(dysarthric_df)}")
    
    # =========================================================================
    # MODEL SETUP
    # =========================================================================
    print("\n" + "=" * 60)
    print("MODEL SETUP")
    print("=" * 60)
    
    print("\nDownloading HuBERT-large from ModelScope...")
    model_dir = snapshot_download(
        config.MODELSCOPE_MODEL_ID,
        cache_dir=str(config.MODEL_CACHE_DIR)
    )
    print(f"Model downloaded to: {model_dir}")
    
    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    print("Feature extractor loaded successfully")
    
    # =========================================================================
    # CURRICULUM LEARNING TRAINING
    # =========================================================================
    print("\n" + "=" * 60)
    print("CURRICULUM LEARNING TRAINING")
    print("=" * 60)
    
    print(f"\nTraining Plan:")
    epochs_a = args.epochs_a or config.CURRICULUM_CONTROL_EPOCHS
    epochs_b = args.epochs_b or config.CURRICULUM_DYSARTHRIC_EPOCHS
    print(f"  Phase A: Control pretraining ({epochs_a} epochs)")
    print(f"  Phase B: Dysarthric fine-tuning ({epochs_b} epochs)")
    if not args.skip_phase_c:
        print(f"  Phase C: LOSO evaluation on {len(dysarthric_speakers)} dysarthric speakers "
              f"(Phase B re-run per fold, {epochs_b} epochs each)")
    else:
        print(f"  Phase C: Skipped")
    
    batch_size = args.batch_size or config.BATCH_SIZE
    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Control LR: {config.CURRICULUM_CONTROL_LR}")
    print(f"  Dysarthric LR: {config.CURRICULUM_DYSARTHRIC_LR}")
    print(f"  Unfreeze layers: {config.CURRICULUM_UNFREEZE_LAYERS}")
    print(f"  Class weighting: {'Enabled' if config.USE_CLASS_WEIGHTS else 'Disabled'}")
    
    # Phase A: Control Speaker Pretraining
    phase_a_checkpoint_path = phase_a_training(
        model_dir, control_df, label2id, feature_extractor, device, args
    )
    
    # Load Phase A accuracy for summary
    phase_a_checkpoint = torch.load(phase_a_checkpoint_path, map_location='cpu')
    phase_a_acc = phase_a_checkpoint['train_acc']
    
    # Phase B: Dysarthric Fine-tuning
    phase_b_checkpoint_path = phase_b_training(
        model_dir, dysarthric_df, label2id, feature_extractor, device,
        phase_a_checkpoint_path, args
    )
    
    # Load Phase B accuracy for summary
    phase_b_checkpoint = torch.load(phase_b_checkpoint_path, map_location='cpu')
    phase_b_acc = phase_b_checkpoint['train_acc']
    
    # Phase C: LOSO Evaluation (optional)
    if not args.skip_phase_c:
        cv_results = phase_c_loso_evaluation(
            model_dir, dysarthric_df, label2id, feature_extractor, device,
            phase_a_checkpoint_path, args
        )
        
        # Print results summary
        print_results_summary(cv_results, phase_a_acc, phase_b_acc)
    else:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE (Phase C skipped)")
        print("=" * 60)
        print(f"\nPhase A training accuracy (control, augmented): {phase_a_acc:.4f}")
        print(f"Phase B training accuracy (dysarthric, augmented): {phase_b_acc:.4f}")
        print("No held-out evaluation was run (Phase C skipped).")
        print(f"\nCheckpoints saved to: {config.RUNS_DIR}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
