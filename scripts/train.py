#!/usr/bin/env python
"""
Training script for dysarthric voice command classifier.

Implements a 3-phase curriculum learning approach:
- Phase A: Control speaker pretraining
- Phase B: Dysarthric speaker fine-tuning
- Phase C: LOSO evaluation on dysarthric speakers

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
    phase_a_checkpoint_path = config.OUTPUT_DIR / 'phase_a_control_pretrained.pt'
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
    args: argparse.Namespace
) -> Path:
    """
    Phase B: Dysarthric Speaker Fine-tuning
    
    Load Phase A checkpoint and fine-tune on dysarthric speakers only.
    
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
    phase_b_checkpoint_path = config.OUTPUT_DIR / 'phase_b_curriculum_trained.pt'
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


def phase_c_loso_evaluation(
    model_dir: str,
    dysarthric_df: pd.DataFrame,
    label2id: dict,
    feature_extractor: Wav2Vec2FeatureExtractor,
    device: torch.device,
    phase_b_checkpoint_path: Path,
    args: argparse.Namespace
) -> dict:
    """
    Phase C: LOSO Evaluation on Dysarthric Speakers
    
    For each dysarthric speaker, hold them out and evaluate.
    
    Returns:
        Dictionary with cross-validation results
    """
    print("\n" + "=" * 60)
    print("PHASE C: LOSO EVALUATION ON DYSARTHRIC SPEAKERS")
    print("=" * 60)
    
    dysarthric_speakers = sorted(dysarthric_df['speaker_id'].unique().tolist())
    print(f"\nEvaluating on {len(dysarthric_speakers)} dysarthric speakers")
    print(f"Each fold: fine-tune on {len(dysarthric_speakers)-1} dysarthric speakers, evaluate on 1")
    
    batch_size = args.batch_size or config.BATCH_SIZE
    
    # Storage for cross-validation results
    cv_results = {
        'fold': [],
        'val_speaker': [],
        'best_val_acc': [],
        'best_val_loss': [],
        'train_samples': [],
        'val_samples': [],
        'all_preds': [],
        'all_labels': [],
        'all_speaker_ids': []
    }
    
    # Number of fine-tuning epochs per fold
    loso_finetune_epochs = args.epochs_loso or 10
    
    for fold_idx, val_speaker in enumerate(dysarthric_speakers):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(dysarthric_speakers)}: Hold out speaker {val_speaker}")
        print(f"{'='*60}")
        
        # Split dysarthric data for this fold
        fold_train_df = dysarthric_df[dysarthric_df['speaker_id'] != val_speaker].copy()
        fold_val_df = dysarthric_df[dysarthric_df['speaker_id'] == val_speaker].copy()
        
        print(f"  Training speakers: {sorted(fold_train_df['speaker_id'].unique().tolist())}")
        print(f"  Validation speaker: {val_speaker}")
        print(f"  Train samples: {len(fold_train_df)}, Val samples: {len(fold_val_df)}")
        
        # Create datasets for this fold
        fold_train_dataset = TORGOCommandDataset(
            fold_train_df, feature_extractor, config=config,
            max_length=config.MAX_AUDIO_SAMPLES, augment=True
        )
        fold_val_dataset = TORGOCommandDataset(
            fold_val_df, feature_extractor, config=config,
            max_length=config.MAX_AUDIO_SAMPLES, augment=False
        )
        
        # Create dataloaders
        fold_train_loader = DataLoader(
            fold_train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=4, pin_memory=True
        )
        fold_val_loader = DataLoader(
            fold_val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4, pin_memory=True
        )
        
        # Compute class weights for this fold's training data
        fold_class_weights = None
        if config.USE_CLASS_WEIGHTS:
            fold_class_weights = get_class_weights(fold_train_df, label2id, device)
        
        # Load Phase B checkpoint
        phase_b_checkpoint = torch.load(phase_b_checkpoint_path, map_location=device)
        
        # Initialize model and load Phase B weights
        fold_model = HuBERTForCommandClassification(
            model_path=model_dir,
            num_labels=len(label2id),
            hidden_size=config.HIDDEN_SIZE,
            classifier_dropout=config.CLASSIFIER_DROPOUT,
            freeze_encoder=False,
            freeze_feature_extractor=True
        ).to(device)
        
        fold_model.load_state_dict({k: v.to(device) for k, v in phase_b_checkpoint['model_state_dict'].items()})
        fold_model.unfreeze_encoder(num_layers=config.CURRICULUM_UNFREEZE_LAYERS)
        
        # Fine-tune on this fold's training data
        print(f"\n  Fine-tuning for {loso_finetune_epochs} epochs...")
        
        optimizer = torch.optim.AdamW([
            {'params': fold_model.classifier.parameters(), 'lr': config.CURRICULUM_DYSARTHRIC_LR * 0.5},
            {'params': fold_model.hubert.encoder.parameters(), 'lr': config.CURRICULUM_DYSARTHRIC_LR_ENCODER * 0.5}
        ], weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=loso_finetune_epochs, eta_min=1e-7
        )
        
        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_state_dict = None
        
        for epoch in range(loso_finetune_epochs):
            train_loss, train_acc = train_epoch(
                fold_model, fold_train_loader, optimizer, device,
                epoch, loso_finetune_epochs,
                max_grad_norm=config.MAX_GRAD_NORM, class_weights=fold_class_weights
            )
            val_loss, val_acc, _, _, _ = validate(
                fold_model, fold_val_loader, device, class_weights=fold_class_weights
            )
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}
        
        # Final evaluation with best model
        fold_model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        _, final_acc, final_preds, final_labels, final_speakers = validate(
            fold_model, fold_val_loader, device, class_weights=fold_class_weights
        )
        
        print(f"\n  Fold {fold_idx + 1} Results:")
        print(f"    Best validation accuracy: {best_val_acc:.4f}")
        print(f"    Samples correctly classified: {sum(p == l for p, l in zip(final_preds, final_labels))}/{len(final_labels)}")
        
        # Store results
        cv_results['fold'].append(fold_idx)
        cv_results['val_speaker'].append(val_speaker)
        cv_results['best_val_acc'].append(best_val_acc)
        cv_results['best_val_loss'].append(best_val_loss)
        cv_results['train_samples'].append(len(fold_train_df))
        cv_results['val_samples'].append(len(fold_val_df))
        cv_results['all_preds'].extend(final_preds)
        cv_results['all_labels'].extend(final_labels)
        cv_results['all_speaker_ids'].extend(final_speakers)
        
        # Save fold model
        fold_model_path = config.OUTPUT_DIR / f'curriculum_fold{fold_idx + 1}_{val_speaker}.pt'
        torch.save({
            'fold': fold_idx,
            'val_speaker': val_speaker,
            'model_state_dict': best_state_dict,
            'val_acc': best_val_acc,
            'val_loss': best_val_loss,
        }, fold_model_path)
        
        # Clean up
        del fold_model
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("PHASE C: LOSO EVALUATION COMPLETE")
    print("=" * 60)
    
    return cv_results


def print_results_summary(cv_results: dict, phase_a_acc: float, phase_b_acc: float):
    """Print a summary of the curriculum learning results."""
    mean_acc = np.mean(cv_results['best_val_acc'])
    std_acc = np.std(cv_results['best_val_acc'])
    
    print("\n" + "=" * 60)
    print("CURRICULUM LEARNING RESULTS BY FOLD")
    print("=" * 60)
    
    # Create results DataFrame
    cv_summary = pd.DataFrame({
        'Fold': [f + 1 for f in cv_results['fold']],
        'Val Speaker': cv_results['val_speaker'],
        'Val Accuracy': cv_results['best_val_acc'],
        'Val Loss': cv_results['best_val_loss'],
        'Train Samples': cv_results['train_samples'],
        'Val Samples': cv_results['val_samples']
    })
    cv_summary['Speaker Type'] = 'Dysarthric'
    
    print(cv_summary.to_string(index=False))
    
    print(f"\n" + "=" * 60)
    print("OVERALL DYSARTHRIC SPEAKER METRICS")
    print("=" * 60)
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Min Accuracy:  {min(cv_results['best_val_acc']):.4f}")
    print(f"Max Accuracy:  {max(cv_results['best_val_acc']):.4f}")
    
    print(f"\nPhase A (Control Pretraining) Accuracy: {phase_a_acc:.4f}")
    print(f"Phase B (Dysarthric Fine-tuning) Accuracy: {phase_b_acc:.4f}")
    print(f"Phase C (LOSO on Dysarthric) Mean Accuracy: {mean_acc:.4f}")
    
    # Save CV results
    cv_summary.to_csv(config.OUTPUT_DIR / 'curriculum_cv_results.csv', index=False)
    print(f"\nResults saved to {config.OUTPUT_DIR / 'curriculum_cv_results.csv'}")
    
    # Save to JSON
    cv_results_json = {
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'fold_results': cv_summary.to_dict(orient='records'),
        'phase_a_accuracy': float(phase_a_acc),
        'phase_b_accuracy': float(phase_b_acc),
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
    parser.add_argument('--epochs-loso', type=int, default=10,
                        help='LOSO fine-tuning epochs per fold (default: 10)')
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
    
    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    
    print("\nScanning TORGO dataset...")
    df = scan_torgo_dataset(config.TORGO_ROOT, config.TARGET_COMMANDS, config.MIC_TYPE)
    
    print(f"\nFound {len(df)} samples matching target commands")
    print(f"Unique speakers: {df['speaker_id'].nunique()}")
    print(f"Unique labels: {df['label'].nunique()}")
    
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
        print(f"  Phase C: LOSO evaluation on {len(dysarthric_speakers)} dysarthric speakers")
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
            phase_b_checkpoint_path, args
        )
        
        # Print results summary
        print_results_summary(cv_results, phase_a_acc, phase_b_acc)
    else:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE (Phase C skipped)")
        print("=" * 60)
        print(f"\nPhase A (Control Pretraining) Accuracy: {phase_a_acc:.4f}")
        print(f"Phase B (Dysarthric Fine-tuning) Accuracy: {phase_b_acc:.4f}")
        print(f"\nCheckpoints saved to: {config.OUTPUT_DIR}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
