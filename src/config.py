"""
Centralized configuration for the dysarthric voice command classifier.

All hyperparameters and paths are defined here for easy experimentation.
"""

from pathlib import Path


class Config:
    """
    Centralized configuration for the dysarthric voice command classifier.
    
    Design decisions documented inline.
    """
    
    # -------------------------------------------------------------------------
    # PATHS
    # -------------------------------------------------------------------------
    TORGO_ROOT = Path("/root/autodl-tmp/TORGO")
    OUTPUT_DIR = Path("/root/autodl-tmp/dysarthric-voice-cmds/outputs")
    MODEL_CACHE_DIR = Path("/root/autodl-tmp/dysarthric-voice-cmds/model_cache")
    
    # -------------------------------------------------------------------------
    # TARGET COMMANDS
    # -------------------------------------------------------------------------
    # These are from TORGO's "short words" category, specifically designed for
    # assistive technology / accessibility software (see TORGO documentation)
    DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'forward', 'back', 'select', 'menu']
    # RADIO_ALPHABET = [
    #     'alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel',
    #     'india', 'juliet', 'kilo', 'lima', 'mike', 'november', 'oscar', 'papa',
    #     'quebec', 'romeo', 'sierra', 'tango', 'uniform', 'victor', 'whiskey',
    #     'xray', 'yankee', 'zulu'
    # ]
    # SCALE UP: 20 classes (digits + directional commands)
    # Change to DIGITS + COMMANDS + RADIO_ALPHABET for full 46 classes
    TARGET_COMMANDS = DIGITS + COMMANDS  # 20 classes
    
    # -------------------------------------------------------------------------
    # AUDIO SETTINGS
    # -------------------------------------------------------------------------
    # HuBERT was trained on 16kHz audio - must match for optimal performance
    SAMPLE_RATE = 16000
    
    # Max audio length: Single-word commands are typically <2 seconds.
    # 3 seconds provides buffer for slower dysarthric speech.
    # Trade-off: Longer = more memory, shorter = potential truncation
    MAX_AUDIO_LENGTH = 3.0  # seconds
    MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_LENGTH)  # 48000 samples
    
    # -------------------------------------------------------------------------
    # MODEL SETTINGS
    # -------------------------------------------------------------------------
    # HuBERT-large: 24 transformer layers, 1024 hidden size, 315M params
    # Why large vs base? Better representations, but slower/more memory
    # For production, consider HuBERT-base (90M params) with slight accuracy trade-off
    MODELSCOPE_MODEL_ID = "facebook/hubert-large-ls960-ft"
    HIDDEN_SIZE = 1024  # Must match HuBERT-large architecture
    
    # Classifier settings
    CLASSIFIER_DROPOUT = 0.1  # Standard dropout for regularization
    
    # -------------------------------------------------------------------------
    # TRAINING SETTINGS
    # -------------------------------------------------------------------------
    # Batch size: Limited by GPU memory with 315M param model
    # RTX 4090 (24GB) can handle 8-16; reduce for smaller GPUs
    BATCH_SIZE = 8
    
    # Learning rates: Following transfer learning best practices
    # - Higher LR (1e-4) for randomly initialized classifier head
    # - Lower LR (1e-5) for pretrained encoder to avoid catastrophic forgetting
    LEARNING_RATE = 1e-4          # For classifier head (warmup phase)
    LEARNING_RATE_FINETUNE = 1e-5  # For encoder fine-tuning
    
    # Epochs: Two-phase training strategy
    # Phase 1 (warmup): Train only classifier, encoder frozen
    # Phase 2 (finetune): Unfreeze top encoder layers, lower LR
    NUM_EPOCHS = 20
    WARMUP_EPOCHS = 5
    
    # Number of top transformer layers to unfreeze in Phase 2
    # Why 4? Trade-off between adaptation and preserving pretrained knowledge
    # More layers = more adaptation but higher overfitting risk on small data
    UNFREEZE_LAYERS = 4
    
    # Gradient clipping to prevent exploding gradients during fine-tuning
    MAX_GRAD_NORM = 1.0
    
    # Weight decay for AdamW optimizer (L2 regularization)
    WEIGHT_DECAY = 0.01
    
    # -------------------------------------------------------------------------
    # DATA SETTINGS
    # -------------------------------------------------------------------------
    # TORGO has two microphone types:
    # - wav_arrayMic: Acoustic Magic array microphone (better quality, recommended)
    # - wav_headMic: Head-mounted microphone (more noise from EMA interference)
    MIC_TYPE = "wav_arrayMic"
    
    # Whether to use class-weighted loss for imbalanced classes
    # IMPORTANT: Set to True when classes have very different sample counts
    # Uses "balanced" strategy: weight = n_samples / (n_classes * n_samples_for_class)
    USE_CLASS_WEIGHTS = True
    
    # -------------------------------------------------------------------------
    # AUGMENTATION SETTINGS
    # -------------------------------------------------------------------------
    # audiomentations-based augmentation pipeline (industry standard)
    # Each augmentation has a probability (p) of being applied
    
    # Time-domain augmentations
    AUG_NOISE_MIN_AMP = 0.001    # Minimum Gaussian noise amplitude
    AUG_NOISE_MAX_AMP = 0.015    # Maximum Gaussian noise amplitude
    AUG_NOISE_PROB = 0.5         # Probability of adding noise
    
    AUG_PITCH_MIN_SEMITONES = -2  # Min pitch shift (semitones)
    AUG_PITCH_MAX_SEMITONES = 2   # Max pitch shift (semitones)
    AUG_PITCH_PROB = 0.5          # Probability of pitch shift
    
    AUG_TIMESTRETCH_MIN = 0.9     # Min time stretch factor
    AUG_TIMESTRETCH_MAX = 1.1     # Max time stretch factor
    AUG_TIMESTRETCH_PROB = 0.5    # Probability of time stretch
    
    AUG_SHIFT_MIN = -0.2          # Min shift as fraction of total length
    AUG_SHIFT_MAX = 0.2           # Max shift as fraction of total length
    AUG_SHIFT_PROB = 0.3          # Probability of time shift
    
    AUG_GAIN_MIN_DB = -6          # Min gain adjustment (dB)
    AUG_GAIN_MAX_DB = 6           # Max gain adjustment (dB)
    AUG_GAIN_PROB = 0.5           # Probability of gain adjustment
    
    # SpecAugment settings (frequency/time masking)
    # From Google's SpecAugment paper (Park et al., 2019)
    SPEC_AUG_FREQ_MASK_PARAM = 27  # Max frequency bands to mask (F)
    SPEC_AUG_TIME_MASK_PARAM = 100 # Max time steps to mask (T)
    SPEC_AUG_NUM_FREQ_MASKS = 1    # Number of frequency masks
    SPEC_AUG_NUM_TIME_MASKS = 1    # Number of time masks
    
    # -------------------------------------------------------------------------
    # CURRICULUM LEARNING SETTINGS
    # -------------------------------------------------------------------------
    # Three-phase curriculum learning approach:
    # Phase A: Train on control speakers (clean speech, closer to HuBERT pretraining)
    # Phase B: Fine-tune on dysarthric speakers only (adapt to dysarthric patterns)
    # Phase C: LOSO evaluation on dysarthric speakers
    
    CURRICULUM_CONTROL_EPOCHS = 15    # Phase A: epochs on control speakers
    CURRICULUM_DYSARTHRIC_EPOCHS = 15  # Phase B: epochs on dysarthric speakers
    CURRICULUM_UNFREEZE_LAYERS = 4     # Layers to unfreeze during fine-tuning
    
    # Phase A learning rates (control pretraining)
    CURRICULUM_CONTROL_LR = 1e-4           # Higher LR for classifier warmup
    CURRICULUM_CONTROL_LR_FINETUNE = 1e-5  # Lower LR when unfreezing encoder
    
    # Phase B learning rates (dysarthric fine-tuning)
    CURRICULUM_DYSARTHRIC_LR = 5e-5        # Lower LR to preserve control knowledge
    CURRICULUM_DYSARTHRIC_LR_ENCODER = 5e-6  # Even lower for encoder


# Create a default config instance
config = Config()
