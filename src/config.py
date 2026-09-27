"""
Centralized configuration for the dysarthric voice command classifier.

Paths, target commands and the audio window; training recipes live next to their code.
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
    # All paths are relative to the repo root, so the same code runs on any
    # machine. data/raw, data/cache and runs/ are gitignored (see data/README.md).
    REPO_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = REPO_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"      # Datasets exactly as downloaded (read-only)
    CACHE_DIR = DATA_DIR / "cache"       # Derived files, safe to delete and rebuild

    TORGO_ROOT = RAW_DATA_DIR / "TORGO"
    SPEECH_COMMANDS_ROOT = RAW_DATA_DIR / "speech_commands_v2"
    # Background noise for training augmentation (src/data/noise.py): the 5 long
    # recordings in Speech Commands' train split. The HF split put a 6th
    # (running_tap) in validation; it is left out so no held-out audio is trained on.
    NOISE_DIR = SPEECH_COMMANDS_ROOT / "train" / "_silence_"
    MODEL_CACHE_DIR = CACHE_DIR / "pretrained"  # Downloaded pretrained weights (SSL backbones, ASR baselines)
    RUNS_DIR = REPO_ROOT / "runs"        # Training checkpoints (large, not committed)
    OUTPUT_DIR = REPO_ROOT / "outputs"   # Results tables and plots (committed)
    LABELS_DIR = DATA_DIR / "labels"     # Hand labels (committed: can't be regenerated)
    # Hand-made word / partial / non-speech segments for the 36 longest clips
    TORGO_MANUAL_SEGMENT_LABELS = LABELS_DIR / "torgo_manual_segment_labels.json"
    
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
    
    # Fixed model window, applied AFTER silence trimming (VAD in src/audio.py).
    # Fixed (not variable) because the target is an on-device streaming model.
    # After trimming, 98.7% of TORGO clips fit in 2 s, and every clip that
    # contains a single sound does (max 1.83 s). The 13 clips that don't are
    # dysarthric struggle + word; they are cropped to hand-labelled word
    # segments (src/data/segments.py), never blindly (the loudest part is often
    # the struggle, not the word).
    MAX_AUDIO_LENGTH = 2.0  # seconds
    MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_LENGTH)  # 32000 samples
    
    # -------------------------------------------------------------------------
    # DATA SETTINGS
    # -------------------------------------------------------------------------
    # TORGO records every take on two microphones:
    # - wav_arrayMic: Acoustic Magic array mic at 61 cm (cleaner)
    # - wav_headMic: head-mounted mic (more noise from EMA interference, more low end)
    # Both are used: this roughly doubles the data, recovers speakers with
    # sessions on only one mic (M05 Session2 has no array mic), and stops mic
    # type from being specific to a few speakers. Splits are by speaker, so the
    # two recordings of one take never land on different sides of a split.
    MIC_TYPES = ("wav_arrayMic", "wav_headMic")

    # Raw WAVs outside this range are skipped: M05 Session2 has ~10 ms head-mic
    # stubs, and one F03 head-mic "no" is a 194 s recording (single words are <= 8.4 s)
    MIN_AUDIO_DURATION = 0.1   # seconds
    MAX_AUDIO_DURATION = 10.0  # seconds
    
    # -------------------------------------------------------------------------
    # AUGMENTATION
    # -------------------------------------------------------------------------
    # Training-only and on the fly. Settings live next to their code as frozen
    # dataclasses: TorgoAugParams (src/data/torgo_augment.py), BCResNetAugParams
    # (src/data/speech_commands_augment.py), SpecAugment (src/model/frontend.py).
    
    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------
    # Fixed per-model recipes live next to their code and are never tuned on
    # held-out results: src/training/ssl_recipe.py (pretrained speech
    # backbones) and src/training/bcresnet_recipe.py (BC-ResNet).


# Create a default config instance
config = Config()
