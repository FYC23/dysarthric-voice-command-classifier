"""
TORGO dataset scanning and preprocessing utilities.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def parse_speaker_info(speaker_id: str) -> Tuple[str, bool]:
    """
    Parse speaker ID to extract gender and dysarthria status.
    
    Speaker ID format:
    - F01, F03, F04: Female dysarthric
    - FC01, FC02, FC03: Female control (no dysarthria)
    - M01-M05: Male dysarthric
    - MC01-MC04: Male control (no dysarthria)
    """
    is_dysarthric = 'C' not in speaker_id
    gender = 'F' if speaker_id.startswith('F') else 'M'
    return gender, is_dysarthric


def scan_torgo_dataset(torgo_root: Path, target_commands: List[str], mic_type: str = "wav_arrayMic") -> pd.DataFrame:
    """
    Scan TORGO dataset and extract samples matching target command words.
    
    Returns DataFrame with columns: file_path, speaker_id, session, utterance_id, 
                                    label, gender, is_dysarthric
    """
    samples = []
    target_commands_lower = [cmd.lower() for cmd in target_commands]
    
    # Scan each speaker folder (F, FC, M, MC)
    for group_dir in torgo_root.iterdir():
        if not group_dir.is_dir() or group_dir.name.startswith('.'):
            continue
            
        # Scan each speaker in the group
        for speaker_dir in group_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            speaker_id = speaker_dir.name
            gender, is_dysarthric = parse_speaker_info(speaker_id)
            
            # Scan each session
            for session_dir in speaker_dir.iterdir():
                if not session_dir.is_dir() or not session_dir.name.startswith('Session'):
                    continue
                    
                session = session_dir.name
                prompts_dir = session_dir / "prompts"
                wav_dir = session_dir / mic_type
                
                if not prompts_dir.exists() or not wav_dir.exists():
                    continue
                
                # Scan each prompt file
                for prompt_file in prompts_dir.glob("*.txt"):
                    utterance_id = prompt_file.stem
                    wav_file = wav_dir / f"{utterance_id}.wav"
                    
                    if not wav_file.exists():
                        continue
                    
                    # Read and normalize the prompt
                    try:
                        prompt_text = prompt_file.read_text().strip().lower()
                    except:
                        continue
                    
                    # Check if it matches a target command (exact match only)
                    first_word = prompt_text.split()[0] if prompt_text else ""
                    
                    if first_word in target_commands_lower:
                        samples.append({
                            'file_path': str(wav_file),
                            'speaker_id': speaker_id,
                            'session': session,
                            'utterance_id': utterance_id,
                            'label': first_word,
                            'gender': gender,
                            'is_dysarthric': is_dysarthric
                        })
    
    df = pd.DataFrame(samples)
    return df


def create_speaker_splits(df: pd.DataFrame, val_speakers: List[str], test_speakers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by speakers for proper evaluation.
    """
    train_df = df[~df['speaker_id'].isin(val_speakers + test_speakers)].copy()
    val_df = df[df['speaker_id'].isin(val_speakers)].copy()
    test_df = df[df['speaker_id'].isin(test_speakers)].copy()
    
    return train_df, val_df, test_df


def create_label_mapping(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create label to ID mappings from dataset.
    """
    labels_in_dataset = sorted(df['label'].unique())
    label2id = {label: idx for idx, label in enumerate(labels_in_dataset)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label