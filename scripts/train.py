#!/usr/bin/env python
"""
Training script for dysarthric voice command classifier.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config

print("Training script placeholder.")
print("For full training, use the main.ipynb notebook.")
print(f"Output directory: {config.OUTPUT_DIR}")