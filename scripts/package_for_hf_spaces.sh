#!/bin/bash
# Script to package the app for Hugging Face Spaces deployment

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/hf_spaces_package"

echo -e "${BLUE}Packaging for Hugging Face Spaces...${NC}"

# Create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/src/inference"
mkdir -p "$OUTPUT_DIR/src/model"

# Copy main app file
echo "Copying app.py..."
cp "$PROJECT_ROOT/app.py" "$OUTPUT_DIR/"

# Copy requirements
echo "Copying requirements..."
cp "$PROJECT_ROOT/requirements_hf.txt" "$OUTPUT_DIR/requirements.txt"

# Copy source modules
echo "Copying source modules..."
cp "$PROJECT_ROOT/src/__init__.py" "$OUTPUT_DIR/src/"
cp "$PROJECT_ROOT/src/inference/__init__.py" "$OUTPUT_DIR/src/inference/"
cp "$PROJECT_ROOT/src/inference/predictor.py" "$OUTPUT_DIR/src/inference/"
cp "$PROJECT_ROOT/src/model/__init__.py" "$OUTPUT_DIR/src/model/"
cp "$PROJECT_ROOT/src/model/architecture.py" "$OUTPUT_DIR/src/model/"

# Copy model files
echo "Copying model checkpoint and label mapping..."
cp "$PROJECT_ROOT/outputs/phase_b_curriculum_trained.pt" "$OUTPUT_DIR/"
cp "$PROJECT_ROOT/outputs/label_mapping.json" "$OUTPUT_DIR/"

# Create Space README with metadata
echo "Creating README.md with Space metadata..."
cat > "$OUTPUT_DIR/README.md" << 'EOF'
---
title: Dysarthric Voice Command Classifier
emoji: 🎤
colorFrom: rose
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# 🎤 Dysarthric Voice Command Classifier

A HuBERT-based model for recognizing voice commands from dysarthric speech.

## Features

- **20 voice commands**: digits (0-9) and commands (yes, no, up, down, left, right, forward, back, select, menu)
- **Trained on TORGO dataset** using curriculum learning
- **~87% accuracy** on dysarthric speech

## Usage

1. Click the microphone icon to record your voice
2. Or upload an audio file
3. The model will automatically classify the command

## Model Details

- **Base Model**: [HuBERT Large](https://huggingface.co/facebook/hubert-large-ls960-ft)
- **Training**: Curriculum learning (control speakers → dysarthric speakers)
- **Evaluation**: Leave-one-speaker-out cross-validation
EOF

# Create .gitattributes for LFS
echo "Creating .gitattributes for Git LFS..."
cat > "$OUTPUT_DIR/.gitattributes" << 'EOF'
*.pt filter=lfs diff=lfs merge=lfs -text
EOF

echo -e "${GREEN}✓ Package created at: $OUTPUT_DIR${NC}"
echo ""
echo "Next steps:"
echo "1. Create a new Space at https://huggingface.co/new-space"
echo "2. Clone your Space: git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE"
echo "3. Copy contents: cp -r $OUTPUT_DIR/* YOUR_SPACE/"
echo "4. Install Git LFS: git lfs install"
echo "5. Push to HF: cd YOUR_SPACE && git add . && git commit -m 'Initial deploy' && git push"
