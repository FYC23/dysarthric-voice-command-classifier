# Deploying to Hugging Face Spaces

This guide explains how to deploy the Dysarthric Voice Command Classifier to Hugging Face Spaces.

## Quick Start

### 1. Create a New Space

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose a name (e.g., `dysarthric-voice-commands`)
3. Select **Gradio** as the SDK
4. Select visibility (public or private)
5. Choose hardware:
   - **CPU Basic (Free)** - Works, but slower first load
   - **CPU Upgrade** - Recommended for better performance
   - **GPU** - Not necessary for inference

### 2. Upload Required Files

Upload these files to your Space (via web UI or Git):

```
your-space/
├── app.py                           # Main entry point (use the one in root)
├── requirements.txt                 # Copy from requirements_hf.txt
├── src/
│   ├── __init__.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py
│   └── model/
│       ├── __init__.py
│       └── architecture.py
├── phase_b_curriculum_trained.pt    # Copy from outputs/
└── label_mapping.json               # Copy from outputs/
```

### 3. Using Git (Recommended)

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy required files
cp /path/to/dysarthric-voice-cmds/app.py .
cp /path/to/dysarthric-voice-cmds/requirements_hf.txt requirements.txt

# Copy source modules
mkdir -p src/inference src/model
cp /path/to/dysarthric-voice-cmds/src/__init__.py src/
cp /path/to/dysarthric-voice-cmds/src/inference/__init__.py src/inference/
cp /path/to/dysarthric-voice-cmds/src/inference/predictor.py src/inference/
cp /path/to/dysarthric-voice-cmds/src/model/__init__.py src/model/
cp /path/to/dysarthric-voice-cmds/src/model/architecture.py src/model/

# Copy model files
cp /path/to/dysarthric-voice-cmds/outputs/phase_b_curriculum_trained.pt .
cp /path/to/dysarthric-voice-cmds/outputs/label_mapping.json .

# Setup Git LFS for large files (model checkpoint)
git lfs install
git lfs track "*.pt"

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

### 4. Using the Web UI

1. Go to your Space's **Files** tab
2. Click **+ Add file** → **Upload files**
3. Upload all the files listed above
4. For the `.pt` model file, ensure Git LFS is enabled

## Alternative: Host Model on Hugging Face Hub

For cleaner separation, you can upload your model to the HF Hub:

### Upload Model to Hub

```python
from huggingface_hub import HfApi

api = HfApi()

# Create a model repository
api.create_repo("YOUR_USERNAME/dysarthric-voice-model", repo_type="model")

# Upload files
api.upload_file(
    path_or_fileobj="outputs/phase_b_curriculum_trained.pt",
    path_in_repo="phase_b_curriculum_trained.pt",
    repo_id="YOUR_USERNAME/dysarthric-voice-model"
)
api.upload_file(
    path_or_fileobj="outputs/label_mapping.json",
    path_in_repo="label_mapping.json",
    repo_id="YOUR_USERNAME/dysarthric-voice-model"
)
```

### Update app.py to Load from Hub

```python
from huggingface_hub import hf_hub_download

# Download model files from Hub
CHECKPOINT_PATH = hf_hub_download(
    repo_id="YOUR_USERNAME/dysarthric-voice-model",
    filename="phase_b_curriculum_trained.pt"
)
LABEL_MAPPING_PATH = hf_hub_download(
    repo_id="YOUR_USERNAME/dysarthric-voice-model",
    filename="label_mapping.json"
)
```

## Space Configuration (Optional)

Create a `README.md` with Space metadata:

```yaml
---
title: Dysarthric Voice Command Classifier
emoji: 🎤
colorFrom: rose
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Dysarthric Voice Command Classifier

A HuBERT-based model for recognizing voice commands from dysarthric speech.
```

## Troubleshooting

### Build Fails

- Check `requirements.txt` for typos
- Ensure all `__init__.py` files are present
- View build logs in the Space's **Settings** tab

### Model Loading Issues

- Verify `.pt` file was uploaded with Git LFS
- Check file paths match in `app.py`

### Out of Memory

- Upgrade to a paid Space with more RAM
- The model requires ~4GB RAM for inference

## Resources

- [Gradio on HF Spaces](https://huggingface.co/docs/hub/spaces-sdks-gradio)
- [Sharing Your App Guide](https://www.gradio.app/guides/sharing-your-app#hosting-on-hf-spaces)
- [Git LFS Guide](https://huggingface.co/docs/hub/repositories-getting-started#terminal)
