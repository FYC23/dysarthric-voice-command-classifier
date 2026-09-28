#!/usr/bin/env python
"""
Assemble the Hugging Face Space bundle for the Gradio demo (app.py).

Writes build/hf_space/: app.py, the src/ modules the demo imports, the
BC-ResNet-2 deploy model as weights/bcresnet-2.pt, the results table, a
pinned requirements.txt and the Space README. It never uploads; push the
folder yourself:

    hf upload <user>/<space> build/hf_space --repo-type space

Usage:
    python scripts/build_hf_space.py
    python scripts/build_hf_space.py --checkpoint runs/bcresnet-2/seed0/deploy.pt --out build/hf_space
"""

import argparse
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.demo.kws import KeywordSpotter  # noqa: E402
from src.demo.results_table import KWS_NAME, KWS_RUN  # noqa: E402

DEFAULT_CHECKPOINT = REPO_ROOT / "runs" / KWS_RUN / "seed0" / "deploy.pt"
DEFAULT_OUT = REPO_ROOT / "build" / "hf_space"
DEPLOY_NAME = "deploy.pt"
SPACE_TAU = 2.0
WEIGHTS = f"weights/{KWS_RUN}.pt"
MARKER = ".hf_space_bundle"
# Everything app.py imports, directly or not. tests/test_build_hf_space.py
# imports the demo from a built bundle, so a missing module fails that test.
BUNDLE_FILES = (
    "app.py",
    "src/__init__.py",
    "src/config.py",
    "src/audio.py",
    "src/model/__init__.py",
    "src/model/architecture.py",  # imported by src/model/__init__.py
    "src/model/bcresnet.py",
    "src/model/frontend.py",
    "src/baselines/__init__.py",
    "src/baselines/asr/__init__.py",
    "src/baselines/asr/scoring.py",
    "src/baselines/asr/transcribers.py",
    "src/eval/__init__.py",
    "src/eval/constants.py",
    "src/eval/units.py",
    "src/demo/__init__.py",
    "src/demo/audio_input.py",
    "src/demo/handler.py",
    "src/demo/kws.py",
    "src/demo/results_table.py",
    "outputs/results/results.csv",
)
# Pinned to what the demo was tested with here. Gradio comes from sdk_version.
# librosa: transformers' ParakeetFeatureExtractor imports it.
PINNED = ("torch", "torchaudio", "transformers", "numpy", "scipy", "librosa")


def check_checkpoint(path: Path) -> None:
    """
    The Space serves the model trained on all 8 dysarthric speakers. Fold
    checkpoints carry the same metadata, so the file name is the check.
    """
    if path.name != DEPLOY_NAME:
        raise ValueError(f"{path} is not {DEPLOY_NAME}: the Space serves the deploy model, "
                         f"never a fold checkpoint")
    tau = KeywordSpotter.from_checkpoint(path).tau
    if tau != SPACE_TAU:
        raise ValueError(f"{path} is BC-ResNet-{tau:g}; the Space serves "
                         f"BC-ResNet-{SPACE_TAU:g}")


def requirements_txt() -> str:
    return "".join(f"{name}=={metadata.version(name)}\n" for name in PINNED)


def space_readme() -> str:
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    return f"""---
title: Dysarthric Voice Commands
emoji: 🎙️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: {metadata.version("gradio")}
python_version: "{python}"
app_file: app.py
pinned: false
license: mit
short_description: Tiny keyword spotter vs zero-shot ASR on dysarthric speech
---

# Dysarthric voice commands

Say one of 20 commands (the digits zero to nine; yes, no, up, down, left, right,
forward, back, select, menu). Two models hear the same 2 s clip:

- **{KWS_NAME}**, a small keyword-spotting network pretrained on Speech Commands and
  fine-tuned on TORGO speakers with dysarthria.
- **Parakeet-TDT-0.6B-v3**, an off-the-shelf speech recogniser, zero-shot; its
  transcript is mapped to the nearest command.

BC-ResNet always answers with one of the 20 commands; it cannot say "not a command".
The accuracies on the page are held-out-speaker results from the project's evaluation.

Code, method and full results: {"https://github.com/FYC23/dysarthric-voice-command-classifier"}

## Data and license

The code is MIT-licensed. The BC-ResNet weights were trained on the TORGO database,
whose license allows free academic, non-commercial use; this demo is for
non-commercial research use and contains no TORGO audio.

> Rudzicz, F., Namasivayam, A.K., Wolff, T. (2012) The TORGO database of acoustic and
> articulatory speech from speakers with dysarthria. *Language Resources and Evaluation*,
> 46(4), pages 523-541.
"""


def _prepare_out(out: Path) -> None:
    """Delete a previous bundle (only one this script made), then recreate the folder."""
    if out.exists():
        if not (out / MARKER).is_file():
            raise FileExistsError(f"{out} exists and was not built by this script; "
                                  f"refusing to delete it")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text("Built by scripts/build_hf_space.py; replaced on every build.\n")


def build(out: Path, checkpoint: Path, repo_root: Path = REPO_ROOT) -> List[Path]:
    out, checkpoint = Path(out), Path(checkpoint)
    check_checkpoint(checkpoint)
    missing = [name for name in BUNDLE_FILES if not (repo_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"bundle sources missing: {missing}")
    _prepare_out(out)
    for name in BUNDLE_FILES:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / name, out / name)
    (out / WEIGHTS).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, out / WEIGHTS)
    (out / "requirements.txt").write_text(requirements_txt())
    (out / "README.md").write_text(space_readme())
    return sorted(p for p in out.rglob("*") if p.is_file())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Hugging Face Space bundle")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    files = build(args.out, args.checkpoint)
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"Wrote {len(files)} files ({size_mb:.1f} MB) to {args.out}")
    print(f"Push it with:\n    hf upload <user>/<space> {args.out} --repo-type space")


if __name__ == "__main__":
    main()
