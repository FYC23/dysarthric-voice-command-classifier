"""Tests for the Hugging Face Space bundle builder."""

import importlib.util
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from tests.demo_factories import save_checkpoint

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_hf_space.py"


def load_script():
    spec = importlib.util.spec_from_file_location("build_hf_space", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script()


@pytest.fixture
def deploy(tmp_path):
    (tmp_path / "run").mkdir()
    return save_checkpoint(tmp_path / "run" / "deploy.pt", tau=8.0)


def test_bundle_has_every_file(script, tmp_path, deploy):
    out = tmp_path / "space"
    script.build(out, deploy)
    for name in script.BUNDLE_FILES + (script.WEIGHTS, "requirements.txt", "README.md",
                                       script.MARKER):
        assert (out / name).is_file(), name


def test_fold_checkpoint_is_refused(script, tmp_path, deploy):
    fold = tmp_path / "run" / "fold1_F01.pt"
    shutil.copy(deploy, fold)
    with pytest.raises(ValueError, match="fold"):
        script.build(tmp_path / "space", fold)
    assert not (tmp_path / "space").exists()


def test_other_width_is_refused(script, tmp_path):
    (tmp_path / "run").mkdir()
    small = save_checkpoint(tmp_path / "run" / "deploy.pt", tau=1.0)
    with pytest.raises(ValueError, match="BC-ResNet-1"):
        script.build(tmp_path / "space", small)


def test_existing_folder_that_is_not_a_bundle_is_left_alone(script, tmp_path, deploy):
    out = tmp_path / "space"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        script.build(out, deploy)
    assert (out / "keep.txt").read_text() == "mine"


def test_rebuild_replaces_the_previous_bundle(script, tmp_path, deploy):
    out = tmp_path / "space"
    script.build(out, deploy)
    (out / "stale.txt").write_text("old")
    script.build(out, deploy)
    assert not (out / "stale.txt").exists()


def test_requirements_pin_the_installed_versions(script):
    lines = script.requirements_txt().splitlines()
    assert lines == [f"{name}=={metadata.version(name)}" for name in script.PINNED]
    assert not any(line.startswith("gradio") for line in lines)


def test_space_readme_header(script):
    readme = script.space_readme()
    assert readme.startswith("---\n")
    assert f"sdk_version: {metadata.version('gradio')}\n" in readme
    assert "app_file: app.py\n" in readme
    assert "Rudzicz" in readme and "non-commercial" in readme


def test_bundle_runs_on_its_own(script, tmp_path, deploy):
    out = tmp_path / "space"
    script.build(out, deploy)
    code = (
        "import pathlib, src, app\n"
        "assert pathlib.Path(src.__file__).resolve().parent.parent == pathlib.Path.cwd().resolve()\n"
        "assert app.default_checkpoint() == app.SPACE_CHECKPOINT\n"
        "from src.demo.kws import KeywordSpotter\n"
        "assert KeywordSpotter.from_checkpoint(app.SPACE_CHECKPOINT).tau == 8.0\n"
        "from src.demo.results_table import load_results\n"
        "load_results()\n"
        "import src.baselines.asr.transcribers\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run([sys.executable, "-c", code], cwd=out, env=env,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_requirements_include_librosa_for_parakeets_feature_extractor(script):
    # transformers' ParakeetFeatureExtractor imports librosa; without it the Space's
    # Parakeet panel only shows a load error
    assert any(line.startswith("librosa==") for line in script.requirements_txt().splitlines())
