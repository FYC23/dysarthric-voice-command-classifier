"""scripts/run_asr_baseline.py argument handling (the pipeline is tested separately)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_asr_baseline.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_asr_baseline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_defaults_are_float32_and_auto_device():
    args = load_script().parse_args(["--model", "whisper-large-v3"])
    assert (args.model, args.dtype, args.device, args.batch_size) == (
        "whisper-large-v3", "float32", None, 16)


def test_unknown_model_is_rejected():
    with pytest.raises(SystemExit):
        load_script().parse_args(["--model", "parakeet-tdt-0.6b-v2"])
