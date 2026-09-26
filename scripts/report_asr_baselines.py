#!/usr/bin/env python
"""
Step 1 report from the saved ASR runs (scripts/run_asr_baseline.py):
results table, per-speaker table, accuracy-vs-MACs and confusion figures,
array mic in the output folder and head mic in its head-mic/ subfolder.

Usage:
    python scripts/report_asr_baselines.py [--out outputs/step1-asr]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.asr.report import write_asr_report
from src.baselines.asr.transcribers import MODELS
from src.config import config


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Report the zero-shot ASR baselines")
    parser.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "step1-asr")
    args = parser.parse_args(argv)
    write_asr_report(config.RUNS_DIR, args.out, list(MODELS))
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
