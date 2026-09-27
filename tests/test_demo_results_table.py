"""Tests for the demo's results table, read from the harness's results.csv."""

import csv

import pytest

from src.demo.results_table import RESULTS_CSV, format_params, load_results

COLUMNS = ["model", "mic", "n_seeds", "accuracy", "ci_low", "ci_high", "severe", "mild",
           "params", "macs"]
ROWS = [
    # Values sit away from .x5 rounding boundaries so the expected strings are exact
    ["whisper-large-v3-lenient", "wav_arrayMic", 1, 0.6627, 0.5360, 0.8021, 0.5413, 0.8973,
     1543490560, 1299791185920],
    ["parakeet-tdt-0.6b-v3-lenient", "wav_arrayMic", 1, 0.7063, 0.5896, 0.8378, 0.5621,
     0.9118, 627008134, 16749277056],
    ["bcresnet-8", "wav_arrayMic", 1, 0.8838, 0.8190, 0.9510, 0.8120, 0.9410,
     323124, 170986976],
    # Head mic: must never reach the demo table
    ["bcresnet-8", "wav_headMic", 1, 0.9240, 0.8800, 0.9700, 0.9000, 0.9500,
     323124, 170986976],
]


def write_csv(path, rows=ROWS, columns=COLUMNS):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def test_table_rows_are_formatted_like_the_readme(tmp_path):
    table = load_results(write_csv(tmp_path / "results.csv")).table_markdown.splitlines()
    assert table[0] == "| Model | Params | MACs / 2 s | Accuracy | 95% CI | Severe | Mild |"
    assert table[2] == ("| Whisper large-v3 (zero-shot) | 1.5B | 1.3T | 66.3% | [53.6, 80.2] "
                        "| 54.1% | 89.7% |")
    assert table[3] == ("| Parakeet-TDT-0.6B-v3 (zero-shot) | 627M | 16.7G | 70.6% "
                        "| [59.0, 83.8] | 56.2% | 91.2% |")
    assert table[4] == "| BC-ResNet-8 | 323k | 171M | 88.4% | [81.9, 95.1] | 81.2% | 94.1% |"
    assert len(table) == 5


def test_head_mic_rows_are_ignored(tmp_path):
    table = load_results(write_csv(tmp_path / "results.csv")).table_markdown
    assert "92.4%" not in table


def test_panel_costs(tmp_path):
    results = load_results(write_csv(tmp_path / "results.csv"))
    assert (results.kws_cost.params, results.kws_cost.macs) == ("323k", "171M")
    assert (results.asr_cost.params, results.asr_cost.macs) == ("627M", "16.7G")


def test_missing_model_row_raises(tmp_path):
    path = write_csv(tmp_path / "results.csv", rows=ROWS[:2])
    with pytest.raises(ValueError, match="no wav_arrayMic row for bcresnet-8"):
        load_results(path)


def test_missing_column_raises(tmp_path):
    rows = [r[:-1] for r in ROWS]
    path = write_csv(tmp_path / "results.csv", rows=rows, columns=COLUMNS[:-1])
    with pytest.raises(ValueError, match="macs"):
        load_results(path)


def test_empty_value_raises(tmp_path):
    rows = [list(r) for r in ROWS]
    rows[2][6] = ""  # bcresnet-8 severe
    with pytest.raises(ValueError, match="bcresnet-8 has no value for severe"):
        load_results(write_csv(tmp_path / "results.csv", rows=rows))


def test_missing_file_names_the_report_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="report_results.py"):
        load_results(tmp_path / "results.csv")


def test_billions_of_parameters_read_as_b():
    assert format_params(1543490560) == "1.5B"
    assert format_params(323124) == "323k"


def test_the_committed_report_loads():
    results = load_results(RESULTS_CSV)
    assert len(results.table_markdown.splitlines()) == 5
