"""The combined report: trained models and zero-shot baselines in one set of tables."""

import pandas as pd
import pytest

from src.eval.combined import (
    DEFAULT_QUESTION, PairGroup, every_pair, load_seed_runs, write_combined_report,
)
from src.eval.constants import ARRAY_MIC, HEAD_MIC
from src.eval.cost import CostProfile
from src.eval.io import save_run
from src.eval.schema import Run
from tests.eval.factories import ALL_DYSARTHRIC, dysarthric_preds, loso_folds, speaker_rows

COSTS = {
    "asr": CostProfile(params=600_000_000, macs=16_000_000_000, input_seconds=2.0),
    "kws-a": CostProfile(params=9_000, macs=5_000_000, input_seconds=2.0),
    "kws-b": CostProfile(params=320_000, macs=170_000_000, input_seconds=2.0),
}


def both_mics(correct):
    return pd.concat([dysarthric_preds(correct), dysarthric_preds(correct, mic=HEAD_MIC)],
                     ignore_index=True)


def trained_run(model, seed, correct):
    return Run(model=model, seed=seed, predictions=both_mics(correct),
               fold_train_speakers=loso_folds())


def asr_run(correct, model="asr"):
    controls = pd.DataFrame(speaker_rows("FC01", 10, 9) + speaker_rows("FC01", 10, 9,
                                                                        mic=HEAD_MIC))
    preds = pd.concat([both_mics(correct), controls], ignore_index=True)
    return Run(model=model, seed=0, predictions=preds, zero_shot=True)


def save_seeds(runs_dir, model, seeds, correct):
    for seed in seeds:
        save_run(trained_run(model, seed, correct), runs_dir / model / f"seed{seed}" / "eval")


def test_seeds_load_in_numeric_order_and_unfinished_seeds_are_skipped(tmp_path):
    save_seeds(tmp_path, "kws-a", [10, 2, 0], {})
    (tmp_path / "kws-a" / "seed3").mkdir()  # still training: no eval/ yet
    runs = load_seed_runs(tmp_path, "kws-a")
    assert [r.seed for r in runs] == [0, 2, 10]


def test_seeds_can_be_restricted(tmp_path):
    save_seeds(tmp_path, "kws-a", [0, 1, 2], {})
    assert [r.seed for r in load_seed_runs(tmp_path, "kws-a", seeds=[2, 0])] == [0, 2]


def test_a_model_without_finished_seeds_is_an_error(tmp_path):
    (tmp_path / "kws-a" / "seed0").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="kws-a"):
        load_seed_runs(tmp_path, "kws-a")


def test_a_requested_seed_that_is_not_finished_is_an_error(tmp_path):
    save_seeds(tmp_path, "kws-a", [0], {})
    with pytest.raises(FileNotFoundError, match="seed 1"):
        load_seed_runs(tmp_path, "kws-a", seeds=[0, 1])


def test_a_run_saved_under_another_models_folder_is_rejected(tmp_path):
    save_run(trained_run("kws-b", 0, {}), tmp_path / "kws-a" / "seed0" / "eval")
    with pytest.raises(ValueError, match="kws-b"):
        load_seed_runs(tmp_path, "kws-a")


def report(tmp_path):
    out = tmp_path / "report"
    baselines = {"asr": [asr_run({s: 5 for s in ALL_DYSARTHRIC})]}
    candidates = {"kws-a": [trained_run("kws-a", i, {s: 7 for s in ALL_DYSARTHRIC})
                            for i in range(2)],
                  "kws-b": [trained_run("kws-b", i, {s: 8 for s in ALL_DYSARTHRIC})
                            for i in range(2)]}
    with pytest.warns(UserWarning, match="seed"):
        write_combined_report(baselines, candidates, COSTS, out,
                              secondary={"kws-a": 0.95, "kws-b": 0.98})
    return out


def test_results_list_baselines_then_candidates_with_their_costs(tmp_path):
    table = pd.read_csv(report(tmp_path) / "results.csv")
    assert table["model"].tolist() == ["asr", "kws-a", "kws-b"]
    assert table["params"].tolist() == [600_000_000, 9_000, 320_000]
    assert table["accuracy"].tolist() == pytest.approx([0.5, 0.7, 0.8])


def test_every_candidate_is_compared_with_every_baseline(tmp_path):
    table = pd.read_csv(report(tmp_path) / "comparisons.csv")
    assert list(zip(table["candidate"], table["baseline"])) == [("kws-a", "asr"),
                                                                ("kws-b", "asr")]
    assert table["n_better"].tolist() == [8, 8]
    assert table["mean_diff"].tolist() == pytest.approx([0.2, 0.3])


def test_array_mic_is_the_top_folder_and_head_mic_is_separate(tmp_path):
    out = report(tmp_path)
    assert set(pd.read_csv(out / "results.csv")["mic"]) == {ARRAY_MIC}
    assert set(pd.read_csv(out / "head-mic" / "comparisons.csv")["mic"]) == {HEAD_MIC}
    for folder in (out, out / "head-mic"):
        for name in ("results.md", "per_speaker.csv", "comparisons.md",
                     "accuracy_vs_macs.png", "accuracy_vs_params.png",
                     "confusion_asr.png", "confusion_kws-b.png"):
            assert (folder / name).stat().st_size > 0


def test_a_model_listed_as_both_baseline_and_candidate_is_rejected(tmp_path):
    runs = [asr_run({})]
    with pytest.raises(ValueError, match="asr"):
        write_combined_report({"asr": runs}, {"asr": runs}, COSTS, tmp_path)


def test_the_default_question_is_each_candidate_against_each_baseline(tmp_path):
    table = pd.read_csv(report(tmp_path) / "comparisons.csv")
    assert table["question"].tolist() == [DEFAULT_QUESTION] * 2


def grouped_report(tmp_path, groups, off_curve=()):
    out = tmp_path / "report"
    baselines = {"asr": [asr_run({s: 5 for s in ALL_DYSARTHRIC})]}
    candidates = {m: [trained_run(m, i, {s: k for s in ALL_DYSARTHRIC}) for i in range(3)]
                  for m, k in (("kws-a", 7), ("kws-b", 8))}
    write_combined_report(baselines, candidates, COSTS, out, groups=groups,
                          off_curve=off_curve)
    return out


def test_comparisons_answer_each_titled_question_in_order(tmp_path):
    groups = [every_pair("Trained vs ASR", ["kws-a", "kws-b"], ["asr"]),
              PairGroup("Small vs large", (("kws-a", "kws-b"),))]
    out = grouped_report(tmp_path, groups)
    table = pd.read_csv(out / "comparisons.csv")
    assert table.columns[0] == "question"
    assert list(zip(table["question"], table["candidate"], table["baseline"])) == [
        ("Trained vs ASR", "kws-a", "asr"), ("Trained vs ASR", "kws-b", "asr"),
        ("Small vs large", "kws-a", "kws-b")]
    assert table["mean_diff"].tolist() == pytest.approx([0.2, 0.3, -0.1])
    md = (out / "comparisons.md").read_text()
    assert md.index("## Trained vs ASR") < md.index("## Small vs large")
    assert md.count("| Candidate |") == 2


def test_a_model_cannot_be_both_sides_of_one_question():
    with pytest.raises(ValueError, match="kws-a"):
        PairGroup("Mixed", (("kws-a", "kws-b"), ("kws-b", "kws-a")))
    with pytest.raises(ValueError, match="kws-a"):
        PairGroup("Self", (("kws-a", "kws-a"),))


def test_a_question_needs_at_least_one_pair():
    with pytest.raises(ValueError, match="Empty"):
        PairGroup("Empty", ())


def test_a_pair_naming_a_model_outside_the_report_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="kws-z"):
        grouped_report(tmp_path, [PairGroup("Typo", (("kws-z", "asr"),))])


def test_off_curve_models_are_left_off_the_cost_figures_only(tmp_path, monkeypatch):
    plotted = []

    def record(summaries, costs, path, axis="macs", *args, **kwargs):
        plotted.append((axis, [s.model for s in summaries]))

    monkeypatch.setattr("src.eval.plots.plot_accuracy_vs_cost", record)
    monkeypatch.setattr("src.eval.combined.plot_accuracy_vs_cost", record)
    out = grouped_report(tmp_path, [every_pair("All", ["kws-a", "kws-b"], ["asr"])],
                         off_curve=["kws-b"])
    assert sorted(plotted) == [("macs", ["asr", "kws-a"])] * 2 + [("params", ["asr", "kws-a"])] * 2
    assert "kws-b" in pd.read_csv(out / "results.csv")["model"].tolist()
    assert (out / "confusion_kws-b.png").stat().st_size > 0


def test_an_unknown_off_curve_model_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="kws-z"):
        grouped_report(tmp_path, None, off_curve=["kws-z"])
