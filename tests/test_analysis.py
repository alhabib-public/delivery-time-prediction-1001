import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from delivery.analysis import (BaselineBuilder, CategoricalEffectReporter, CorrelationReporter, DataReporter,
                               ErrorAnalysisReporter, ExperimentsReporter, FeatureSummaryReporter,
                               LeaderboardReporter, MetricsBuilder, MissingnessReporter, PostAnalysisReporter,
                               PreAnalysisReporter, SubmissionReporter, TargetReporter, TrainingReporter,
                               display_dataframe, display_figure, display_info_box, display_markdown,
                               display_title, in_notebook)
from delivery.preprocessing import DeliveryCleaner, DeliveryFeatureEngineer
from delivery.evaluation import DeliveryEvaluator, leaderboard
from delivery.models import make_candidate
from delivery.pipeline import ExecutionResult


def test_display_helpers_text_fallback(capsys):
    assert in_notebook() is False
    display_title("#", "phase")
    display_title("##", "analysis")
    display_markdown("**md**")
    display_info_box("hello", kind="success")
    display_dataframe(pd.DataFrame({"mae": [1.0, 2.0]}, index=["a", "b"]), "tbl", caption="cap")
    import matplotlib.pyplot as plt
    fig, _ = plt.subplots()
    display_figure(fig)
    out = capsys.readouterr().out
    assert "# phase" in out and "## analysis" in out and "#### tbl" in out
    assert "[success] hello" in out and "| mae |" in out and "(cap)" in out and "**md**" in out
    with pytest.raises(ValueError):
        display_title("######", "too deep")


def test_metrics_builder():
    tab = MetricsBuilder().build([10, 20, 30], [12, 20, 24], name="m")
    assert list(tab.index) == ["m"] and tab.loc["m", "mae"] == pytest.approx(8 / 3)
    assert "within_5min" in tab.columns


def test_baseline_builder():
    b = BaselineBuilder(methods=("median",), random_state=1).build()
    assert list(b) == ["median"] and hasattr(b["median"], "fit")
    with pytest.raises(ValueError):
        BaselineBuilder(methods=("nope",))


@pytest.fixture(scope="module")
def small_result(raw_sample_module):
    X, y = raw_sample_module
    ev = DeliveryEvaluator(cv=2, random_state=0)
    board, reports = ev.compare({"median": make_candidate("median"), "ridge": make_candidate("ridge")},
                                X, y, verbose=False)
    exp = {"ridge (variant)": reports["ridge"]}
    diag = {"split": "80/20", "holdout": MetricsBuilder().build(y, y + 1, name="ridge"),
            "mlp": {"name": "mlp", "n_iter": 3, "epochs": 5, "loss": "huber", "patience": 2,
                    "loss_curve": [1.0, 0.8, 0.7], "val_curve": [3.0, 2.5, 2.6],
                    "best_val_mae": 2.5, "holdout_mae": 2.6}}
    sub = pd.DataFrame({"ID": ["a", "b"], "Time_taken (min)": [20.0, 30.0]})
    meta = {"n_train": len(X), "n_test": 2, "target_mean": float(y.mean()), "target_std": float(y.std()),
            "target_min": float(y.min()), "target_max": float(y.max()), "seed": 0, "cv": "KFold(2)",
            "candidates": ["median", "ridge"]}
    return X, y, ExecutionResult(exp_name="test_exp", board=board, reports=reports, best_name=board.index[0],
                                 experiments=exp, diagnostics=diag, model=None, submission=sub,
                                 paths={"output_dir": "/tmp/none"}, meta=meta)


@pytest.mark.parametrize("reporter", [DataReporter(), LeaderboardReporter(), TrainingReporter(),
                                      ExperimentsReporter(), ErrorAnalysisReporter(), SubmissionReporter()])
def test_each_reporter_runs_in_text_mode(small_result, reporter, capsys):
    X, y, result = small_result
    reporter.display(result, X=X, y=y)
    assert capsys.readouterr().out.strip()


def test_post_analysis_reporter_sections(small_result, capsys):
    X, y, result = small_result
    PostAnalysisReporter(training_reporter=None).display(result, X=X, y=y)
    out = capsys.readouterr().out
    assert "# Post-analysis: test_exp" in out
    for title in ("[Analysis] Data", "[Analysis] Cross-validated leaderboard", "[Analysis] Experiments",
                  "[Analysis] Error analysis", "[Analysis] Submission"):
        assert title in out
    assert "[Analysis] Training diagnostics" not in out


def test_reporters_handle_missing_parts(small_result, capsys):
    X, y, result = small_result
    bare = ExecutionResult(exp_name="bare", board=result.board, reports=result.reports, best_name=result.best_name,
                           experiments={}, diagnostics=None, submission=None,
                           paths={"output_dir": "/tmp/none"}, meta=result.meta)
    TrainingReporter().display(bare)
    ExperimentsReporter().display(bare)
    SubmissionReporter().display(bare)
    out = capsys.readouterr().out
    assert out.count("[warning]") == 3


@pytest.fixture(scope="module")
def pre_context(raw_sample_module):
    X, y = raw_sample_module
    features = DeliveryFeatureEngineer().fit_transform(DeliveryCleaner().fit_transform(X))
    return dict(X=X, y=y, features=features)


@pytest.mark.parametrize("reporter", [MissingnessReporter(), TargetReporter(), FeatureSummaryReporter(),
                                      CorrelationReporter(), CategoricalEffectReporter()])
def test_each_preanalysis_reporter_runs_in_text_mode(pre_context, reporter, capsys):
    reporter.display(**pre_context)
    assert capsys.readouterr().out.strip()


def test_preanalysis_composite(pre_context, capsys):
    PreAnalysisReporter(correlation_reporter=None).display(exp_name="t", **pre_context)
    out = capsys.readouterr().out
    assert "# Pre-analysis: t" in out
    for title in ("[Analysis] Missing values", "[Analysis] Target", "[Analysis] Engineered features",
                  "[Analysis] Categorical effects"):
        assert title in out
    assert "[Analysis] Correlations" not in out
    assert "Road_traffic_density" in out and "unknown" in out


def test_missingness_reporter_reports_no_missing(capsys):
    X = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    MissingnessReporter().display(X=X, y=np.array([1.0, 2.0]))
    assert "no missing-value literals" in capsys.readouterr().out
