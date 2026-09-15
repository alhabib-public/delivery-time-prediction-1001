import json

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from delivery.analysis import BaselineBuilder, MetricsBuilder, PostAnalysisReporter, PreAnalysisReporter
from delivery.data import ID, SUBMISSION_TARGET, load_sample_submission, load_test
from delivery.evaluation import DeliveryEvaluator
from delivery.models import make_candidate
from delivery.pipeline import ExecutionResult, Experiment, execute
from delivery.preprocessing import DeliveryCleaner


@pytest.fixture(scope="module")
def inputs(raw_sample_module):
    X, y = raw_sample_module
    return dict(X=X, y=y, X_test=load_test(), sample_submission=load_sample_submission())


def _run(inputs, tmp_path, **overrides):
    hgb = make_candidate("hgb", 0).set_params(model__max_iter=20)
    kwargs = dict(
        exp_name="test_exp", **inputs,
        preanalysis_reporter=PreAnalysisReporter(), postanalysis_reporter=PostAnalysisReporter(),
        metrics_builder=MetricsBuilder(), baseline_builder=BaselineBuilder(methods=("median",)),
        cleaner=DeliveryCleaner(), candidates={"hgb": hgb},
        evaluator=DeliveryEvaluator(cv=2, random_state=0),
        extra_experiments={"hgb (variant)": Experiment(pipeline=make_candidate("hgb", 1).set_params(model__max_iter=10))},
        output_dir=tmp_path, seed=0,
    )
    kwargs.update(overrides)
    return execute(**kwargs)


def test_execute_end_to_end(inputs, tmp_path):
    res = _run(inputs, tmp_path)
    assert isinstance(res, ExecutionResult) and res.exp_name == "test_exp"
    assert set(res.reports) == {"median", "hgb"} and res.best_name == "hgb"
    assert res.best.mae == pytest.approx(res.board.loc["hgb", "mae"])
    assert list(res.experiments) == ["hgb (variant)"]
    assert res.diagnostics["holdout"].index.tolist() == ["hgb"] and res.diagnostics["mlp"] is None
    assert res.model is not None and res.submission is not None
    assert list(res.submission.columns) == [ID, SUBMISSION_TARGET] and len(res.submission) == len(inputs["X_test"])
    for key in ("metrics", "submission", "model", "best_model", "run_meta", "figures"):
        assert res.paths[key].exists(), key
    assert res.paths["submission"] == tmp_path / "COMPLETED_PREDICTIONS.csv"
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["exp_name"] == "test_exp" and meta["candidates"] == ["median", "hgb"] and "versions" in meta
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["best"] == "hgb" and "hgb (variant)" in metrics["reports"]


def test_execute_evaluate_only_and_display_text(inputs, tmp_path, capsys):
    res = _run(inputs, tmp_path, refit=False, predict=False, diagnostics=False, extra_experiments=None,
               figures=False, display_preanalysis=True, display_postanalysis=True)
    assert res.model is None and res.submission is None and res.diagnostics is None and res.experiments == {}
    assert "submission" not in res.paths and not (tmp_path / "figures").exists()
    out = capsys.readouterr().out
    assert "# Pre-analysis: test_exp" in out and "[Analysis] Missing values" in out
    assert "# Post-analysis: test_exp" in out and "[Analysis] Submission" in out and "[warning]" in out
    assert out.index("# Pre-analysis") < out.index("Step5:") < out.index("# Post-analysis")


def test_execute_validates_inputs(inputs, tmp_path):
    bad = dict(inputs); bad["X_test"] = inputs["X_test"].drop(columns=["City"])
    with pytest.raises(ValueError, match="columns differ"):
        _run(bad, tmp_path)
    bad = dict(inputs); bad["sample_submission"] = inputs["sample_submission"].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="IDs"):
        _run(bad, tmp_path)
