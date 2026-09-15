import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

from delivery.evaluation import (DEFAULT_SLICES, SCORERS, DeliveryEvaluator, DeliverySlicer,
                                 EvaluationReport, ResidualDisplay, SliceErrorDisplay, compute_metrics,
                                 md_table, plot_report, save_report, slice_table, style_leaderboard,
                                 style_slices, style_table, within_tolerance)
from delivery.models import make_candidate, make_pipeline


def test_metrics_on_toy_arrays():
    y = np.array([10.0, 20.0, 30.0]); p = np.array([12.0, 20.0, 24.0])
    m = compute_metrics(y, p)
    assert m["mae"] == pytest.approx(8 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 0 + 36) / 3))
    assert m["within_5min"] == pytest.approx(2 / 3)
    assert m["bias"] == pytest.approx((2 + 0 - 6) / 3)
    assert within_tolerance(y, p, tol=1) == pytest.approx(1 / 3)


def test_scorers_follow_sklearn_sign_convention():
    X = np.zeros((20, 1)); y = np.arange(20, dtype=float)
    scores = cross_val_score(DummyRegressor(strategy="mean"), X, y, cv=2, scoring=SCORERS["mae"])
    assert (scores <= 0).all()                     # neg MAE
    assert SCORERS["within_5min"]._sign == 1


def test_slicer_labels(toy_raw):
    S = DeliverySlicer().fit_transform(toy_raw)
    assert list(S.columns) == list(DEFAULT_SLICES)
    assert S["traffic"].tolist() == ["High", "Jam", "unknown", "Low"]
    assert S["distance"].tolist()[0] == "2-5 km" and S["distance"].tolist()[2] == "unknown"
    assert S["time_of_day"].tolist()[0] == "lunch (11-14)" and S["time_of_day"].tolist()[2] == "unknown"
    assert S["multiple_deliveries"].tolist()[1] == "1 extra"
    with pytest.raises(ValueError):
        DeliverySlicer(slices=("nope",)).fit_transform(toy_raw)


def test_slice_table_counts_sum():
    y = np.arange(10.0); p = y + 1; g = np.array(list("aabbbcccc") + ["a"])
    t = slice_table(y, p, g)
    assert t["n"].sum() == 10 and t["share"].sum() == pytest.approx(1)
    assert t.index[0] == "c" and (t["mae"] == 1).all() and (t["bias"] == 1).all()
    assert list(slice_table(y, p, g, order=["a", "b"]).index) == ["a", "b", "c"]


def test_evaluator_cv_report(raw_sample):
    X, y = raw_sample
    ev = DeliveryEvaluator(cv=3, random_state=0)
    rep = ev.evaluate(make_candidate("median"), X, y, name="median")
    assert isinstance(rep, EvaluationReport)
    assert len(rep.y_pred) == len(y) and np.isfinite(rep.y_pred).all()
    assert rep.folds.shape[0] == 3 and (rep.folds["test_mae"] > 0).all()
    assert set(rep.slices.columns) == set(DEFAULT_SLICES)
    for s in DEFAULT_SLICES:
        assert rep.slice_table(s)["n"].sum() == len(y)
    assert "| model |" in md_table(rep.summary_frame()) or "|  |" in md_table(rep.summary_frame())
    assert rep.to_markdown(slices=("traffic",)).startswith("### median")


def test_evaluator_grouped_cv_and_compare(raw_sample, tmp_path):
    X, y = raw_sample
    groups = X["Delivery_person_ID"].str.strip().to_numpy()
    ev = DeliveryEvaluator(cv=GroupKFold(n_splits=3, shuffle=True, random_state=0))
    cands = {"median": make_candidate("median"), "ridge": make_candidate("ridge")}
    board, reports = ev.compare(cands, X, y, groups=groups, verbose=False)
    assert list(board.index) == sorted(board.index, key=lambda k: board.loc[k, "mae"])
    assert board.loc["ridge", "mae"] <= board.loc["median", "mae"]
    assert reports["ridge"].method.startswith("GroupKFold")
    path = save_report(reports, board, tmp_path, figures=True)
    assert path.exists() and (tmp_path / "figures" / "ridge.png").exists()


def test_holdout_report(raw_sample):
    X, y = raw_sample
    rep = DeliveryEvaluator().holdout(make_candidate("median"), X[:400], y[:400], X[400:], y[400:])
    assert rep.method == "holdout" and len(rep.y_true) == 100 and hasattr(rep, "estimator_")


def test_displays_follow_sklearn_convention(raw_sample):
    X, y = raw_sample
    y = y.to_numpy(); p = y + np.random.default_rng(0).normal(size=len(y))
    labels = DeliverySlicer(slices=("traffic",)).fit_transform(X)["traffic"]
    d = SliceErrorDisplay.from_predictions(y, p, labels, slice_name="traffic")
    assert d.ax_ is not None and d.figure_ is not None and len(d.bars_) == d.table_.shape[0]
    r = ResidualDisplay.from_predictions(y, p, kind="vs_predicted")
    assert r.ax_.get_xlabel().startswith("predicted")
    est = make_candidate("median").fit(X, y)
    d2 = SliceErrorDisplay.from_estimator(est, X, y, slice_name="city")
    assert d2.slice_name == "city"
    rep = DeliveryEvaluator(cv=2).evaluate(est, X, y)
    fig = plot_report(rep, slices=("traffic", "distance"))
    assert len([a for a in fig.axes if a.get_visible()]) == 4


def test_style_helpers_render_html(raw_sample):
    X, y = raw_sample
    ev = DeliveryEvaluator(cv=2)
    _, reports = ev.compare({"median": make_candidate("median"), "ridge": make_candidate("ridge")}, X, y, verbose=False)
    from delivery.evaluation import leaderboard
    html = style_leaderboard(leaderboard(reports)).to_html()
    assert "<table" in html and "ridge" in html and "%" in html
    html = style_slices(reports["ridge"], "traffic").to_html()
    assert "Low" in html or "unknown" in html
    assert "<table" in style_table(pd.DataFrame({"n": [3, 4], "share": [0.3, 0.7], "bias": [-1.0, 2.0]})).to_html()


def test_style_table_bars_every_numeric_column():
    df = pd.DataFrame({"n": [3, 4, 5], "mean_target": [20.0, 30.0, 40.0], "bias": [-1.0, 0.5, 2.0],
                       "label": ["a", "b", "c"], "const": [1.0, 1.0, 1.0]})
    html = style_table(df).to_html()
    # one gradient per barred cell; a zero-width bar (e.g. the column minimum) renders without one
    assert 7 <= html.count("linear-gradient") <= 9       # 3 rows x (n, mean_target, bias)
    assert "const" in html
    assert style_table(df, bars=False).to_html().count("linear-gradient") == 0
    assert style_table(df.head(1)).to_html().count("linear-gradient") == 0   # single row: no bars
    assert 2 <= style_table(df, bars=("n",)).to_html().count("linear-gradient") <= 3


def test_style_table_tolerates_non_unique_index_and_columns():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=["a", "a", "b"])
    html = style_table(df).to_html()
    assert "linear-gradient" in html and ">a<" in html
    dup = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["x", "x"])
    assert "x_1" in style_table(dup).to_html()
