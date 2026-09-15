"""Experiment ladder summarised in REPORT.ipynb.

    python -m delivery.experiments [--cv 5] [--seed 0] [--out outputs/experiments.json]

Each entry is a named variant of a candidate pipeline; all are evaluated with the same
``DeliveryEvaluator`` (shuffled K-fold, out-of-fold MAE). Runs sequentially so timings
are comparable; expect ~10-20 min on a laptop CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline

from .data import load_train
from .evaluation import DeliveryEvaluator, leaderboard, md_table
from .models import TorchMLPRegressor, make_candidate, make_pipeline
from .preprocessing import (DeliveryCleaner, DeliveryFeatureEngineer, NOMINAL_FEATURES,
                            make_preprocessor)


class DropColumns(DeliveryFeatureEngineer):
    """Feature engineer that blanks out a set of engineered columns (ablation helper)."""

    def __init__(self, drop=(), drop_ids=True):
        self.drop = drop
        super().__init__(drop_ids=drop_ids)

    def transform(self, X):
        f = super().transform(X)
        for c in self.drop:   # constant column == feature removed (all-NaN would break HGB binning)
            f[c] = 0.0 if f[c].dtype.kind == "f" else "unknown"
        return f


def variants(seed: int) -> dict[str, Pipeline]:
    hgb_default = make_pipeline(
        HistGradientBoostingRegressor(categorical_features=NOMINAL_FEATURES, random_state=seed), "tree")
    hgb_l1 = make_pipeline(
        HistGradientBoostingRegressor(loss="absolute_error", categorical_features=NOMINAL_FEATURES,
                                      random_state=seed), "tree")
    hgb_tuned = make_candidate("hgb", seed)
    hgb_no_dist = Pipeline([
        ("clean", DeliveryCleaner()),
        ("features", DropColumns(drop=("distance_km", "coords_missing"))),
        ("encode", make_preprocessor("tree")),
        ("model", hgb_tuned.named_steps["model"]),
    ])
    hgb_no_time = Pipeline([
        ("clean", DeliveryCleaner()),
        ("features", DropColumns(drop=("order_hour", "order_minute_of_day", "picked_minute_of_day",
                                       "prep_time_min", "order_time_missing"))),
        ("encode", make_preprocessor("tree")),
        ("model", hgb_tuned.named_steps["model"]),
    ])
    hgb_log = make_pipeline(TransformedTargetRegressor(
        regressor=hgb_tuned.named_steps["model"], func=np.log1p, inverse_func=np.expm1), "tree")

    mlp_default = make_candidate("mlp", seed)                                   # huber
    mlp_mse = make_candidate("mlp", seed).set_params(model__loss="mse")
    mlp_l1 = make_candidate("mlp", seed).set_params(model__loss="l1")
    mlp_wide = make_candidate("mlp", seed).set_params(model__hidden=(512, 256, 128), model__dropout=0.2,
                                                     model__epochs=120, model__patience=15)
    mlp_small = make_candidate("mlp", seed).set_params(model__hidden=(64,), model__dropout=0.0)

    blend = VotingRegressor([("hgb", make_candidate("hgb", seed)), ("mlp", make_candidate("mlp", seed))],
                            weights=[0.7, 0.3])
    return {
        "hgb: default (squared loss, 100 iters)": hgb_default,
        "hgb: absolute_error loss": hgb_l1,
        "hgb: tuned (lr .05, 600 iters, 63 leaves, min_leaf 40)": hgb_tuned,
        "hgb: tuned, log1p target": hgb_log,
        "hgb: tuned, without distance": hgb_no_dist,
        "hgb: tuned, without order/pickup time": hgb_no_time,
        "mlp: (256,128) Huber, early stop": mlp_default,
        "mlp: same, MSE loss": mlp_mse,
        "mlp: same, L1 loss": mlp_l1,
        "mlp: (512,256,128) dropout .2": mlp_wide,
        "mlp: (64,) no dropout": mlp_small,
        "blend: 0.7 hgb + 0.3 mlp": blend,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma separated substring filters on variant names")
    ap.add_argument("--n-rows", type=int, default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "outputs" / "experiments.json"))
    args = ap.parse_args(argv)

    X, y = load_train()
    if args.n_rows:
        idx = np.random.default_rng(args.seed).choice(len(X), size=args.n_rows, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

    ev = DeliveryEvaluator(cv=args.cv, random_state=args.seed)
    todo = variants(args.seed)
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        todo = {n: p for n, p in todo.items() if any(k in n for k in keys)}
    board, reports = ev.compare(todo, X, y)
    cols = ["mae", "rmse", "within_5min", "bias", "mae_fold_std", "fit_time_s"]
    print("\n=== experiments (out-of-fold, %d-fold) ===" % args.cv)
    print(md_table(board[cols]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cv": args.cv, "seed": args.seed, "n_rows": int(len(X)),
                               "results": board[cols].reset_index().to_dict(orient="records")}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
