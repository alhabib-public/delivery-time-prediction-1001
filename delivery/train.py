"""Command line wrapper around an experiment module.

    python -m delivery.train                                  # v001 end-to-end -> COMPLETED_PREDICTIONS.csv
    python -m delivery.train --exp v001_hgb_vs_mlp_kfold5     # explicit experiment
    python -m delivery.train --models hgb,mlp --evaluate-only # subset, leaderboard only
    python -m delivery.train --predict-only                   # reuse outputs/model.joblib
    python -m delivery.train --n-rows 3000 --cv 3             # quick smoke run

The same experiment runs in the notebook with
``from delivery.exp import v001_hgb_vs_mlp_kfold5 as exp;
exp.run(display_preanalysis=True, display_postanalysis=True)``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import joblib
import numpy as np

from .data import ID, data_dir, load_sample_submission, load_test, write_submission
from .evaluation import md_table
from .pipeline.regression import OUTPUTS, SUBMISSION

DEFAULT_EXP = "v001_hgb_vs_mlp_kfold5"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m delivery.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=DEFAULT_EXP, help="module name under delivery/exp/ (default: %(default)s)")
    p.add_argument("--models", default=None, help="comma separated subset of the experiment's candidates")
    p.add_argument("--cv", type=int, default=5, help="number of KFold splits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-rows", type=int, default=None, help="subsample training rows (smoke runs)")
    p.add_argument("--output-dir", default=str(OUTPUTS))
    p.add_argument("--evaluate-only", action="store_true", help="skip refit + submission")
    p.add_argument("--predict-only", action="store_true", help="only predict with outputs/model.joblib")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--preanalysis", action="store_true", help="print the pre-analysis section (text)")
    p.add_argument("--postanalysis", action="store_true", help="print the post-analysis section (text)")
    p.add_argument("--no-experiments", action="store_true", help="skip the log-target / grouped-CV variants")
    p.add_argument("--no-diagnostics", action="store_true", help="skip the holdout fit + MLP curves")
    return p


def predict_only(output_dir: Path) -> int:
    model_path = output_dir / "model.joblib"
    if not model_path.exists():
        print(f"no saved model at {model_path}; run without --predict-only first", file=sys.stderr)
        return 2
    X_test = load_test()
    preds = np.asarray(joblib.load(model_path).predict(X_test), dtype=float)
    path = SUBMISSION if output_dir == OUTPUTS else output_dir / SUBMISSION.name
    sub = write_submission(X_test[ID], preds, path, sample=load_sample_submission())
    print(f"[predict] wrote {path} ({len(sub)} rows, mean={preds.mean():.2f})")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if args.predict_only:
        return predict_only(output_dir)

    try:
        module = importlib.import_module(f"delivery.exp.{args.exp}")
    except ModuleNotFoundError as e:
        print(f"unknown experiment {args.exp!r} ({e}); see delivery/exp/", file=sys.stderr)
        return 2
    print(f"[train] experiment {args.exp} on {data_dir()}")
    result = module.run(
        display_preanalysis=args.preanalysis, display_postanalysis=args.postanalysis, cv=args.cv, seed=args.seed, models=args.models, n_rows=args.n_rows,
        output_dir=output_dir, refit=not args.evaluate_only, predict=not args.evaluate_only,
        diagnostics=not args.no_diagnostics, experiments=not args.no_experiments, figures=not args.no_figures,
    )

    print("\n=== leaderboard (out-of-fold, lower MAE is better) ===")
    print(md_table(result.board))
    print(f"\nbest model: {result.best_name}  (MAE {result.best.mae:.3f})")
    for s in ("traffic", "distance"):
        print(f"\n--- {result.best_name} by {s} ---")
        print(md_table(result.best.slice_table(s)))
    if result.experiments:
        print("\n--- experiments ---")
        for name, rep in result.experiments.items():
            print(f"  {name}: mae={rep.mae:.3f}")
    print("\nartefacts:")
    for k, v in result.paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
