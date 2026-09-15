"""Generic regression pipeline for the delivery-time task.

An experiment module (``delivery/exp/vNNN_*.py``) builds its inputs and calls
:func:`execute` exactly once. ``execute`` runs the numbered phases below, writes the
artefacts under ``output_dir`` and, when ``display_postanalysis`` is on, hands the
:class:`ExecutionResult` to the post-analysis reporter which renders the report.

Phases
------
1. Validation       – columns, submission IDs, candidate names
2. Preparation      – cleaner + feature-engineer pass on the training rows, run metadata
3. Pre-analysis     – reporters on the raw/engineered training data (gated by ``display_preanalysis``)
4. Baselines        – ``baseline_builder.build()`` merged with the experiment's candidates
5. Cross-validation – ``evaluator.compare`` → leaderboard + out-of-fold reports; winner = lowest MAE
6. Diagnostics      – single holdout fit of the non-baseline candidates; PyTorch training curves
7. Experiments      – extra variants evaluated with their own evaluator/groups (e.g. log target, GroupKFold)
8. Refit + predict  – winner on all rows → COMPLETED_PREDICTIONS.csv, model.joblib, best_model.json
9. Dump             – metrics.json, figures/, run_meta.json
10. Post-analysis   – reporters on the results (gated by ``display_postanalysis``)
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .. import __version__
from ..analysis.builders import BaselineBuilder, MetricsBuilder
from ..analysis.reporters import PostAnalysisReporter, PreAnalysisReporter
from ..data import ID, write_submission
from ..evaluation import DeliveryEvaluator, EvaluationReport, leaderboard, save_report
from ..models import TorchMLPRegressor
from ..preprocessing import DeliveryCleaner, DeliveryFeatureEngineer

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS = ROOT / "outputs"
SUBMISSION = ROOT / "COMPLETED_PREDICTIONS.csv"


@dataclass
class Experiment:
    """An extra variant evaluated next to the winner.

    ``evaluator=None`` reuses the main evaluator; ``groups`` enables grouped splitters.
    """

    pipeline: Pipeline
    evaluator: DeliveryEvaluator | None = None
    groups: np.ndarray | None = None


@dataclass
class ExecutionResult:
    exp_name: str
    board: pd.DataFrame
    reports: dict[str, EvaluationReport]
    best_name: str
    experiments: dict[str, EvaluationReport] = field(default_factory=dict)
    diagnostics: dict | None = None
    model: Pipeline | None = None
    submission: pd.DataFrame | None = None
    paths: dict[str, Path] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def best(self) -> EvaluationReport:
        return self.reports[self.best_name]


def _log(exp_name: str, msg: str) -> None:
    print(f"[{exp_name}] {msg}", flush=True)


def execute(
    exp_name: str,
    # Input: for X, y
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    # Reporters: for analysis
    preanalysis_reporter: PreAnalysisReporter | None,
    postanalysis_reporter: PostAnalysisReporter,
    # Builders: for metrics and baselines
    metrics_builder: MetricsBuilder,
    baseline_builder: BaselineBuilder,
    # Cleaner (first step of every candidate; validated once here)
    cleaner: DeliveryCleaner,
    # Model(s): name -> full raw-frame -> prediction pipeline
    candidates: dict[str, Pipeline],
    # Validation
    evaluator: DeliveryEvaluator,
    # Experiments: name -> Experiment
    extra_experiments: dict[str, Experiment] | None = None,
    # Flags
    display_preanalysis: bool = False,
    display_postanalysis: bool = False,
    refit: bool = True,
    predict: bool = True,
    diagnostics: bool = True,
    figures: bool = True,
    # Storage
    output_dir: str | Path = OUTPUTS,
    submission_path: str | Path | None = None,
    seed: int = 0,
) -> ExecutionResult:
    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = Path(submission_path) if submission_path else output_dir / SUBMISSION.name
    y = pd.Series(np.asarray(y, dtype=float), index=X.index, name="y")

    # Step1: Validation
    if list(X.columns) != list(X_test.columns):
        raise ValueError("train and test columns differ")
    if not np.array_equal(sample_submission[ID].astype(str).str.strip().to_numpy(),
                          X_test[ID].astype(str).str.strip().to_numpy()):
        raise ValueError("sample submission IDs do not match test.csv IDs (count or order)")
    if not candidates and not baseline_builder.methods:
        raise ValueError("no candidates and no baselines: nothing to evaluate")
    if y.isna().any():
        raise ValueError("target contains NaN")

    # Step2: Preparation - cleaned + engineered view of the training rows, run metadata
    cleaned = clone(cleaner).fit_transform(X)
    if len(cleaned) != len(X):
        raise ValueError("cleaner changed the number of rows")
    features = DeliveryFeatureEngineer().fit_transform(cleaned)
    import sklearn, torch  # noqa: E401  (versions for the run record)

    meta = {
        "exp_name": exp_name, "seed": seed, "cv": repr(evaluator._cv()),
        "n_train": int(len(X)), "n_test": int(len(X_test)),
        "target_mean": float(y.mean()), "target_std": float(y.std()),
        "target_min": float(y.min()), "target_max": float(y.max()),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"delivery": __version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
                     "pandas": pd.__version__, "python": platform.python_version()},
    }

    # Step3: Pre-analysis (training rows only, before any model is fitted)
    if display_preanalysis and preanalysis_reporter is not None:
        preanalysis_reporter.display(exp_name=exp_name, X=X, y=y, features=features)

    # Step4: Baselines + candidates
    all_candidates = {**baseline_builder.build(), **candidates}
    meta["candidates"] = list(all_candidates)
    _log(exp_name, f"Step4: {len(all_candidates)} candidates: {list(all_candidates)}")

    # Step5: Cross-validation
    _log(exp_name, f"Step5: cross-validation with {meta['cv']}")
    board, reports = evaluator.compare(all_candidates, X, y, verbose=True)
    best_name = str(board.index[0])
    _log(exp_name, f"Step5: winner = {best_name} (MAE {board.loc[best_name, 'mae']:.3f})")

    # Step6: Diagnostics - one holdout fit of the non-baseline candidates + PyTorch curves
    diag = None
    if diagnostics and candidates:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
        rows, mlp = [], None
        for name, pipe in candidates.items():
            _log(exp_name, f"Step6: holdout fit of {name}")
            rep = evaluator.holdout(pipe, X_tr, y_tr, X_val, y_val, name=name)
            rows.append(metrics_builder.build(rep.y_true, rep.y_pred, name=name).assign(fit_time_s=rep.fit_time))
            model = rep.estimator_[-1] if isinstance(rep.estimator_, Pipeline) else rep.estimator_
            if isinstance(model, TorchMLPRegressor) and mlp is None:
                mlp = {"name": name, "n_iter": model.n_iter_, "epochs": model.epochs, "loss": model.loss,
                       "patience": model.patience, "loss_curve": list(model.loss_curve_),
                       "val_curve": list(model.val_curve_),
                       "best_val_mae": float(getattr(model, "best_val_mae_", np.nan)),
                       "holdout_mae": float(rep.mae)}
        diag = {"split": f"80/20 (seed {seed})", "holdout": pd.concat(rows), "mlp": mlp}

    # Step7: Experiments
    experiments: dict[str, EvaluationReport] = {}
    for name, exp in (extra_experiments or {}).items():
        _log(exp_name, f"Step7: experiment {name!r}")
        ev = exp.evaluator or evaluator
        experiments[name] = ev.evaluate(exp.pipeline, X, y, groups=exp.groups, name=name)
        _log(exp_name, f"Step7: {name}: mae={experiments[name].mae:.3f}")

    # Step8: Refit + predict
    model = submission = None
    paths: dict[str, Path] = {"output_dir": output_dir}
    if refit:
        _log(exp_name, f"Step8: refit {best_name} on all {len(X)} rows")
        model = clone(all_candidates[best_name]).fit(X, y.to_numpy())
        joblib.dump(model, output_dir / "model.joblib")
        paths["model"] = output_dir / "model.joblib"
        (output_dir / "best_model.json").write_text(json.dumps({
            "exp_name": exp_name, "best": best_name, "cv_mae": float(board.loc[best_name, "mae"]),
            "params": {k: repr(v) for k, v in model[-1].get_params().items()}}, indent=2))
        paths["best_model"] = output_dir / "best_model.json"
        if predict:
            preds = np.asarray(model.predict(X_test), dtype=float)
            submission = write_submission(X_test[ID], preds, submission_path, sample=sample_submission)
            paths["submission"] = submission_path
            _log(exp_name, f"Step8: wrote submission ({len(submission)} rows, mean {preds.mean():.2f})")

    # Step9: Dump
    meta["wall_time_s"] = round(time.perf_counter() - t_start, 1)
    all_board = leaderboard({**reports, **experiments})
    paths["metrics"] = save_report({**reports, **experiments}, all_board, output_dir, best=best_name,
                                   figures=figures, extra={"exp_name": exp_name, "seed": seed, "cv": meta["cv"],
                                                          "n_rows": meta["n_train"]})
    if figures:
        paths["figures"] = output_dir / "figures"
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    paths["run_meta"] = output_dir / "run_meta.json"
    _log(exp_name, f"Step9: artefacts under {output_dir} ({meta['wall_time_s']}s)")

    result = ExecutionResult(exp_name=exp_name, board=board, reports=reports, best_name=best_name,
                             experiments=experiments, diagnostics=diag, model=model, submission=submission,
                             paths=paths, meta=meta)

    # Step10: Post-analysis
    if display_postanalysis:
        postanalysis_reporter.display(result, X=X, y=y)
    return result
