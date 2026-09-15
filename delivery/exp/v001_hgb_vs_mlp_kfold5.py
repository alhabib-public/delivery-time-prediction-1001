"""Gradient boosting vs. a PyTorch MLP for delivery-time prediction, selected by
out-of-fold MAE under shuffled 5-fold cross-validation.

Arm
---
* Input: ``dataset/train.csv`` loaded as raw strings; every fix (``"NaN"`` literals,
  whitespace, string numerics, ``"(min) 24"`` target, impossible ratings/coordinates)
  lives in ``DeliveryCleaner`` so ``test.csv`` takes the identical path.
* Features (``DeliveryFeatureEngineer``): haversine distance, order/pickup minute of day,
  prep time with midnight wrap, calendar fields, missing flags, courier city code.
* Candidates: ``hgb`` = ``HistGradientBoostingRegressor(loss="absolute_error")`` with
  native categoricals on the NaN-preserving ``"tree"`` encoding; ``mlp`` =
  ``TorchMLPRegressor`` (256-128, dropout 0.1, AdamW + cosine, Huber loss, early stopping)
  on the imputed/scaled/one-hot ``"dense"`` encoding. Baselines: median, Ridge.
* Validation: ``KFold(5, shuffle=True, random_state=seed)``. The test set shares the
  training date range and every test courier appears in train, so an i.i.d. split is the
  right proxy; a ``GroupKFold`` by courier runs as a robustness experiment, and a log1p
  target transform as a target-scale experiment. Selection = lowest out-of-fold MAE.
* Pre-analysis (training rows only): missing-value literals and their co-occurrence,
  target distribution, engineered-feature summary, Spearman correlations with the target
  and between features, mean target per categorical level.
* Post-analysis: run summary, leaderboard + per-fold scores, holdout diagnostics with the
  MLP's training curves, experiments, error analysis by traffic / distance / city / time of
  day / weather / multiple deliveries, submission summary.

v001: initial arm (numbers in REPORT.ipynb: hgb 3.153, mlp 3.410, ridge 4.857, median 7.564).

How to add v002
---------------
Copy this file to ``v002_<slug>.py``, change exactly one thing (a feature, a candidate, the
selection rule, ...) and prepend a paragraph "v002 on top of v001: <change>, <why>;
everything else unchanged". Run it with ``make train EXP=v002_<slug>`` or from a notebook:
``from delivery.exp import v002_<slug> as exp; exp.run(display_preanalysis=True, display_postanalysis=True)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold

from ..analysis import BaselineBuilder, MetricsBuilder, PostAnalysisReporter, PreAnalysisReporter
from ..data import load_sample_submission, load_test, load_train
from ..evaluation import DeliveryEvaluator
from ..models import CANDIDATE_NAMES, make_candidate
from ..pipeline import Experiment, ExecutionResult, execute
from ..pipeline.regression import OUTPUTS, SUBMISSION
from ..preprocessing import DeliveryCleaner

# experiment name = file name (never hand-written); it labels the run record and the report
EXP_NAME = Path(__file__).stem
BASELINES = ("median", "ridge")
CANDIDATES = ("hgb", "mlp")


def make_cleaner() -> DeliveryCleaner:
    return DeliveryCleaner(max_rating=5.0, min_coord_abs=1.0)


def make_candidates(seed: int, names=CANDIDATES) -> dict:
    return {name: make_candidate(name, random_state=seed) for name in names}


def make_baseline_builder(seed: int, names=BASELINES) -> BaselineBuilder:
    return BaselineBuilder(methods=names, random_state=seed)


def make_evaluator(cv: int, seed: int) -> DeliveryEvaluator:
    return DeliveryEvaluator(cv=cv, random_state=seed)


def make_experiments(seed: int, cv: int, groups: np.ndarray, subject: str = "hgb") -> dict[str, Experiment]:
    """Variants of the expected winner: log1p target and courier-grouped CV."""
    log_pipe = make_candidate(subject, random_state=seed)
    log_pipe.set_params(model=TransformedTargetRegressor(regressor=log_pipe.named_steps["model"],
                                                         func=np.log1p, inverse_func=np.expm1))
    grouped = DeliveryEvaluator(cv=GroupKFold(n_splits=cv, shuffle=True, random_state=seed), random_state=seed)
    return {
        f"{subject} (log1p target)": Experiment(pipeline=log_pipe),
        f"{subject} (GroupKFold by courier)": Experiment(pipeline=make_candidate(subject, random_state=seed),
                                                         evaluator=grouped, groups=groups),
    }


def make_partials() -> tuple[PreAnalysisReporter, PostAnalysisReporter]:
    """Pre- and post-analysis reporters with their defaults (sections can be swapped or disabled here)."""
    return PreAnalysisReporter(), PostAnalysisReporter()


def _split_models(models) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if models is None:
        return BASELINES, CANDIDATES
    names = [m.strip() for m in (models.split(",") if isinstance(models, str) else models) if m.strip()]
    unknown = set(names) - set(CANDIDATE_NAMES)
    if unknown:
        raise ValueError(f"unknown model(s) {sorted(unknown)}; choose from {CANDIDATE_NAMES}")
    return tuple(n for n in names if n in BASELINES), tuple(n for n in names if n not in BASELINES)


def run(
    display_preanalysis: bool = False,
    display_postanalysis: bool = False,
    cv: int = 5,
    seed: int = 0,
    models=None,
    n_rows: int | None = None,
    output_dir=None,
    submission_path=None,
    refit: bool = True,
    predict: bool = True,
    diagnostics: bool = True,
    experiments: bool = True,
    figures: bool = True,
    *args,
    **kwargs,
) -> ExecutionResult:
    """Load the data, build the parts, execute once.

    ``models`` limits the candidate set (``"median,ridge,hgb"`` or a list); ``n_rows``
    subsamples the training rows for smoke runs; ``output_dir`` defaults to ``outputs/`` and
    ``submission_path`` to ``COMPLETED_PREDICTIONS.csv`` at the repo root (inside ``output_dir``
    when a custom ``output_dir`` is given).
    """
    # Prepare
    X, y = load_train()
    if n_rows:
        idx = X.sample(n=min(n_rows, len(X)), random_state=seed).index
        X, y = X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)
    X_test = load_test()
    sample = load_sample_submission()
    baselines, candidates = _split_models(models)
    groups = X["Delivery_person_ID"].astype(str).str.strip().to_numpy()
    subject = "hgb" if "hgb" in candidates else (candidates or baselines)[0]
    preanalysis_reporter, postanalysis_reporter = make_partials()

    # Execute
    return execute(
        exp_name=EXP_NAME,
        # Input: for X, y
        X=X, y=y, X_test=X_test, sample_submission=sample,
        # Reporters: for analysis
        preanalysis_reporter=preanalysis_reporter,
        postanalysis_reporter=postanalysis_reporter,
        # Builders: for metrics and baselines
        metrics_builder=MetricsBuilder(),
        baseline_builder=make_baseline_builder(seed, baselines),
        # Cleaner
        cleaner=make_cleaner(),
        # Model(s)
        candidates=make_candidates(seed, candidates),
        # Validation
        evaluator=make_evaluator(cv, seed),
        # Experiments
        extra_experiments=make_experiments(seed, cv, groups, subject) if experiments else None,
        # Flags
        display_preanalysis=display_preanalysis,
        display_postanalysis=display_postanalysis,
        refit=refit, predict=predict, diagnostics=diagnostics, figures=figures,
        # Storage
        output_dir=output_dir if output_dir is not None else OUTPUTS,
        submission_path=submission_path if submission_path is not None else (SUBMISSION if output_dir is None else None),
        seed=seed,
    )


if __name__ == "__main__":
    run()
