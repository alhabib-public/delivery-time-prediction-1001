"""Builders: construct auxiliary objects for the pipeline (``build(...)``)."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from ..evaluation import METRICS, compute_metrics
from ..models import CANDIDATE_NAMES, make_candidate


class MetricsBuilder:
    """Turn ``(y_true, y_pred)`` into a one-row metrics frame (``mae``, ``rmse``, ...).

    Parameters
    ----------
    metrics : dict or None
        ``name -> (fn, greater_is_better)`` as in ``delivery.evaluation.METRICS`` (default).
    """

    def __init__(self, metrics: dict | None = None):
        self.metrics = metrics

    def build(self, y_true, y_pred, name: str = "model") -> pd.DataFrame:
        row = compute_metrics(y_true, y_pred, self.metrics or METRICS)
        return pd.DataFrame([row], index=pd.Index([name], name="model"))


class BaselineBuilder:
    """Build the reference pipelines every experiment is compared against.

    Parameters
    ----------
    methods : tuple of str
        Candidate names from ``delivery.models.CANDIDATE_NAMES`` (default median + ridge).
    random_state : int
    """

    def __init__(self, methods=("median", "ridge"), random_state: int = 0):
        self.methods = tuple(methods)
        self.random_state = random_state
        unknown = set(self.methods) - set(CANDIDATE_NAMES)
        if unknown:
            raise ValueError(f"unknown baseline(s) {sorted(unknown)}; choose from {CANDIDATE_NAMES}")

    def build(self) -> dict[str, Pipeline]:
        return {m: make_candidate(m, random_state=self.random_state) for m in self.methods}
