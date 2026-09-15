"""Bespoke, sklearn-style evaluation pipeline for delivery-time regressors.

Built from the same primitives scikit-learn uses for its own tooling:

* metrics -> ``make_scorer`` scorers (``SCORERS``) usable in ``cross_validate``/``GridSearchCV``;
* a ``DeliverySlicer`` transformer that maps raw rows to error-analysis slices
  (traffic, distance bucket, city, weather, time of day, ...);
* ``DeliveryEvaluator`` (a ``BaseEstimator`` so its settings are ``get_params``-able) that
  runs cross-validation, collects out-of-fold predictions and produces an
  ``EvaluationReport`` with overall, per-fold and per-slice metrics;
* ``*Display`` classes following the ``from_estimator`` / ``from_predictions`` /
  ``plot(ax=)`` convention of ``sklearn.metrics.PredictionErrorDisplay``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    PredictionErrorDisplay,
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, check_cv, cross_validate
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .preprocessing import DeliveryCleaner, DeliveryFeatureEngineer, _FrameTransformer

# ---------------------------------------------------------------------------
# metrics & scorers
# ---------------------------------------------------------------------------
def within_tolerance(y_true, y_pred, *, tol: float = 5.0) -> float:
    """Fraction of predictions within ``tol`` minutes of the truth (business-readable)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def mean_bias(y_true, y_pred) -> float:
    """Mean signed error (pred - true); >0 means over-prediction."""
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


# name -> (function, greater_is_better)
METRICS: dict[str, tuple] = {
    "mae": (mean_absolute_error, False),
    "rmse": (root_mean_squared_error, False),
    "medae": (median_absolute_error, False),
    "mape": (mean_absolute_percentage_error, False),
    "r2": (r2_score, True),
    "within_5min": (within_tolerance, True),
    "bias": (mean_bias, None),  # signed, not a scorer
}
PRIMARY_METRIC = "mae"

SCORERS = {
    name: make_scorer(fn, greater_is_better=gib)
    for name, (fn, gib) in METRICS.items()
    if gib is not None
}


def compute_metrics(y_true, y_pred, metrics: dict | None = None) -> dict[str, float]:
    metrics = metrics or METRICS
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {name: float(fn(y_true, y_pred)) for name, (fn, _) in metrics.items()}


# ---------------------------------------------------------------------------
# slices
# ---------------------------------------------------------------------------
DISTANCE_BINS = [0, 2, 5, 10, 15, np.inf]
DISTANCE_LABELS = ["0-2 km", "2-5 km", "5-10 km", "10-15 km", "15+ km"]
HOUR_BINS = [-1, 5, 10, 14, 17, 21, 24]
HOUR_LABELS = ["night (0-5)", "morning (6-10)", "lunch (11-14)", "afternoon (15-17)",
               "evening (18-21)", "late (22-23)"]
DEFAULT_SLICES = ("traffic", "distance", "city", "weather", "time_of_day",
                  "multiple_deliveries", "festival", "vehicle")
# natural display order for ordered slices (others are sorted by row count)
SLICE_ORDER = {
    "traffic": ["Low", "Medium", "High", "Jam", "unknown"],
    "distance": DISTANCE_LABELS + ["unknown"],
    "time_of_day": HOUR_LABELS + ["unknown"],
    "multiple_deliveries": ["0 extra", "1 extra", "2 extra", "3 extra", "unknown"],
}


class DeliverySlicer(_FrameTransformer):
    """Raw rows -> one categorical label column per slice (``DEFAULT_SLICES``).

    Reuses the cleaner/feature engineer so slice definitions (e.g. distance) are
    the same quantities the models see. Missing values map to ``"unknown"``.
    """

    def __init__(self, slices=DEFAULT_SLICES):
        self.slices = slices

    def _output_columns(self, X):
        return list(self.slices)

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = self._check_frame(X)
        f = DeliveryFeatureEngineer(drop_ids=True).fit_transform(DeliveryCleaner().fit_transform(X))
        out = pd.DataFrame(index=X.index)
        sources = {
            "traffic": lambda: f["Road_traffic_density"],
            "distance": lambda: pd.cut(f["distance_km"], DISTANCE_BINS, labels=DISTANCE_LABELS,
                                       right=False).astype(object),
            "city": lambda: f["City"],
            "weather": lambda: f["Weatherconditions"],
            "time_of_day": lambda: pd.cut(f["order_hour"], HOUR_BINS, labels=HOUR_LABELS).astype(object),
            "multiple_deliveries": lambda: f["multiple_deliveries"].map(
                lambda v: f"{int(v)} extra" if pd.notna(v) else np.nan),
            "festival": lambda: f["Festival"],
            "vehicle": lambda: f["Type_of_vehicle"],
            "driver_city": lambda: f["driver_city"],
            "order_type": lambda: f["Type_of_order"],
        }
        for name in self.slices:
            if name not in sources:
                raise ValueError(f"unknown slice {name!r}; choose from {sorted(sources)}")
            s = sources[name]()
            out[name] = s.astype(object).where(pd.notna(s), "unknown").astype(str)
        return out


def slice_table(y_true, y_pred, labels, *, order: list | None = None) -> pd.DataFrame:
    """Per-group ``n, share, mae, rmse, bias, within_5min`` sorted by ``order`` or by n."""
    df = pd.DataFrame({"y": np.asarray(y_true, float).ravel(),
                       "p": np.asarray(y_pred, float).ravel(),
                       "g": np.asarray(labels).astype(str).ravel()})
    rows = []
    for g, part in df.groupby("g", sort=False):
        rows.append({
            "slice": g,
            "n": len(part),
            "share": len(part) / len(df),
            "mae": mean_absolute_error(part.y, part.p),
            "rmse": root_mean_squared_error(part.y, part.p),
            "bias": mean_bias(part.y, part.p),
            "within_5min": within_tolerance(part.y, part.p),
            "mean_true": float(part.y.mean()),
        })
    tab = pd.DataFrame(rows).set_index("slice")
    if order:
        present = [o for o in order if o in tab.index] + [i for i in tab.index if i not in order]
        tab = tab.loc[present]
    else:
        tab = tab.sort_values("n", ascending=False)
    return tab


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
@dataclass
class EvaluationReport:
    """Everything produced by one evaluation run of one estimator."""

    name: str
    overall: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    slices: pd.DataFrame                      # slice labels aligned with y_true
    folds: pd.DataFrame | None = None         # per-fold scorer results (CV only)
    fit_time: float = float("nan")
    per_slice: dict[str, pd.DataFrame] = field(default_factory=dict)
    method: str = "cv"
    wall_time: float = float("nan")

    @property
    def mae(self) -> float:
        return self.overall[PRIMARY_METRIC]

    def slice_table(self, name: str) -> pd.DataFrame:
        if name not in self.per_slice:
            self.per_slice[name] = slice_table(self.y_true, self.y_pred, self.slices[name],
                                               order=SLICE_ORDER.get(name))
        return self.per_slice[name]

    def summary_frame(self) -> pd.DataFrame:
        row = dict(self.overall)
        if self.folds is not None and "test_mae" in self.folds:
            row["mae_fold_std"] = float(self.folds["test_mae"].std(ddof=0))
        row["fit_time_s"] = self.fit_time
        return pd.DataFrame([row], index=[self.name])

    def to_markdown(self, slices: tuple[str, ...] | None = None, digits: int = 3) -> str:
        parts = [f"### {self.name} ({self.method})", "", md_table(self.summary_frame(), digits), ""]
        for s in slices or tuple(self.slices.columns):
            parts += [f"**by {s}**", "", md_table(self.slice_table(s), digits), ""]
        return "\n".join(parts)

    def to_dict(self) -> dict:
        d = {"name": self.name, "method": self.method, "overall": self.overall,
             "fit_time_s": self.fit_time,
             "per_slice": {s: self.slice_table(s).reset_index().to_dict(orient="records")
                           for s in self.slices.columns}}
        if self.folds is not None:
            d["folds"] = self.folds.to_dict(orient="records")
        return d


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------
class DeliveryEvaluator(BaseEstimator):
    """Evaluate delivery-time regressors with CV + slice-level error analysis.

    Parameters
    ----------
    cv : int or CV splitter
        Integer -> ``KFold(cv, shuffle=True, random_state=random_state)``. Pass a
        ``GroupKFold`` and ``groups`` to ``evaluate`` for grouped validation.
    scorers : dict
        sklearn scorers used per fold (defaults to ``SCORERS``).
    slices : tuple of str
        Slice names understood by ``DeliverySlicer``.
    random_state : int
    n_jobs : int or None
        Parallel folds (``None`` = sequential, safest with torch).
    """

    def __init__(self, cv=5, scorers=None, slices=DEFAULT_SLICES, random_state=0, n_jobs=None):
        self.cv = cv
        self.scorers = scorers
        self.slices = slices
        self.random_state = random_state
        self.n_jobs = n_jobs

    # helpers -------------------------------------------------------------
    def _cv(self, groups=None):
        if isinstance(self.cv, (int, np.integer)):
            return KFold(n_splits=int(self.cv), shuffle=True, random_state=self.random_state)
        return check_cv(self.cv, y=None, classifier=False)

    def _scorers(self) -> dict:
        return dict(self.scorers) if self.scorers is not None else dict(SCORERS)

    def _slice_frame(self, X) -> pd.DataFrame:
        return DeliverySlicer(slices=self.slices).fit_transform(X).reset_index(drop=True)

    # public API ----------------------------------------------------------
    def evaluate(self, estimator, X, y, *, groups=None, name: str | None = None) -> EvaluationReport:
        """Cross-validate ``estimator`` and return a report built on out-of-fold predictions."""
        name = name or type(estimator).__name__
        y_arr = np.asarray(y, dtype=float).ravel()
        cv = self._cv(groups)
        t0 = time.perf_counter()
        res = cross_validate(
            clone(estimator), X, y_arr, groups=groups, cv=cv, scoring=self._scorers(),
            n_jobs=self.n_jobs, return_estimator=True, return_indices=True, error_score="raise",
        )
        # out-of-fold predictions from the fold estimators
        y_pred = np.full(len(y_arr), np.nan)
        for est, test_idx in zip(res["estimator"], res["indices"]["test"]):
            y_pred[test_idx] = est.predict(X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx])
        if np.isnan(y_pred).any():          # e.g. a splitter that does not cover every row
            mask = ~np.isnan(y_pred)
            y_arr, y_pred, X_used = y_arr[mask], y_pred[mask], (X.iloc[mask] if hasattr(X, "iloc") else X[mask])
        else:
            X_used = X
        folds = pd.DataFrame({k: v for k, v in res.items() if k.startswith("test_") or k.endswith("_time")})
        folds.index.name = "fold"
        # sklearn scorers negate "lower is better" metrics; flip them back for readability
        for k in list(folds.columns):
            if k.startswith("test_") and k[5:] in METRICS and METRICS[k[5:]][1] is False:
                folds[k] = -folds[k]
        return EvaluationReport(
            name=name, overall=compute_metrics(y_arr, y_pred), y_true=y_arr, y_pred=y_pred,
            slices=self._slice_frame(X_used), folds=folds, fit_time=float(np.sum(res["fit_time"])),
            method=f"{cv.__class__.__name__}({cv.get_n_splits(groups=groups)})",
            wall_time=time.perf_counter() - t0,
        )

    def holdout(self, estimator, X_train, y_train, X_val, y_val, *, name: str | None = None) -> EvaluationReport:
        """Fit on the train split, report on the validation split (estimator is cloned)."""
        name = name or type(estimator).__name__
        est = clone(estimator)
        t0 = time.perf_counter()
        est.fit(X_train, np.asarray(y_train, dtype=float).ravel())
        fit_time = time.perf_counter() - t0
        y_val = np.asarray(y_val, dtype=float).ravel()
        y_pred = np.asarray(est.predict(X_val), dtype=float).ravel()
        rep = EvaluationReport(
            name=name, overall=compute_metrics(y_val, y_pred), y_true=y_val, y_pred=y_pred,
            slices=self._slice_frame(X_val), fit_time=fit_time, method="holdout",
        )
        rep.estimator_ = est
        return rep

    def compare(self, candidates: dict, X, y, *, groups=None, verbose: bool = True):
        """Evaluate several candidates; return ``(leaderboard, reports)`` sorted by MAE."""
        reports: dict[str, EvaluationReport] = {}
        for name, est in candidates.items():
            if verbose:
                print(f"[evaluate] {name} ...", flush=True)
            reports[name] = self.evaluate(est, X, y, groups=groups, name=name)
            if verbose:
                r = reports[name]
                print(f"[evaluate] {name}: mae={r.mae:.3f}  rmse={r.overall['rmse']:.3f}  "
                      f"within_5min={r.overall['within_5min']:.3f}  fit={r.fit_time:.1f}s", flush=True)
        board = leaderboard(reports)
        return board, reports


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    """Minimal GitHub-markdown table (avoids the optional ``tabulate`` dependency)."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: f"{v:.{digits}f}")
    header = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)


# bar colouring by column semantics: lower-is-better (warm), higher-is-better (cool),
# signed quantities (two-colour, centred on zero), anything else numeric (neutral)
_LOWER_BETTER = ("mae", "rmse", "medae", "mape", "mae_fold_std", "fit_time_s", "std_target", "missing_share")
_HIGHER_BETTER = ("r2", "within_5min")
_SIGNED_HINTS = ("bias", "lift", "spearman", "pearson", "kendall", "corr")
_PERCENT = ("within_5min", "share", "missing_share")
_COLORS = {"low": "#f2c4a0", "high": "#a2c8e8", "signed": ["#e8a2a2", "#a2c8e8"], "neutral": "#cfd8e3"}


def _bar_kind(name: str, values: pd.Series) -> str:
    key = str(name).lower()
    if key in _LOWER_BETTER:
        return "low"
    if key in _HIGHER_BETTER:
        return "high"
    v = values.dropna()
    if any(h in key for h in _SIGNED_HINTS) or (len(v) and v.min() < 0 < v.max()):
        return "signed"
    return "neutral"


def style_table(df: pd.DataFrame, *, bars=None, digits: int = 3, caption: str | None = None,
                highlight_best: bool = True):
    """Notebook-friendly ``pandas.Styler``.

    Every numeric column gets an in-cell bar (``bars=None``; pass a tuple of names to restrict,
    ``False`` for none). Bars are per column, so each column uses its own scale: warm for
    lower-is-better metrics, cool for higher-is-better, two-colour centred on zero for signed
    quantities (bias, lift, correlations), neutral for plain quantities (counts, means).
    Single-row tables and constant columns get no bars. Percent-like columns are formatted
    as percentages; the best row of the primary metric is highlighted.
    """
    df = df.copy()
    if not df.index.is_unique:          # Styler.bar/highlight need unique labels
        df = df.reset_index()
    if not df.columns.is_unique:
        df.columns = [f"{c}_{i}" if list(df.columns).count(c) > 1 else c for i, c in enumerate(df.columns)]
    sty = df.style
    fmt = {c: (lambda v: f"{v:.1%}") for c in df if str(c).lower() in _PERCENT}
    fmt.update({c: (lambda v: f"{v:,.0f}") for c in df if str(c).lower() == "n"})
    sty = sty.format(fmt, precision=digits, na_rep="")
    if bars is False or len(df) < 2:
        bar_cols = ()
    elif bars is None:
        bar_cols = tuple(c for c in df if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1)
    else:
        bar_cols = tuple(c for c in bars if c in df and pd.api.types.is_numeric_dtype(df[c]))
    for c in bar_cols:
        kind = _bar_kind(c, df[c])
        if kind == "signed":
            lim = float(np.nanmax(np.abs(df[c].to_numpy(dtype=float)))) or 1.0
            sty = sty.bar(subset=[c], align="zero", vmin=-lim, vmax=lim, color=_COLORS["signed"])
        else:
            lo = float(np.nanmin(df[c].to_numpy(dtype=float)))
            sty = sty.bar(subset=[c], vmin=min(0.0, lo), color=_COLORS[kind])
    if highlight_best and PRIMARY_METRIC in df and len(df) > 1:
        sty = sty.highlight_min(subset=[PRIMARY_METRIC], props="font-weight:bold; background-color:#d9ead3")
    sty = sty.set_table_styles([{"selector": "th", "props": "text-align:left; white-space:nowrap"},
                                {"selector": "td", "props": "white-space:nowrap"}])
    if caption:
        sty = sty.set_caption(caption)
    return sty


def style_leaderboard(board: pd.DataFrame, caption: str = "out-of-fold leaderboard (lower MAE is better)"):
    cols = [c for c in ("mae", "rmse", "medae", "mape", "r2", "within_5min", "bias", "mae_fold_std", "fit_time_s")
            if c in board]
    return style_table(board[cols], caption=caption)


def style_slices(report: EvaluationReport, slice_name: str):
    return style_table(report.slice_table(slice_name), caption=f"{report.name} by {slice_name}",
                       highlight_best=False)


def leaderboard(reports: dict[str, EvaluationReport]) -> pd.DataFrame:
    board = pd.concat([r.summary_frame() for r in reports.values()])
    board.index.name = "model"
    return board.sort_values(PRIMARY_METRIC)


# ---------------------------------------------------------------------------
# displays (sklearn *Display convention)
# ---------------------------------------------------------------------------
class SliceErrorDisplay:
    """Bar chart of MAE per slice value, annotated with row counts.

    Use ``from_predictions`` / ``from_estimator`` / ``from_report`` rather than the
    constructor. After ``plot``: ``ax_``, ``figure_``, ``bars_``, ``table_``.
    """

    def __init__(self, *, table: pd.DataFrame, slice_name: str, overall_mae: float | None = None):
        self.table = table
        self.slice_name = slice_name
        self.overall_mae = overall_mae

    @classmethod
    def from_predictions(cls, y_true, y_pred, slice_labels, *, slice_name="slice", ax=None, **kwargs):
        tab = slice_table(y_true, y_pred, slice_labels, order=SLICE_ORDER.get(slice_name))
        disp = cls(table=tab, slice_name=slice_name, overall_mae=mean_absolute_error(y_true, y_pred))
        return disp.plot(ax=ax, **kwargs)

    @classmethod
    def from_estimator(cls, estimator, X, y, *, slice_name="traffic", ax=None, **kwargs):
        labels = DeliverySlicer(slices=(slice_name,)).fit_transform(X)[slice_name]
        return cls.from_predictions(y, estimator.predict(X), labels, slice_name=slice_name, ax=ax, **kwargs)

    @classmethod
    def from_report(cls, report: EvaluationReport, slice_name: str, *, ax=None, **kwargs):
        disp = cls(table=report.slice_table(slice_name), slice_name=slice_name, overall_mae=report.mae)
        return disp.plot(ax=ax, **kwargs)

    def plot(self, ax=None, *, metric="mae", annotate=True, bar_kwargs=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3.6))
        tab = self.table
        bar_kwargs = {**{"color": "tab:blue", "alpha": 0.85}, **(bar_kwargs or {})}
        self.bars_ = ax.bar(range(len(tab)), tab[metric].to_numpy(), **bar_kwargs)
        ax.set_xticks(range(len(tab)))
        ax.set_xticklabels(tab.index, rotation=30, ha="right")
        ax.set_ylabel(metric.upper() + (" (min)" if metric in ("mae", "rmse") else ""))
        ax.set_title(f"{metric.upper()} by {self.slice_name}")
        if self.overall_mae is not None and metric == "mae":
            ax.axhline(self.overall_mae, color="tab:red", ls="--", lw=1, label=f"overall {self.overall_mae:.2f}")
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        if annotate:
            for rect, n in zip(self.bars_, tab["n"]):
                ax.annotate(f"n={n:,}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")
        ax.margins(y=0.2)
        self.ax_ = ax
        self.figure_ = ax.figure
        self.table_ = tab
        return self


class ResidualDisplay:
    """Residual (pred - true) histogram or residuals against the truth/prediction."""

    def __init__(self, *, y_true, y_pred):
        self.y_true = np.asarray(y_true, dtype=float).ravel()
        self.y_pred = np.asarray(y_pred, dtype=float).ravel()

    @classmethod
    def from_predictions(cls, y_true, y_pred, *, ax=None, **kwargs):
        return cls(y_true=y_true, y_pred=y_pred).plot(ax=ax, **kwargs)

    @classmethod
    def from_estimator(cls, estimator, X, y, *, ax=None, **kwargs):
        return cls.from_predictions(y, estimator.predict(X), ax=ax, **kwargs)

    def plot(self, ax=None, *, kind="hist", bins=40, subsample=2000, random_state=0):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3.6))
        res = self.y_pred - self.y_true
        if kind == "hist":
            ax.hist(res, bins=bins, color="tab:blue", alpha=0.85)
            ax.axvline(0, color="k", lw=1)
            ax.set(xlabel="residual = predicted - actual (min)", ylabel="count",
                   title=f"residuals: mean {res.mean():+.2f}, MAE {np.abs(res).mean():.2f}")
        elif kind in ("vs_actual", "vs_predicted"):
            rng = check_random_state(random_state)
            idx = rng.choice(len(res), size=min(subsample, len(res)), replace=False)
            x = self.y_true[idx] if kind == "vs_actual" else self.y_pred[idx]
            ax.scatter(x, res[idx], s=6, alpha=0.4, color="tab:blue")
            ax.axhline(0, color="k", lw=1)
            ax.set(xlabel=("actual" if kind == "vs_actual" else "predicted") + " (min)",
                   ylabel="residual (min)", title=f"residuals {kind.replace('_', ' ')}")
        else:
            raise ValueError("kind must be 'hist', 'vs_actual' or 'vs_predicted'")
        self.ax_ = ax
        self.figure_ = ax.figure
        return self


def plot_report(report: EvaluationReport, slices: tuple[str, ...] | None = None, *, ncols: int = 2):
    """Standard figure set for a report: actual-vs-predicted, residual hist, one panel per slice."""
    import matplotlib.pyplot as plt

    slices = tuple(slices or report.slices.columns)
    n = 2 + len(slices)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    PredictionErrorDisplay.from_predictions(report.y_true, report.y_pred, kind="actual_vs_predicted",
                                            subsample=2000, random_state=0, ax=axes[0])
    axes[0].set_title(f"{report.name}: actual vs predicted (MAE {report.mae:.2f})")
    ResidualDisplay(y_true=report.y_true, y_pred=report.y_pred).plot(ax=axes[1])
    for ax, s in zip(axes[2:], slices):
        SliceErrorDisplay.from_report(report, s, ax=ax)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def safe_name(name: str) -> str:
    """``"hgb (log target)"`` -> ``"hgb_log_target"`` for file names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def save_report(reports: dict[str, EvaluationReport], board: pd.DataFrame, out_dir, *,
                best: str | None = None, extra: dict | None = None, figures: bool = True) -> Path:
    """Write ``metrics.json`` (+ ``figures/<name>.png``) under ``out_dir``; return the json path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary_metric": PRIMARY_METRIC,
        "best": best or board.index[0],
        "leaderboard": board.reset_index().to_dict(orient="records"),
        "reports": {k: r.to_dict() for k, r in reports.items()},
    }
    if extra:
        payload.update(extra)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    if figures:
        import matplotlib.pyplot as plt

        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        for k, r in reports.items():
            fig = plot_report(r)
            fig.savefig(fig_dir / f"{safe_name(k)}.png", dpi=110)
            plt.close(fig)
    return path
