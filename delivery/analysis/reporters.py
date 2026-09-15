"""Post-analysis reporters.

Each reporter is configuration-only in its constructor and exposes one
``display(result, **context)`` method that renders a section of the run report
through :mod:`delivery.analysis.display`. ``PostAnalysisReporter`` composes them in
order under ``##``-level headers, the way ml-pipe's compose reporters do.

``result`` is a :class:`delivery.pipeline.regression.ExecutionResult` (duck-typed here
to avoid a circular import).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..evaluation import leaderboard, plot_report, style_leaderboard, style_slices, style_table
from ..preprocessing import ALL_FEATURES, NOMINAL_FEATURES, NUMERIC_FEATURES, ORDINAL_FEATURES
from .display import (display_dataframe, display_figure, display_info_box, display_markdown,
                      display_title)


class Reporter(Protocol):
    """Interface for components that display one part of a run's results."""

    def display(self, result: Any, **context: Any) -> None: ...


# ---------------------------------------------------------------------------
class DataReporter:
    """One-line run summary (shapes, target statistics, seed, CV). Details live in pre-analysis."""

    def display(self, result, **_):
        m = result.meta
        display_info_box(
            f"train rows <b>{m['n_train']:,}</b> · test rows <b>{m['n_test']:,}</b> · "
            f"target mean <b>{m['target_mean']:.2f}</b> min, std {m['target_std']:.2f}, "
            f"range [{m['target_min']:.0f}, {m['target_max']:.0f}] · seed {m['seed']} · cv {m['cv']} · "
            f"candidates {', '.join(m['candidates'])}")


class LeaderboardReporter:
    """Styled out-of-fold leaderboard, per-fold MAE of the winner, selected-model banner."""

    def __init__(self, show_folds: bool = True):
        self.show_folds = show_folds

    def display(self, result, **_):
        display_dataframe(style_leaderboard(result.board, caption=f"{result.meta['cv']} — out-of-fold"))
        best = result.reports[result.best_name]
        display_info_box(
            f"Selected model: <b>{result.best_name}</b> — CV MAE <b>{best.mae:.3f} min</b>, "
            f"{best.overall['within_5min']:.1%} of deliveries predicted within ±5 min "
            f"(RMSE {best.overall['rmse']:.3f}, bias {best.overall['bias']:+.2f}).", kind="success")
        if self.show_folds and best.folds is not None:
            folds = best.folds[[c for c in best.folds.columns if c.startswith("test_")]].copy()
            folds.columns = [c[5:] for c in folds.columns]
            display_dataframe(folds, f"{result.best_name}: per-fold scores", highlight_best=False)


class TrainingReporter:
    """Holdout fit of every non-baseline candidate + the PyTorch model's training curves."""

    def __init__(self, figsize=(11, 3.4)):
        self.figsize = figsize

    def display(self, result, **_):
        diag = result.diagnostics
        if not diag:
            display_info_box("diagnostics disabled for this run.", kind="warning")
            return
        if diag.get("holdout") is not None:
            display_dataframe(style_leaderboard(diag["holdout"], caption=f"{diag['split']} holdout"))
        mlp = diag.get("mlp")
        if mlp is None:
            display_info_box("no TorchMLPRegressor among the candidates — no training curves.", kind="warning")
            return
        display_info_box(
            f"<b>{mlp['name']}</b>: {mlp['n_iter']} / {mlp['epochs']} epochs "
            f"(early stopping, patience {mlp['patience']}), loss <code>{mlp['loss']}</code>, "
            f"best internal-val MAE {mlp['best_val_mae']:.3f}, holdout MAE {mlp['holdout_mae']:.3f}.")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        axes[0].plot(mlp["loss_curve"], color="tab:blue")
        axes[0].set(xlabel="epoch", ylabel=f"{mlp['loss']} loss (standardised target)", title="training loss")
        axes[1].plot(mlp["val_curve"], color="tab:orange")
        axes[1].axvline(int(np.argmin(mlp["val_curve"])), color="k", ls=":", lw=1, label="best epoch")
        axes[1].set(xlabel="epoch", ylabel="MAE (min)", title="internal validation MAE")
        axes[1].legend()
        fig.tight_layout()
        display_figure(fig)


class ExperimentsReporter:
    """Extra experiments run alongside the winner (target transform, grouped CV) and,
    if ``outputs/experiments.json`` exists, the offline experiment ladder."""

    def __init__(self, ladder_file: str = "experiments.json"):
        self.ladder_file = ladder_file

    def display(self, result, **_):
        if result.experiments:
            board = leaderboard({result.best_name: result.reports[result.best_name], **result.experiments})
            display_dataframe(style_table(board[["mae", "rmse", "within_5min", "bias", "mae_fold_std"]],
                                          caption="winner vs. its variants (same folds)"))
        else:
            display_info_box("no extra experiments configured for this run.", kind="warning")
        ladder = Path(result.paths["output_dir"]) / self.ladder_file
        if ladder.exists():
            data = json.loads(ladder.read_text())
            tab = pd.DataFrame(data["results"]).set_index("model")
            tab.index.name = "variant"
            display_dataframe(style_table(tab, caption=f"experiment ladder ({data.get('cv')}-fold, "
                                                       f"`python -m delivery.experiments`)"))


class ErrorAnalysisReporter:
    """Out-of-fold error analysis of the winner: figure panel + per-slice tables."""

    def __init__(self, figure_slices=("traffic", "distance", "city", "time_of_day", "weather", "multiple_deliveries"),
                 table_slices=("traffic", "distance", "city", "time_of_day")):
        self.figure_slices = tuple(figure_slices)
        self.table_slices = tuple(table_slices)

    def display(self, result, **_):
        best = result.reports[result.best_name]
        display_markdown("`bias` = mean(predicted − actual): negative means under-prediction. "
                         "All numbers are out-of-fold.")
        display_figure(plot_report(best, slices=self.figure_slices))
        for s in self.table_slices:
            display_dataframe(style_slices(best, s))


class SubmissionReporter:
    """Prediction statistics, head of the submission file and the artefacts written."""

    def __init__(self, n_head: int = 5):
        self.n_head = n_head

    def display(self, result, **_):
        sub = result.submission
        if sub is None:
            display_info_box("refit / prediction skipped for this run (evaluate-only).", kind="warning")
        else:
            col = sub.columns[-1]
            preds = sub[col].to_numpy(dtype=float)
            display_info_box(
                f"<b>{result.best_name}</b> refit on all {result.meta['n_train']:,} rows → "
                f"{len(sub):,} test predictions: mean {preds.mean():.2f} min "
                f"(train mean {result.meta['target_mean']:.2f}), range [{preds.min():.1f}, {preds.max():.1f}].",
                kind="success")
            display_dataframe(sub.head(self.n_head), "submission head", bars=False)
        lines = [f"* `{k}`: `{v}`" for k, v in result.paths.items()]
        display_markdown("**artefacts**\n\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# pre-analysis: what the data looks like before any model is fitted
# ---------------------------------------------------------------------------
def _nan_literal_mask(X: pd.DataFrame) -> pd.DataFrame:
    return X.apply(lambda s: s.astype(str).str.strip().isin(["NaN", "conditions NaN", ""]))


class MissingnessReporter:
    """Share of ``"NaN"`` literals per raw column, how missingness co-occurs, and its effect on the target."""

    def __init__(self, min_share: float = 0.0):
        self.min_share = min_share

    def display(self, X=None, y=None, **_):
        mask = _nan_literal_mask(X)
        share = mask.mean()
        cols = share[share > self.min_share].sort_values(ascending=False)
        if cols.empty:
            display_info_box("no missing-value literals found.", kind="success")
            return
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        tab = pd.DataFrame({"share": cols, "n": (cols * len(X)).astype(int)})
        tab["mean_target_when_missing"] = [y[mask[c]].mean() for c in cols.index]
        tab["mean_target_when_present"] = [y[~mask[c]].mean() for c in cols.index]
        tab.index.name = "column"
        display_dataframe(tab, "missing-value literals per raw column", highlight_best=False)
        any_missing = mask[cols.index].any(axis=1)
        n_cols = mask[cols.index].sum(axis=1)
        display_info_box(
            f"{any_missing.mean():.1%} of rows have at least one missing field; "
            f"{(n_cols >= 2).mean():.1%} have two or more (max {int(n_cols.max())}). "
            f"Mean target with any missing field: {y[any_missing].mean():.2f} vs {y[~any_missing].mean():.2f} without.")
        co = pd.DataFrame(index=cols.index, columns=cols.index, dtype=float)
        for a in cols.index:
            for b in cols.index:
                co.loc[a, b] = (mask[a] & mask[b]).mean() / mask[a].mean() if mask[a].any() else np.nan
        co.index.name = "P(col missing | row missing)"
        display_dataframe(co, "co-missingness: P(column missing | row missing)", bars=False, digits=2)


class TargetReporter:
    """Target distribution and its spread across the strongest raw categorical."""

    def __init__(self, by: str = "Road_traffic_density", bins: int = 45, figsize=(11, 3.4)):
        self.by = by
        self.bins = bins
        self.figsize = figsize

    def display(self, X=None, y=None, **_):
        y = pd.Series(np.asarray(y, dtype=float))
        display_dataframe(y.describe().to_frame("target").T.assign(skew=y.skew()), "target statistics (minutes)",
                          bars=False, digits=2)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        axes[0].hist(y, bins=self.bins, color="tab:blue", alpha=0.85)
        axes[0].set(title=f"target (skew {y.skew():.2f})", xlabel="Time_taken (min)", ylabel="count")
        cat = X[self.by].astype(str).str.strip().to_numpy()
        levels = [l for l in pd.Series(cat).value_counts().index if l != "NaN"] + (["NaN"] if "NaN" in cat else [])
        axes[1].boxplot([y[cat == l] for l in levels], tick_labels=levels, showfliers=False)
        axes[1].set(title=f"target by {self.by}", ylabel="min")
        fig.tight_layout()
        display_figure(fig)


class FeatureSummaryReporter:
    """Summary statistics of the engineered numeric features (after cleaning)."""

    def __init__(self, columns=None):
        self.columns = columns

    def display(self, features=None, **_):
        cols = [c for c in (self.columns or NUMERIC_FEATURES) if c in features]
        num = features[cols].astype(float)
        tab = num.describe().T[["count", "mean", "std", "min", "50%", "max"]]
        tab["missing_share"] = 1 - tab["count"] / len(num)
        tab.index.name = "feature"
        display_dataframe(tab.drop(columns="count"), "engineered numeric features", highlight_best=False, digits=2)
        sizes = {"numeric": len(cols), "ordinal": len([c for c in ORDINAL_FEATURES if c in features]),
                 "nominal": len([c for c in NOMINAL_FEATURES if c in features])}
        display_info_box(f"{len(ALL_FEATURES)} engineered features: " + ", ".join(f"{v} {k}" for k, v in sizes.items())
                         + f" — from {len(features):,} rows.")


class CorrelationReporter:
    """Spearman correlation of numeric features with the target, and between features."""

    def __init__(self, method: str = "spearman", high: float = 0.9, figsize=(7.5, 6.5)):
        self.method = method
        self.high = high
        self.figsize = figsize

    def display(self, features=None, y=None, **_):
        cols = [c for c in NUMERIC_FEATURES if c in features and features[c].nunique(dropna=True) > 1]
        num = features[cols].astype(float)
        y = pd.Series(np.asarray(y, dtype=float), index=num.index)
        with_target = num.corrwith(y, method=self.method).rename(f"{self.method} with target").to_frame()
        with_target["abs"] = with_target.iloc[:, 0].abs()
        with_target = with_target.sort_values("abs", ascending=False).drop(columns="abs")
        with_target.index.name = "feature"
        display_dataframe(with_target, f"{self.method} correlation with the target", highlight_best=False, digits=3)

        corr = num.corr(method=self.method)
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:]
                 if abs(corr.loc[a, b]) >= self.high]
        if pairs:
            tab = pd.DataFrame(pairs, columns=["feature_a", "feature_b", self.method])
            tab.index = pd.Index([f"{a} ~ {b}" for a, b in zip(tab["feature_a"], tab["feature_b"])], name="pair")
            display_dataframe(tab[[self.method]].sort_values(self.method, key=abs, ascending=False),
                              f"feature pairs with |{self.method}| ≥ {self.high} (redundant for linear models)",
                              highlight_best=False, digits=3)
        else:
            display_info_box(f"no feature pair with |{self.method}| ≥ {self.high}.", kind="success")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=8)
        ax.set_yticks(range(len(cols)), cols, fontsize=8)
        ax.set_title(f"{self.method} correlation between numeric features")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        display_figure(fig)


class CategoricalEffectReporter:
    """Mean target per level of each categorical feature (with counts), and a figure for the main drivers."""

    def __init__(self, columns=("Road_traffic_density", "multiple_deliveries", "Festival", "Weatherconditions",
                                "City", "Type_of_vehicle", "Type_of_order", "driver_city"),
                 plot=("Road_traffic_density", "multiple_deliveries", "Festival", "Weatherconditions"),
                 max_levels: int = 12, figsize=(12, 3.2)):
        self.columns = tuple(columns)
        self.plot = tuple(plot)
        self.max_levels = max_levels
        self.figsize = figsize

    def _table(self, features, y, col):
        s = features[col]
        s = s.astype(object).where(pd.notna(s), "unknown").astype(str) if col not in NUMERIC_FEATURES \
            else s.map(lambda v: f"{v:g}" if pd.notna(v) else "unknown")
        g = pd.DataFrame({"level": s.to_numpy(), "y": y.to_numpy()}).groupby("level")["y"]
        tab = pd.DataFrame({"n": g.size(), "share": g.size() / len(s), "mean_target": g.mean(), "std_target": g.std()})
        return tab.sort_values("mean_target", ascending=False).head(self.max_levels)

    def display(self, features=None, y=None, **_):
        y = pd.Series(np.asarray(y, dtype=float), index=features.index)
        rows = []
        for col in self.columns:
            if col not in features:
                continue
            t = self._table(features, y, col)
            t.insert(0, "feature", col)
            rows.append(t.reset_index())
        tab = pd.concat(rows).set_index(["feature", "level"])
        tab["lift_vs_mean"] = tab["mean_target"] - y.mean()
        display_dataframe(style_table(tab, highlight_best=False,
                                      caption="mean target per level (top levels by mean; 'unknown' = missing)"))

        import matplotlib.pyplot as plt

        cols = [c for c in self.plot if c in features]
        fig, axes = plt.subplots(1, len(cols), figsize=self.figsize)
        for ax, col in zip(np.atleast_1d(axes), cols):
            t = self._table(features, y, col).sort_values("mean_target")
            ax.barh(t.index.astype(str), t["mean_target"], color="tab:blue", alpha=0.85)
            ax.axvline(y.mean(), color="tab:red", ls="--", lw=1)
            ax.set(title=col, xlabel="mean target (min)")
            ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        display_figure(fig)


class PreAnalysisReporter:
    """Compose the pre-analysis sections; ``None`` skips a section.

    ``display(X=raw frame, y=target, features=engineered frame)``.
    """

    def __init__(
        self,
        missingness_reporter: Reporter | None = MissingnessReporter(),
        target_reporter: Reporter | None = TargetReporter(),
        feature_reporter: Reporter | None = FeatureSummaryReporter(),
        correlation_reporter: Reporter | None = CorrelationReporter(),
        categorical_reporter: Reporter | None = CategoricalEffectReporter(),
    ):
        self.missingness_reporter = missingness_reporter
        self.target_reporter = target_reporter
        self.feature_reporter = feature_reporter
        self.correlation_reporter = correlation_reporter
        self.categorical_reporter = categorical_reporter

    def sections(self):
        return [
            ("[Analysis] Missing values", self.missingness_reporter),
            ("[Analysis] Target", self.target_reporter),
            ("[Analysis] Engineered features", self.feature_reporter),
            ("[Analysis] Correlations", self.correlation_reporter),
            ("[Analysis] Categorical effects", self.categorical_reporter),
        ]

    def display(self, exp_name: str = "", **context) -> None:
        display_title("#", f"Pre-analysis: {exp_name}")
        display_info_box("Everything in this section is computed on the training rows only, before any model is "
                         "fitted; it informs feature choices, it is not used for model selection.")
        for title, reporter in self.sections():
            if reporter is None:
                continue
            display_title("##", title)
            reporter.display(**context)


# ---------------------------------------------------------------------------
class PostAnalysisReporter:
    """Compose the post-analysis sections; ``None`` skips a section."""

    def __init__(
        self,
        data_reporter: Reporter | None = DataReporter(),
        leaderboard_reporter: Reporter | None = LeaderboardReporter(),
        training_reporter: Reporter | None = TrainingReporter(),
        experiments_reporter: Reporter | None = ExperimentsReporter(),
        error_reporter: Reporter | None = ErrorAnalysisReporter(),
        submission_reporter: Reporter | None = SubmissionReporter(),
    ):
        self.data_reporter = data_reporter
        self.leaderboard_reporter = leaderboard_reporter
        self.training_reporter = training_reporter
        self.experiments_reporter = experiments_reporter
        self.error_reporter = error_reporter
        self.submission_reporter = submission_reporter

    def sections(self):
        return [
            ("[Analysis] Data", self.data_reporter),
            ("[Analysis] Cross-validated leaderboard", self.leaderboard_reporter),
            ("[Analysis] Training diagnostics", self.training_reporter),
            ("[Analysis] Experiments", self.experiments_reporter),
            ("[Analysis] Error analysis", self.error_reporter),
            ("[Analysis] Submission", self.submission_reporter),
        ]

    def display(self, result, **context) -> None:
        display_title("#", f"Post-analysis: {result.exp_name}")
        for title, reporter in self.sections():
            if reporter is None:
                continue
            display_title("##", title)
            reporter.display(result, **context)
