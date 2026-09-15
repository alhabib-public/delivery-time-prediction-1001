"""Loading the assignment CSVs and writing the submission file.

The raw files are deliberately loaded *as strings* wherever the source is noisy
(``"NaN"`` literals, trailing whitespace, ``"(min) 24"`` targets). All cleaning
happens inside the sklearn pipeline (see :mod:`delivery.preprocessing`) so that
``test.csv`` goes through exactly the same code path as ``train.csv``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Time_taken(min)"            # column name in train.csv
SUBMISSION_TARGET = "Time_taken (min)"  # column name in Sample_Submission.csv (note the space)
ID = "ID"

_HERE = Path(__file__).resolve().parent


def data_dir() -> Path:
    """Locate the ``dataset/`` folder: ``$DELIVERY_DATA_DIR`` if set, else ``<repo>/dataset``."""
    env = os.environ.get("DELIVERY_DATA_DIR")
    candidates = [Path(env)] if env else []
    candidates.append(_HERE.parent / "dataset")
    for c in candidates:
        if (c / "train.csv").exists():
            return c
    raise FileNotFoundError(
        f"dataset/train.csv not found (looked in {[str(c) for c in candidates]}); "
        "set DELIVERY_DATA_DIR to the folder holding train.csv/test.csv/Sample_Submission.csv"
    )


def _read_raw(path: Path) -> pd.DataFrame:
    # keep_default_na=False: the literal string "NaN" must reach the cleaner unchanged
    # so that *all* missing-value handling lives in one place (DeliveryCleaner).
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_target(series: pd.Series) -> pd.Series:
    """``"(min) 24 "`` -> ``24.0``."""
    s = series.astype(str).str.replace("(min)", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce").astype(float)


def load_train(path: str | os.PathLike | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with ``y`` parsed to float minutes and ``X`` still raw strings."""
    path = Path(path) if path is not None else data_dir() / "train.csv"
    df = _read_raw(path)
    y = parse_target(df[TARGET]).rename("time_taken_min")
    X = df.drop(columns=[TARGET])
    return X, y


def load_test(path: str | os.PathLike | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else data_dir() / "test.csv"
    return _read_raw(path)


def load_sample_submission(path: str | os.PathLike | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else data_dir() / "Sample_Submission.csv"
    return _read_raw(path)


def make_submission(ids: pd.Series | np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    """Build the submission frame in the exact ``Sample_Submission.csv`` layout."""
    ids = pd.Series(ids).astype(str).str.strip().to_numpy()
    preds = np.asarray(predictions, dtype=float).ravel()
    if len(ids) != len(preds):
        raise ValueError(f"{len(ids)} ids but {len(preds)} predictions")
    if not np.isfinite(preds).all():
        raise ValueError("predictions contain NaN/inf")
    return pd.DataFrame({ID: ids, SUBMISSION_TARGET: preds})


def write_submission(
    ids: pd.Series | np.ndarray,
    predictions: np.ndarray,
    path: str | os.PathLike,
    sample: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write ``ID,Time_taken (min)`` and, if a sample is given, verify header + ID order."""
    sub = make_submission(ids, predictions)
    if sample is not None:
        if list(sample.columns) != list(sub.columns):
            raise ValueError(f"columns {list(sub.columns)} != sample {list(sample.columns)}")
        sample_ids = sample[ID].astype(str).str.strip().to_numpy()
        if len(sample_ids) != len(sub) or not (sample_ids == sub[ID].to_numpy()).all():
            raise ValueError("submission IDs differ from Sample_Submission.csv (count or order)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return sub
