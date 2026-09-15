import numpy as np
import pandas as pd
import pytest

from delivery.data import (ID, SUBMISSION_TARGET, load_sample_submission, load_test, load_train,
                           make_submission, parse_target, write_submission)
from tests.conftest import requires_data


def test_parse_target_strips_prefix():
    s = pd.Series(["(min) 24", "(min) 33 ", " (min) 10", "garbage"])
    out = parse_target(s)
    assert out.tolist()[:3] == [24.0, 33.0, 10.0]
    assert np.isnan(out.iloc[3])


def test_make_submission_layout():
    sub = make_submission(pd.Series(["0x1 ", "0x2"]), np.array([1.5, 2.5]))
    assert list(sub.columns) == [ID, SUBMISSION_TARGET]
    assert sub[ID].tolist() == ["0x1", "0x2"]


def test_make_submission_rejects_nan_and_length_mismatch():
    with pytest.raises(ValueError):
        make_submission(["a", "b"], np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        make_submission(["a"], np.array([1.0, 2.0]))


def test_write_submission_checks_sample(tmp_path):
    sample = pd.DataFrame({ID: ["a", "b"], SUBMISSION_TARGET: ["1", "2"]})
    out = write_submission(["a", "b"], [10.0, 20.0], tmp_path / "s.csv", sample=sample)
    assert (tmp_path / "s.csv").read_text().splitlines()[0] == f"{ID},{SUBMISSION_TARGET}"
    assert len(out) == 2
    with pytest.raises(ValueError):
        write_submission(["b", "a"], [10.0, 20.0], tmp_path / "s2.csv", sample=sample)


@requires_data
def test_real_files_load_consistently():
    X, y = load_train()
    T = load_test()
    S = load_sample_submission()
    assert len(X) == len(y) and y.notna().all()
    assert 10 <= y.min() and y.max() <= 60
    assert list(T.columns) == [c for c in X.columns]
    assert list(S.columns) == [ID, SUBMISSION_TARGET]
    assert (S[ID].str.strip().to_numpy() == T[ID].str.strip().to_numpy()).all()
