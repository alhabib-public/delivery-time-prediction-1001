import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator

from delivery.models import CANDIDATE_NAMES, TorchMLPRegressor, make_candidate


def test_sklearn_estimator_contract():
    # the full sklearn check suite: clone/get_params/set_params, n_features_in_, fit
    # returns self, predict shape/dtype, unfitted errors, pickling, ...
    # (a config that can actually fit sklearn's small check datasets: the suite asserts R2 > 0.5)
    check_estimator(TorchMLPRegressor(hidden=(32,), dropout=0.0, epochs=150, lr=1e-2, batch_size=64,
                                      early_stopping=False, random_state=0))


def test_mlp_learns_a_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5)).astype(np.float32)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.1 * rng.normal(size=2000) + 10
    est = TorchMLPRegressor(hidden=(32, 32), epochs=40, batch_size=128, lr=5e-3, random_state=0)
    est.fit(X[:1500], y[:1500])
    mae = np.abs(est.predict(X[1500:]) - y[1500:]).mean()
    baseline = np.abs(np.median(y[:1500]) - y[1500:]).mean()
    assert mae < 0.5 * baseline
    assert est.n_features_in_ == 5 and 1 <= est.n_iter_ <= 40
    assert len(est.loss_curve_) == est.n_iter_ == len(est.val_curve_)


def test_mlp_is_deterministic_given_seed():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3)); y = X.sum(1)
    a = TorchMLPRegressor(hidden=(8,), epochs=3, random_state=7).fit(X, y).predict(X)
    b = TorchMLPRegressor(hidden=(8,), epochs=3, random_state=7).fit(X, y).predict(X)
    np.testing.assert_allclose(a, b)


def test_mlp_params_roundtrip_and_validation():
    est = TorchMLPRegressor(loss="huber", epochs=2)
    assert clone(est).get_params() == est.get_params()
    est.set_params(loss="mse"); assert est.get_params()["loss"] == "mse"
    with pytest.raises(ValueError):
        TorchMLPRegressor(loss="nope").fit(np.zeros((10, 2)), np.zeros(10))
    with pytest.raises(ValueError):
        TorchMLPRegressor(epochs=0).fit(np.zeros((10, 2)), np.zeros(10))


@pytest.mark.parametrize("name", CANDIDATE_NAMES)
def test_candidates_fit_and_predict_on_raw_rows(name, toy_raw, toy_y):
    pipe = make_candidate(name, random_state=0)
    if name == "mlp":
        pipe.set_params(model__epochs=2, model__early_stopping=False)
    if name == "hgb":
        pipe.set_params(model__max_iter=5)
    pipe.fit(toy_raw, toy_y)
    pred = pipe.predict(toy_raw)
    assert pred.shape == (4,) and np.isfinite(pred).all()
