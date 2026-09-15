"""Models: a sklearn-compliant PyTorch MLP plus the candidate registry.

``TorchMLPRegressor`` follows the scikit-learn estimator contract (sklearn 1.9):

* ``__init__`` only stores hyper-parameters (so ``clone``/``get_params`` work),
* ``fit`` validates with ``validate_data``, sets ``n_features_in_`` and trailing-
  underscore attributes, returns ``self``,
* ``predict`` calls ``check_is_fitted`` and re-validates with ``reset=False``,
* tags are declared through ``__sklearn_tags__``.

It is verified with ``sklearn.utils.estimator_checks.check_estimator`` in the tests.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import nn

from .preprocessing import NOMINAL_FEATURES, DeliveryCleaner, DeliveryFeatureEngineer, make_preprocessor


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    """Multi-layer perceptron regressor trained with PyTorch.

    Parameters
    ----------
    hidden : tuple of int
        Width of each hidden layer.
    dropout : float
        Dropout after every hidden layer.
    lr, weight_decay : float
        AdamW settings.
    epochs : int
        Maximum number of passes over the data.
    batch_size : int
    loss : {"l1", "huber", "mse"}
        ``"l1"`` optimises MAE directly (the assignment metric).
    early_stopping : bool
        Hold out ``val_fraction`` of the training rows, stop after ``patience``
        epochs without improvement and restore the best weights.
    val_fraction : float
    patience : int
    random_state : int or None
    device : str
        ``"cpu"`` in the workbook container (no GPU passthrough).
    verbose : int
        ``1`` prints one line per epoch.

    Attributes
    ----------
    model_ : torch.nn.Module
    n_features_in_ : int
    n_iter_ : int
        Epochs actually run.
    loss_curve_, val_curve_ : list of float
        Training loss per epoch and validation MAE per epoch (empty if no early stopping).
    """

    def __init__(
        self,
        hidden=(256, 128),
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=60,
        batch_size=512,
        loss="l1",
        early_stopping=True,
        val_fraction=0.1,
        patience=8,
        random_state=None,
        device="cpu",
        verbose=0,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.early_stopping = early_stopping
        self.val_fraction = val_fraction
        self.patience = patience
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    # -- sklearn plumbing ---------------------------------------------------
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.non_deterministic = False
        return tags

    def _loss_fn(self):
        if self.loss == "l1":
            return nn.L1Loss()
        if self.loss == "huber":
            return nn.HuberLoss(delta=1.0)
        if self.loss == "mse":
            return nn.MSELoss()
        raise ValueError(f"loss must be 'l1', 'huber' or 'mse', got {self.loss!r}")

    def _build(self, n_in: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev = n_in
        for width in self.hidden:
            layers += [nn.Linear(prev, int(width)), nn.ReLU()]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = int(width)
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    # -- fit / predict --------------------------------------------------------
    def fit(self, X, y):
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        loss_fn = self._loss_fn()
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")

        rng = check_random_state(self.random_state)
        seed = int(rng.randint(0, 2**31 - 1))
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        device = torch.device(self.device)

        # target standardisation makes the optimiser scale-free; undone in predict
        self.y_mean_ = float(y.mean())
        self.y_scale_ = float(y.std()) or 1.0
        y_std = (y - self.y_mean_) / self.y_scale_

        n = X.shape[0]
        use_val = self.early_stopping and n >= 10 and 0 < self.val_fraction < 1
        if use_val:
            perm = rng.permutation(n)
            n_val = max(1, int(round(n * self.val_fraction)))
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
        else:
            tr_idx, val_idx = np.arange(n), np.array([], dtype=int)

        Xt = torch.tensor(X[tr_idx], device=device)
        yt = torch.tensor(y_std[tr_idx], device=device).unsqueeze(1)
        if use_val:
            Xv = torch.tensor(X[val_idx], device=device)
            yv = torch.tensor(y[val_idx], device=device)

        model = self._build(X.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=min(int(self.batch_size), len(tr_idx)),
            shuffle=True,
            generator=gen,
        )

        self.loss_curve_, self.val_curve_ = [], []
        best_mae, best_state, bad_epochs = np.inf, None, 0
        for epoch in range(int(self.epochs)):
            model.train()
            total, count = 0.0, 0
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
                count += len(xb)
            sched.step()
            self.loss_curve_.append(total / max(count, 1))
            self.n_iter_ = epoch + 1

            if use_val:
                model.eval()
                with torch.no_grad():
                    pred = model(Xv).squeeze(1) * self.y_scale_ + self.y_mean_
                    mae = float((pred - yv).abs().mean())
                self.val_curve_.append(mae)
                if self.verbose:
                    print(f"epoch {epoch + 1:3d}  train_loss={self.loss_curve_[-1]:.4f}  val_mae={mae:.4f}")
                if mae < best_mae - 1e-6:
                    best_mae, bad_epochs = mae, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        break
            elif self.verbose:
                print(f"epoch {epoch + 1:3d}  train_loss={self.loss_curve_[-1]:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
            self.best_val_mae_ = best_mae
        model.eval()
        self.model_ = model
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = validate_data(self, X, reset=False, dtype=np.float32)
        with torch.no_grad():
            out = self.model_(torch.tensor(X, device=torch.device(self.device))).squeeze(1)
        return (out.cpu().numpy() * self.y_scale_ + self.y_mean_).astype(np.float64)


# ---------------------------------------------------------------------------
# candidate registry: name -> full pipeline (clean -> features -> encode -> model)
# ---------------------------------------------------------------------------
def make_pipeline(model, kind: str = "tree") -> Pipeline:
    """Wrap any regressor in the full preprocessing pipeline."""
    return Pipeline([
        ("clean", DeliveryCleaner()),
        ("features", DeliveryFeatureEngineer()),
        ("encode", make_preprocessor(kind)),
        ("model", model),
    ])


def make_candidate(name: str, random_state: int = 0) -> Pipeline:
    """Build one named candidate. See ``CANDIDATE_NAMES``."""
    if name == "median":
        return make_pipeline(DummyRegressor(strategy="median"), kind="tree")
    if name == "ridge":
        return make_pipeline(Ridge(alpha=1.0), kind="dense")
    if name == "hgb":
        return make_pipeline(
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.05,
                max_iter=600,
                max_leaf_nodes=63,
                min_samples_leaf=40,
                l2_regularization=1.0,
                categorical_features=NOMINAL_FEATURES,
                random_state=random_state,
            ),
            kind="tree",
        )
    if name == "mlp":
        return make_pipeline(
            TorchMLPRegressor(
                hidden=(256, 128),
                dropout=0.1,
                lr=2e-3,
                weight_decay=1e-4,
                epochs=80,
                batch_size=512,
                loss="huber",     # best of l1 / huber / mse on CV (see delivery.experiments)
                early_stopping=True,
                val_fraction=0.1,
                patience=10,
                random_state=random_state,
            ),
            kind="dense",
        )
    raise ValueError(f"unknown candidate {name!r}; choose from {CANDIDATE_NAMES}")


CANDIDATE_NAMES = ("median", "ridge", "hgb", "mlp")
