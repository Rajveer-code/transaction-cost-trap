"""
models.py
=========
Four model classes with a unified interface for the walk-forward training loop.

All classes expose exactly:
    fit(X_train, y_train)         -> self
    predict_proba(X)              -> np.ndarray shape (n_samples,)  — always 1-D
    get_params()                  -> dict

predict_proba ALWAYS returns a 1-D array of P(y=1).  No exceptions.

UNIFIED INTERFACE CONTRACT:
  - X_train, X_test are scaled numpy arrays (output of walk_forward.get_fold_arrays)
  - y_train is a 1-D binary float array {0.0, 1.0}
  - Each model is stateless between folds: always call fit() before predict_proba()

HYPERPARAMETER TUNING:
  tune_hyperparameters() wraps GridSearchCV with TimeSeriesSplit and must ONLY
  be called on the model-train portion of a fold.  Calling it on full-dataset
  data is lookahead bias.

Author: Rajveer Singh Pall
Paper : "Overcoming the Transaction Cost Trap: Cross-Sectional Conviction
         Ranking in Machine Learning Equity Prediction"
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np

TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
except Exception as e:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    print(f"[WARNING] PyTorch unavailable: {e}. DNNModel and EnsembleModel will use fallback (CatBoost+RF).")

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


# ---------------------------------------------------------------------------
# TYPE ALIAS
# ---------------------------------------------------------------------------

ParamGrid = Dict[str, list]


# ---------------------------------------------------------------------------
# CLASS 1: CatBoostModel
# ---------------------------------------------------------------------------

class CatBoostModel:
    """
    CatGradient Boosting classifier wrapped with a unified predict_proba interface.

    Uses an internal 15% temporal validation split for early stopping.  The
    split is NOT shuffled — the final 15% of training rows (time-ordered) form
    the eval set, preserving temporal ordering.

    HYPERPARAMETER GRID (class attribute PARAM_GRID):
        depth        : [4, 6, 8]
        learning_rate: [0.01, 0.05, 0.1]
        l2_leaf_reg  : [1, 3, 5]
    """

    PARAM_GRID: ParamGrid = {
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5],
    }

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        verbose: int = 0,
        eval_metric: str = "AUC",
        early_stopping_rounds: int = 50,
    ) -> None:
        """
        Parameters
        ----------
        iterations : int
            Maximum number of boosting rounds.
        learning_rate : float
            Step size shrinkage.
        depth : int
            Maximum tree depth.
        l2_leaf_reg : float
            L2 regularisation on leaf values.
        random_seed : int
            Reproducibility seed.
        verbose : int
            0 = silent.  Set > 0 only for debugging.
        eval_metric : str
            Metric monitored on the eval set for early stopping.
        early_stopping_rounds : int
            Stop if eval metric does not improve for this many rounds.
        """
        self._params = dict(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            verbose=verbose,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.model: Optional[CatBoostClassifier] = None
        self.best_iteration_: Optional[int] = None

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CatBoostModel":
        """
        Fit CatBoost with an internal temporal eval split for early stopping.

        Parameters
        ----------
        X_train : np.ndarray, shape (n, n_features)
            Scaled, time-ordered feature matrix.  MUST be sorted by time.
        y_train : np.ndarray, shape (n,)
            Binary labels {0.0, 1.0}.

        Returns
        -------
        self

        Notes
        -----
        The eval set is the final 15% of rows (time order preserved).
        Shuffling the split would constitute lookahead bias for time series.
        """
        assert X_train.shape[0] == y_train.shape[0], "X/y length mismatch."
        assert X_train.shape[0] > 10, "Training set too small for CatBoost."

        val_idx = int(0.85 * len(X_train))
        X_tr, X_val = X_train[:val_idx], X_train[val_idx:]
        y_tr, y_val = y_train[:val_idx], y_train[val_idx:]

        params = {k: v for k, v in self._params.items()
                  if k not in ("eval_metric", "early_stopping_rounds")}
        self.model = CatBoostClassifier(
            **params,
            eval_metric=self._params["eval_metric"],
            early_stopping_rounds=self._params["early_stopping_rounds"],
        )
        self.model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            verbose=self._params["verbose"],
        )
        self.best_iteration_ = self.model.get_best_iteration()
        print(f"  CatBoost: stopped at iter {self.best_iteration_}")
        return self

    # ── predict_proba ────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return P(y=1) as a 1-D array.

        Parameters
        ----------
        X : np.ndarray, shape (n, n_features)

        Returns
        -------
        np.ndarray, shape (n,)
            Values in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("CatBoostModel.fit() must be called before predict_proba().")
        return self.model.predict_proba(X)[:, 1]

    # ── get_params ───────────────────────────────────────────────────────────

    def get_params(self) -> Dict[str, Any]:
        """Return all hyperparameters as a flat dictionary."""
        return dict(self._params)


# ---------------------------------------------------------------------------
# CLASS 2: RandomForestModel
# ---------------------------------------------------------------------------

class RandomForestModel:
    """
    Scikit-learn RandomForestClassifier wrapped with a unified interface.

    class_weight='balanced' is set by default to compensate for the ~53/47
    target imbalance observed in the dataset (8652 ups vs 7553 downs).
    This up-weights the minority class during tree construction without any
    explicit resampling.

    HYPERPARAMETER GRID (class attribute PARAM_GRID):
        max_depth       : [6, 8, 10]
        min_samples_leaf: [10, 20, 40]
        max_features    : ['sqrt', 'log2']
    """

    PARAM_GRID: ParamGrid = {
        "max_depth": [6, 8, 10],
        "min_samples_leaf": [10, 20, 40],
        "max_features": ["sqrt", "log2"],
    }

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        min_samples_leaf: int = 20,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
        class_weight: str = "balanced",
    ) -> None:
        """
        Parameters
        ----------
        n_estimators : int
            Number of trees.
        max_depth : int
            Maximum depth per tree.
        min_samples_leaf : int
            Minimum samples required at a leaf node.
        max_features : str
            Feature subset strategy: 'sqrt' or 'log2'.
        n_jobs : int
            -1 = use all available cores.
        random_state : int
        class_weight : str
            'balanced' compensates for the ~53/47 label imbalance.
        """
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight,
        )
        self.model: Optional[RandomForestClassifier] = None

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestModel":
        """
        Fit the Random Forest.

        Parameters
        ----------
        X_train : np.ndarray, shape (n, n_features)
        y_train : np.ndarray, shape (n,)

        Returns
        -------
        self
        """
        assert X_train.shape[0] == y_train.shape[0], "X/y length mismatch."
        self.model = RandomForestClassifier(**self._params)
        self.model.fit(X_train, y_train)
        return self

    # ── predict_proba ────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return P(y=1) as a 1-D array.

        Parameters
        ----------
        X : np.ndarray, shape (n, n_features)

        Returns
        -------
        np.ndarray, shape (n,)
            Values in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("RandomForestModel.fit() must be called before predict_proba().")
        return self.model.predict_proba(X)[:, 1]

    # ── get_params ───────────────────────────────────────────────────────────

    def get_params(self) -> Dict[str, Any]:
        """Return all hyperparameters as a flat dictionary."""
        return dict(self._params)


# ---------------------------------------------------------------------------
# CLASS 3: DNNModel (PyTorch MLP)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class _FinancialMLP(nn.Module):
        """
        Private MLP backbone used by DNNModel.

    Architecture:
        Linear(in_features, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.3)
        Linear(256, 128)         -> BatchNorm1d(128) -> ReLU -> Dropout(0.3)
        Linear(128, 64)          -> BatchNorm1d(64)  -> ReLU -> Dropout(0.2)
        Linear(64, 1)            -> Sigmoid

    BatchNorm is placed before ReLU (pre-activation style) to stabilise
    training on financial feature distributions that vary substantially
    across walk-forward folds.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


if TORCH_AVAILABLE:
    class DNNModel:
        """
        PyTorch MLP classifier with early stopping and LR scheduling.

    Validation split: last 15% of the training rows (time-ordered, NOT random).
    Early stopping monitors validation BCE loss with patience=15.
    LR scheduler reduces LR on validation loss plateau (patience=7).

    POSITIONAL NOTE ON INPUT DIMENSION:
        The default in_features=47 matches the paper's 47-feature matrix.
        Pass a different value only if the feature set has changed.

    DEVICE:
        Automatically uses CUDA if available, otherwise CPU.
        Training is transparent to the caller — predict_proba always returns
        a CPU numpy array regardless of device.
    """

    PARAM_GRID: ParamGrid = {
        "learning_rate": [1e-4, 1e-3, 5e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    }

    def __init__(
        self,
        in_features: int = 47,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        batch_size: int = 256,
        patience: int = 15,
        scheduler_patience: int = 7,
        scheduler_factor: float = 0.5,
        random_seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        in_features : int
            Number of input features.  Must match feature_cols length.
        learning_rate : float
        weight_decay : float
            L2 regularisation in Adam (weight decay).
        max_epochs : int
            Hard ceiling on training epochs.
        batch_size : int
        patience : int
            Early stopping patience (epochs without val loss improvement).
        scheduler_patience : int
            ReduceLROnPlateau patience.
        scheduler_factor : float
            LR reduction factor on plateau.
        random_seed : int
        """
        self._params = dict(
            in_features=in_features,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            random_seed=random_seed,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[_FinancialMLP] = None
        self.best_epoch_: Optional[int] = None
        self.best_val_loss_: Optional[float] = None

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DNNModel":
        """
        Train the MLP with early stopping and LR scheduling.

        Parameters
        ----------
        X_train : np.ndarray, shape (n, n_features)
            Scaled, time-ordered feature matrix.  The final 15% of rows form
            the internal validation set (no shuffling).
        y_train : np.ndarray, shape (n,)
            Binary labels {0.0, 1.0}.

        Returns
        -------
        self

        Notes
        -----
        Best model weights are restored after early stopping triggers.
        The scaler should already have been applied to X_train before calling
        this method (done by walk_forward.get_fold_arrays).
        """
        assert X_train.shape[0] == y_train.shape[0], "X/y length mismatch."
        assert X_train.shape[0] > 20, "Training set too small for DNN."

        torch.manual_seed(self._params["random_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._params["random_seed"])

        # ── Train / val split (time-ordered) ────────────────────────────────
        val_idx = int(0.85 * len(X_train))
        X_tr, X_val = X_train[:val_idx], X_train[val_idx:]
        y_tr, y_val = y_train[:val_idx], y_train[val_idx:]

        # ── Tensors ─────────────────────────────────────────────────────────
        Xtr_t = torch.from_numpy(X_tr.astype(np.float32)).to(self.device)
        ytr_t = torch.from_numpy(y_tr.astype(np.float32)).to(self.device)
        Xval_t = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
        yval_t = torch.from_numpy(y_val.astype(np.float32)).to(self.device)

        # ── Model, loss, optimiser, scheduler ───────────────────────────────
        self.model = _FinancialMLP(in_features=self._params["in_features"]).to(self.device)
        criterion = nn.BCELoss()
        optimiser = torch.optim.Adam(
            self.model.parameters(),
            lr=self._params["learning_rate"],
            weight_decay=self._params["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            patience=self._params["scheduler_patience"],
            factor=self._params["scheduler_factor"],
        )

        # ── Training loop ────────────────────────────────────────────────────
        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        patience_counter = 0
        batch_size = self._params["batch_size"]
        max_epochs = self._params["max_epochs"]
        pat = self._params["patience"]

        for epoch in range(1, max_epochs + 1):
            # Shuffle training batch order each epoch (within-epoch only;
            # the train/val split itself is never shuffled)
            perm = torch.randperm(len(Xtr_t), device=self.device)
            Xtr_shuf = Xtr_t[perm]
            ytr_shuf = ytr_t[perm]

            # ── Train epoch ──────────────────────────────────────────────────
            self.model.train()
            train_loss_accum = 0.0
            n_batches = 0
            for start in range(0, len(Xtr_t), batch_size):
                xb = Xtr_shuf[start: start + batch_size]
                yb = ytr_shuf[start: start + batch_size]
                optimiser.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimiser.step()
                train_loss_accum += loss.item()
                n_batches += 1

            # ── Val loss ─────────────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(Xval_t)
                val_loss = criterion(val_preds, yval_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1
                if patience_counter >= pat:
                    break

        # ── Restore best weights ─────────────────────────────────────────────
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.best_epoch_ = best_epoch if "best_epoch" in dir() else epoch
        self.best_val_loss_ = best_val_loss
        print(
            f"  DNN: stopped epoch {epoch}/{max_epochs}, "
            f"best val loss {best_val_loss:.4f}"
        )
        return self

    # ── predict_proba ────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return P(y=1) as a 1-D numpy array on CPU.

        Parameters
        ----------
        X : np.ndarray, shape (n, n_features)

        Returns
        -------
        np.ndarray, shape (n,)
            Values in [0, 1] (from Sigmoid output layer).

        Notes
        -----
        Processed in batches of 512 to avoid GPU OOM on large test sets.
        Always returns a CPU array regardless of training device.
        """
        if self.model is None:
            raise RuntimeError("DNNModel.fit() must be called before predict_proba().")
        self.model.eval()
        results = []
        X_t = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            for start in range(0, len(X_t), 512):
                batch = X_t[start: start + 512].to(self.device)
                out = self.model(batch).cpu().numpy()
                results.append(out)
        return np.concatenate(results, axis=0)

    # ── get_params ───────────────────────────────────────────────────────────

    def get_params(self) -> Dict[str, Any]:
        """Return all hyperparameters as a flat dictionary."""
        return dict(self._params)

else:
    class DNNModel:
        def __init__(self, *args, **kwargs):
            self.available = False
            print("[WARNING] DNNModel is disabled because PyTorch is unavailable.")

        def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DNNModel":
            # no-op to preserve interface; ensemble handles missing DNN.
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            raise RuntimeError("DNNModel is unavailable due to missing PyTorch.")

        def get_params(self) -> Dict[str, Any]:
            return {}


# ---------------------------------------------------------------------------
# CLASS 4: EnsembleModel
# ---------------------------------------------------------------------------

class EnsembleModel:
    """
    Equal-weight average ensemble of CatBoostModel, RandomForestModel, DNNModel.

    All component models are fitted on identical (X_train, y_train).
    predict_proba returns a weighted mean of the calibrated probability outputs.

    The ensemble gracefully degrades to CatBoost+RandomForest when DNN is unavailable.
    """

    def __init__(self) -> None:
        self.catboost = CatBoostModel()
        self.rf = RandomForestModel()
        self.dnn = DNNModel()
        self.dnn_available = getattr(self.dnn, "available", True)
        self._fitted = False

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "EnsembleModel":
        """
        Fit all three component models on identical training data.

        Parameters
        ----------
        X_train : np.ndarray, shape (n, n_features)
        y_train : np.ndarray, shape (n,)

        Returns
        -------
        self
        """
        self.catboost.fit(X_train, y_train)
        print("  Ensemble: CatBoost OK", end="  ")
        self.rf.fit(X_train, y_train)
        print("RF OK", end="  ")

        if self.dnn_available:
            self.dnn.fit(X_train, y_train)
            print("DNN OK")
        else:
            print("DNN skipped (PyTorch unavailable)")

        self._fitted = True
        return self

    # ── predict_proba ────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return equal-weight average probability as a 1-D array.

        Parameters
        ----------
        X : np.ndarray, shape (n, n_features)

        Returns
        -------
        np.ndarray, shape (n,)
            Values in [0, 1].

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        AssertionError
            If component predictions have incompatible shapes or the
            averaged output is outside [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("EnsembleModel.fit() must be called before predict_proba().")

        p_cb = self.catboost.predict_proba(X)
        p_rf = self.rf.predict_proba(X)

        if self.dnn_available:
            p_dnn = self.dnn.predict_proba(X)
            assert p_cb.shape == p_rf.shape == p_dnn.shape, (
                f"Component prediction shape mismatch: CB={p_cb.shape}, "
                f"RF={p_rf.shape}, DNN={p_dnn.shape}"
            )
            ensemble = (p_cb + p_rf + p_dnn) / 3.0
        else:
            assert p_cb.shape == p_rf.shape, (
                f"Component prediction shape mismatch: CB={p_cb.shape}, RF={p_rf.shape}"
            )
            ensemble = (p_cb + p_rf) / 2.0

        assert ensemble.shape == (len(X),), (
            f"Ensemble output shape {ensemble.shape} != ({len(X)},)"
        )
        assert ensemble.min() >= 0.0 and ensemble.max() <= 1.0, (
            f"Ensemble probabilities out of [0,1]: "
            f"min={ensemble.min():.4f}, max={ensemble.max():.4f}"
        )
        return ensemble

    # ── get_params ───────────────────────────────────────────────────────────

    def get_params(self) -> Dict[str, Any]:
        """Return nested dict of all component hyperparameters."""
        params = {
            "catboost": self.catboost.get_params(),
            "random_forest": self.rf.get_params(),
        }
        if self.dnn is not None:
            params["dnn"] = self.dnn.get_params()
        else:
            params["dnn"] = None
        return params


# ---------------------------------------------------------------------------
# STANDALONE HYPERPARAMETER TUNING
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    model_class: Type,
    param_grid: ParamGrid,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3,
    scoring: str = "roc_auc",
) -> Dict[str, Any]:
    """
    Run nested time-series cross-validation to select hyperparameters.

    Uses sklearn GridSearchCV with TimeSeriesSplit as the inner CV splitter.
    TimeSeriesSplit is chronologically ordered (no shuffling), which preserves
    the temporal structure of the training data.

    Parameters
    ----------
    model_class : Type
        One of CatBoostModel, RandomForestModel, DNNModel.
        DNNModel is NOT compatible with GridSearchCV directly — skip or use
        RandomizedSearchCV with a custom wrapper for the DNN.
    param_grid : dict
        Hyperparameter search grid (use the class's PARAM_GRID attribute).
    X_train : np.ndarray, shape (n, n_features)
        Model-training features for the current fold (80% of train).
    y_train : np.ndarray, shape (n,)
        Binary labels.
    n_splits : int
        Number of inner TimeSeriesSplit splits.
    scoring : str
        Sklearn scoring metric.

    Returns
    -------
    dict
        Best hyperparameters as a flat dictionary.

    CRITICAL WARNING:
        This function MUST ONLY be called on model_train_dates data — i.e.,
        the 80% model-train portion of a single walk-forward fold.
        Calling it on the full dataset or on calibration/test data
        constitutes lookahead bias and will produce inflated estimates.

    Notes
    -----
    CatBoostModel and RandomForestModel use a sklearn-compatible wrapper
    (fit/predict_proba API) so GridSearchCV works directly.
    For DNNModel, use manual epoch sweeping or Optuna instead of GridSearchCV.
    """
    _SKLERAN_COMPATIBLE = (CatBoostModel, RandomForestModel)
    if not issubclass(model_class, _SKLERAN_COMPATIBLE):
        raise ValueError(
            f"tune_hyperparameters does not support {model_class.__name__} with "
            "GridSearchCV.  For DNNModel, tune learning_rate/weight_decay manually."
        )

    # Build a sklearn-compatible wrapper that delegates to our model class
    class _SklearnWrapper:
        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs
            self._model = model_class(**kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "_SklearnWrapper":
            self._model.fit(X, y)
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            p = self._model.predict_proba(X)
            return np.column_stack([1 - p, p])

        def get_params(self, deep: bool = True) -> Dict[str, Any]:
            return dict(self._kwargs)

        def set_params(self, **params: Any) -> "_SklearnWrapper":
            self._kwargs.update(params)
            self._model = model_class(**self._kwargs)
            return self

    tscv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(
        estimator=_SklearnWrapper(),
        param_grid=param_grid,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1,
        refit=False,  # we re-train with best params in the main loop
        verbose=0,
    )
    gs.fit(X_train, y_train.astype(int))
    return gs.best_params_


# ---------------------------------------------------------------------------
# MODULE SELF-TEST (synthetic data — no yfinance, no network calls)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("models.py — self-test (synthetic data)")
    print("=" * 60)

    N_SAMPLES = 800
    N_FEATURES = 47
    rng = np.random.default_rng(seed=0)

    X = rng.standard_normal((N_SAMPLES, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, 2, size=N_SAMPLES).astype(np.float64)

    results: Dict[str, bool] = {}

    # ── 1. CatBoostModel ────────────────────────────────────────────────────
    print("\n[1/4] CatBoostModel …")
    cb = CatBoostModel()
    cb.fit(X, y)
    p_cb = cb.predict_proba(X)
    assert p_cb.ndim == 1, f"predict_proba must be 1-D, got shape {p_cb.shape}"
    assert p_cb.shape == (N_SAMPLES,), f"Shape mismatch: {p_cb.shape}"
    assert p_cb.min() >= 0.0 and p_cb.max() <= 1.0, \
        f"Values outside [0,1]: min={p_cb.min():.4f} max={p_cb.max():.4f}"
    results["CatBoost"] = True
    print("  OK shape OK, values in [0,1]")

    # ── 2. RandomForestModel ─────────────────────────────────────────────────
    print("\n[2/4] RandomForestModel …")
    rf = RandomForestModel()
    rf.fit(X, y)
    p_rf = rf.predict_proba(X)
    assert p_rf.ndim == 1, f"predict_proba must be 1-D, got shape {p_rf.shape}"
    assert p_rf.shape == (N_SAMPLES,), f"Shape mismatch: {p_rf.shape}"
    assert p_rf.min() >= 0.0 and p_rf.max() <= 1.0, \
        f"Values outside [0,1]: min={p_rf.min():.4f} max={p_rf.max():.4f}"
    results["RF"] = True
    print("  OK shape OK, values in [0,1]")

    # ── 3. DNNModel ───────────────────────────────────────────────────────────
    print("\n[3/4] DNNModel …")
    dnn = DNNModel(in_features=N_FEATURES, max_epochs=10, patience=3)
    dnn.fit(X, y)
    p_dnn = dnn.predict_proba(X)
    assert p_dnn.ndim == 1, f"predict_proba must be 1-D, got shape {p_dnn.shape}"
    assert p_dnn.shape == (N_SAMPLES,), f"Shape mismatch: {p_dnn.shape}"
    assert p_dnn.min() >= 0.0 and p_dnn.max() <= 1.0, \
        f"Values outside [0,1]: min={p_dnn.min():.4f} max={p_dnn.max():.4f}"
    results["DNN"] = True
    print("  OK shape OK, values in [0,1]")

    # ── 4. EnsembleModel ─────────────────────────────────────────────────────
    print("\n[4/4] EnsembleModel …")
    # Re-fit individual models to compare against ensemble output
    cb2 = CatBoostModel()
    rf2 = RandomForestModel()
    dnn2 = DNNModel(in_features=N_FEATURES, max_epochs=10, patience=3)
    cb2.fit(X, y)
    rf2.fit(X, y)
    dnn2.fit(X, y)

    ens = EnsembleModel()
    # Replace components with the already-fitted models to guarantee
    # identical weights when verifying the mean
    ens.catboost = cb2
    ens.rf = rf2
    ens.dnn = dnn2
    ens._fitted = True

    p_ens = ens.predict_proba(X)
    expected = (cb2.predict_proba(X) + rf2.predict_proba(X) + dnn2.predict_proba(X)) / 3.0

    assert np.allclose(p_ens, expected, atol=1e-6), \
        "Ensemble output does not equal mean of component probabilities."
    assert p_ens.shape == (N_SAMPLES,), f"Shape mismatch: {p_ens.shape}"
    results["Ensemble"] = True
    print("  OK equal-weight mean verified, shape OK")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_passed = all(results.values())
    marks = "  ".join(
        f"{k} {'OK' if v else '✗'}" for k, v in results.items()
    )
    status = "PASSED" if all_passed else "FAILED"
    print(
        f"\n[PASS] models.py {status}: {marks} "
        f"— all shapes and value ranges verified"
    )
    if not all_passed:
        raise SystemExit(1)
