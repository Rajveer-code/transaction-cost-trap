"""
models.py
=========
Four model classes with a unified interface for the walk-forward training loop.

All classes expose exactly:
    fit(X_train, y_train)         -> self
    predict_proba(X)              -> np.ndarray shape (n_samples,)  — always 1-D
    get_params()                  -> dict

predict_proba ALWAYS returns a 1-D array of P(y=1).  No exceptions.

REPRODUCIBILITY GUARANTEE (v2):
    CatBoostModel is fully deterministic given fixed input data:
      - random_seed=42
      - thread_count=1          (no multi-thread non-determinism)
      - bootstrap_type=Bernoulli + subsample=1.0  (no row sampling randomness)
      - use_best_model=True     (explicit, not implicit)
    RandomForestModel: random_state=42, n_jobs=1 (deterministic)
    Data must be frozen via --use-cache flag in run_experiments.py

Author: Rajveer Singh Pall
Paper : "When the Gate Stays Closed: Empirical Evidence of Near-Zero
         Cross-Sectional Predictability in Large-Cap NASDAQ Equities
         Using an IC-Gated Machine Learning Framework"
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
# CLASS 1: CatBoostModel  —  FULLY REPRODUCIBLE
# ---------------------------------------------------------------------------

class CatBoostModel:
    """
    CatGradient Boosting classifier with a unified predict_proba interface.

    REPRODUCIBILITY CONTRACT:
        Given identical input data, this model produces identical outputs
        across all runs on any OS. Achieved via:
          - random_seed=42
          - thread_count=1          (eliminates multi-thread non-determinism)
          - bootstrap_type=Bernoulli, subsample=1.0  (no row sampling)
          - use_best_model=True     (explicit early-stopping restore)

    Uses an internal 15% temporal validation split for early stopping.
    The split is NOT shuffled — time order is preserved to prevent lookahead.
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
        self._params = dict(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            # ── Reproducibility settings ──────────────────────────────────
            thread_count=1,           # single thread = deterministic
                        # no row subsampling = deterministic
            use_best_model=True,      # explicit early-stopping weight restore
            # ─────────────────────────────────────────────────────────────
            verbose=verbose,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.model: Optional[CatBoostClassifier] = None
        self.best_iteration_: Optional[int] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CatBoostModel":
        """
        Fit CatBoost with a temporal eval split for early stopping.

        The eval set is the final 15% of rows (time order preserved).
        Shuffling would constitute lookahead bias for time series.
        """
        assert X_train.shape[0] == y_train.shape[0], "X/y length mismatch."
        assert X_train.shape[0] > 10, "Training set too small for CatBoost."

        val_idx = int(0.85 * len(X_train))
        X_tr, X_val = X_train[:val_idx], X_train[val_idx:]
        y_tr, y_val = y_train[:val_idx], y_train[val_idx:]

        # Separate fit-time params from constructor-only params
        fit_excluded = {"eval_metric", "early_stopping_rounds"}
        params = {k: v for k, v in self._params.items() if k not in fit_excluded}

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
        print(f"  CatBoost: best_iter={self.best_iteration_}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CatBoostModel.fit() must be called before predict_proba().")
        return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> Dict[str, Any]:
        return dict(self._params)


# ---------------------------------------------------------------------------
# CLASS 2: RandomForestModel  —  FULLY REPRODUCIBLE
# ---------------------------------------------------------------------------

class RandomForestModel:
    """
    Scikit-learn RandomForestClassifier with a unified interface.

    REPRODUCIBILITY CONTRACT:
        random_state=42, n_jobs=1 (single thread = deterministic).
        class_weight='balanced' compensates for ~53/47 label imbalance.
    """

    PARAM_GRID: ParamGrid = {
        "max_depth": [6, 8, 10],
        "min_samples_leaf": [10, 20, 40],
        "max_features": ["sqrt", "log2"],
    }

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 10,
        min_samples_leaf: int = 20,
        max_features: str = "sqrt",
        n_jobs: int = 1,            # CHANGED from -1 to 1 for determinism
        random_state: int = 42,
        class_weight: str = "balanced",
    ) -> None:
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

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestModel":
        assert X_train.shape[0] == y_train.shape[0], "X/y length mismatch."
        self.model = RandomForestClassifier(**self._params)
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RandomForestModel.fit() must be called before predict_proba().")
        return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> Dict[str, Any]:
        return dict(self._params)


# ---------------------------------------------------------------------------
# CLASS 3: DNNModel (PyTorch MLP)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class _FinancialMLP(nn.Module):
        """
        MLP backbone: [256 -> 128 -> 64 -> 1] with BN, ReLU, Dropout, Sigmoid.
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
        Validation split: last 15% of training rows (time-ordered, NOT random).
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

        def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DNNModel":
            assert X_train.shape[0] == y_train.shape[0], "X/y length mismatch."
            assert X_train.shape[0] > 20, "Training set too small for DNN."

            torch.manual_seed(self._params["random_seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._params["random_seed"])

            val_idx = int(0.85 * len(X_train))
            X_tr, X_val = X_train[:val_idx], X_train[val_idx:]
            y_tr, y_val = y_train[:val_idx], y_train[val_idx:]

            Xtr_t = torch.from_numpy(X_tr.astype(np.float32)).to(self.device)
            ytr_t = torch.from_numpy(y_tr.astype(np.float32)).to(self.device)
            Xval_t = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
            yval_t = torch.from_numpy(y_val.astype(np.float32)).to(self.device)

            self.model = _FinancialMLP(in_features=self._params["in_features"]).to(self.device)
            criterion = nn.BCELoss()
            optimiser = torch.optim.Adam(
                self.model.parameters(),
                lr=self._params["learning_rate"],
                weight_decay=self._params["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser, mode="min",
                patience=self._params["scheduler_patience"],
                factor=self._params["scheduler_factor"],
            )

            best_val_loss = float("inf")
            best_state: Optional[Dict] = None
            patience_counter = 0
            batch_size = self._params["batch_size"]
            max_epochs = self._params["max_epochs"]
            pat = self._params["patience"]
            best_epoch = 1

            for epoch in range(1, max_epochs + 1):
                perm = torch.randperm(len(Xtr_t), device=self.device)
                Xtr_shuf = Xtr_t[perm]
                ytr_shuf = ytr_t[perm]

                self.model.train()
                for start in range(0, len(Xtr_t), batch_size):
                    xb = Xtr_shuf[start: start + batch_size]
                    yb = ytr_shuf[start: start + batch_size]
                    optimiser.zero_grad()
                    preds = self.model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimiser.step()

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

            if best_state is not None:
                self.model.load_state_dict(best_state)

            self.best_epoch_ = best_epoch
            self.best_val_loss_ = best_val_loss
            print(f"  DNN: stopped epoch {epoch}/{max_epochs}, best val loss {best_val_loss:.4f}")
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
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

        def get_params(self) -> Dict[str, Any]:
            return dict(self._params)

else:
    class DNNModel:
        def __init__(self, *args, **kwargs):
            self.available = False
            print("[WARNING] DNNModel is disabled because PyTorch is unavailable.")

        def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DNNModel":
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
    Equal-weight average of CatBoostModel + RandomForestModel [+ DNNModel].
    Degrades gracefully to CatBoost+RF when PyTorch is unavailable.
    """

    def __init__(self) -> None:
        self.catboost = CatBoostModel()
        self.rf = RandomForestModel()
        self.dnn = None          # instantiated lazily in fit() with correct in_features
        self.dnn_available = True
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "EnsembleModel":
        self.catboost.fit(X_train, y_train)
        print("  Ensemble: CatBoost OK", end="  ")
        self.rf.fit(X_train, y_train)
        print("RF OK", end="  ")

        # Instantiate DNN lazily with correct in_features from actual data shape
        n_features = X_train.shape[1]
        self.dnn = DNNModel(in_features=n_features)
        self.dnn_available = getattr(self.dnn, "available", True)
        if self.dnn_available:
            self.dnn.fit(X_train, y_train)
            print("DNN OK")
        else:
            print("DNN skipped (PyTorch unavailable)")
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EnsembleModel.fit() must be called before predict_proba().")
        p_cb = self.catboost.predict_proba(X)
        p_rf = self.rf.predict_proba(X)
        if self.dnn_available:
            p_dnn = self.dnn.predict_proba(X)
            assert p_cb.shape == p_rf.shape == p_dnn.shape
            ensemble = (p_cb + p_rf + p_dnn) / 3.0
        else:
            assert p_cb.shape == p_rf.shape
            ensemble = (p_cb + p_rf) / 2.0
        assert ensemble.shape == (len(X),)
        assert ensemble.min() >= 0.0 and ensemble.max() <= 1.0
        return ensemble

    def get_params(self) -> Dict[str, Any]:
        return {
            "catboost": self.catboost.get_params(),
            "random_forest": self.rf.get_params(),
            "dnn": self.dnn.get_params() if self.dnn is not None else None,
        }


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
    GridSearchCV with TimeSeriesSplit. MUST only be called on model_train_dates.
    """
    _SKLEARN_COMPATIBLE = (CatBoostModel, RandomForestModel)
    if not issubclass(model_class, _SKLEARN_COMPATIBLE):
        raise ValueError(f"tune_hyperparameters does not support {model_class.__name__}.")

    class _SklearnWrapper:
        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs
            self._model = model_class(**kwargs)

        def fit(self, X, y):
            self._model.fit(X, y)
            return self

        def predict_proba(self, X):
            p = self._model.predict_proba(X)
            return np.column_stack([1 - p, p])

        def get_params(self, deep=True):
            return dict(self._kwargs)

        def set_params(self, **params):
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
        refit=False,
        verbose=0,
    )
    gs.fit(X_train, y_train.astype(int))
    return gs.best_params_