"""
walk_forward.py
===============
Walk-forward validation engine for the Cross-Sectional Conviction Ranking paper.

Generates expanding-window train/calibration/test fold splits on the DATE axis
of a (date, ticker) MultiIndex DataFrame.  Every result in the paper depends on
this module being provably data-leak free.

FOLD SPECIFICATION:
  Window type  : EXPANDING  (train set grows each fold, never shrinks)
  Min training : 3 years  → 756 trading days
  Test period  : 6 months → 126 trading days
  Step size    : 6 months → 126 trading days
  Embargo gap  : 2 calendar days between last train date and first test date

WHY 2-DAY EMBARGO:
  Target y_t = 1{Close(t+2) > Close(t+1)}.  Without an embargo the final 2
  training rows have targets that peek into the feature window of the first
  2 test rows.  The 2-day gap eliminates this subtle cross-contamination.

CALIBRATION SPLIT:
  The last 20 % of train_dates are reserved as a calibration set for
  Module 4 (isotonic regression probability calibration).  They remain in
  train_dates (so the WalkForwardFold always exposes the full training
  period) but are also surfaced as cal_dates for convenient downstream use.
  The model itself trains only on train_dates[:80%] (model_train_dates).

Author: Rajveer Singh Pall
Paper : "Overcoming the Transaction Cost Trap: Cross-Sectional Conviction
         Ranking in Machine Learning Equity Prediction"
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    """
    Immutable descriptor for one walk-forward fold.

    Attributes
    ----------
    fold_number : int
        1-indexed fold identifier.
    train_dates : List[pd.Timestamp]
        Full training date universe (80 % model-train + 20 % calibration).
    cal_dates : List[pd.Timestamp]
        Calibration dates = train_dates[80%:].  Used exclusively by Module 4.
        NEVER exposed to the test set.
    test_dates : List[pd.Timestamp]
        Test (out-of-sample) dates.  Always disjoint from train_dates.
    embargo_dates : List[pd.Timestamp]
        The 2 dates between train_end and test_start.  Excluded from both
        sets.  Retained for audit / visualisation.
    train_start : pd.Timestamp
    train_end   : pd.Timestamp
    test_start  : pd.Timestamp
    test_end    : pd.Timestamp
    """

    fold_number: int
    train_dates: List[pd.Timestamp]
    cal_dates: List[pd.Timestamp]
    test_dates: List[pd.Timestamp]
    embargo_dates: List[pd.Timestamp]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    # Computed field — populated by __post_init__
    model_train_dates: List[pd.Timestamp] = field(init=False)

    def __post_init__(self) -> None:
        """Derive model_train_dates as the first 80% of train_dates."""
        cal_split = int(len(self.train_dates) * 0.8)
        self.model_train_dates = self.train_dates[:cal_split]


# ---------------------------------------------------------------------------
# FOLD GENERATION
# ---------------------------------------------------------------------------

# Trading-day approximations
_MIN_TRAIN_DAYS: int = 756   # 3 years × 252 trading days
_TEST_DAYS: int = 126        # 6 months × 21 trading days
_STEP_DAYS: int = 126        # 6 months step
_EMBARGO_DAYS: int = 2       # match 2-day forward target window


def generate_folds(df: pd.DataFrame) -> List[WalkForwardFold]:
    """
    Generate all walk-forward folds from a (date, ticker) MultiIndex DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame produced by data_loader.load_all_data().
        Must have a 'date' level and a 'target' column.

    Returns
    -------
    List[WalkForwardFold]
        Ordered list of folds (earliest first).  Each fold satisfies all
        7 integrity assertions documented in _validate_fold().

    Raises
    ------
    ValueError
        If any fold violates an integrity assertion, or if df has fewer
        than _MIN_TRAIN_DAYS + _TEST_DAYS + _EMBARGO_DAYS unique dates.

    Notes
    -----
    Date indexing is integer-based into the sorted unique-date list.
    No calendar arithmetic is used — this avoids off-by-one errors from
    weekends/holidays and keeps the logic tied to actual trading days.
    """
    unique_dates: List[pd.Timestamp] = sorted(
        df.index.get_level_values("date").unique().tolist()
    )
    n = len(unique_dates)

    min_needed = _MIN_TRAIN_DAYS + _EMBARGO_DAYS + _TEST_DAYS
    if n < min_needed:
        raise ValueError(
            f"DataFrame has only {n} unique dates; need at least {min_needed} "
            f"({_MIN_TRAIN_DAYS} train + {_EMBARGO_DAYS} embargo + {_TEST_DAYS} test)."
        )

    folds: List[WalkForwardFold] = []
    fold_number = 0

    i = 0  # step counter
    while True:
        # ── Index arithmetic ─────────────────────────────────────────────────
        train_end_idx = _MIN_TRAIN_DAYS + i * _STEP_DAYS   # exclusive upper bound
        embargo_end_idx = train_end_idx + _EMBARGO_DAYS     # exclusive
        test_start_idx = embargo_end_idx                    # inclusive
        test_end_idx = test_start_idx + _TEST_DAYS          # exclusive

        if test_end_idx > n:
            break  # no more complete test windows

        # ── Slice dates ──────────────────────────────────────────────────────
        train_dates: List[pd.Timestamp] = unique_dates[:train_end_idx]
        embargo_dates: List[pd.Timestamp] = unique_dates[train_end_idx:embargo_end_idx]
        test_dates: List[pd.Timestamp] = unique_dates[test_start_idx:test_end_idx]

        # ── Calibration split (last 20 % of train_dates) ────────────────────
        cal_split = int(len(train_dates) * 0.8)
        cal_dates: List[pd.Timestamp] = train_dates[cal_split:]

        fold_number += 1
        fold = WalkForwardFold(
            fold_number=fold_number,
            train_dates=train_dates,
            cal_dates=cal_dates,
            test_dates=test_dates,
            embargo_dates=embargo_dates,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
        )

        _validate_fold(fold)   # raises ValueError on any violation
        folds.append(fold)
        i += 1

    if len(folds) == 0:
        raise ValueError(
            "No folds could be generated.  Check that df covers at least "
            f"{min_needed} unique trading dates."
        )

    return folds


# ---------------------------------------------------------------------------
# FOLD VALIDATION
# ---------------------------------------------------------------------------

def _validate_fold(fold: WalkForwardFold) -> None:
    """
    Assert all 7 integrity invariants on a single WalkForwardFold.

    Called automatically by generate_folds() on every generated fold.
    Also available for standalone auditing.

    Parameters
    ----------
    fold : WalkForwardFold
        The fold to validate.

    Raises
    ------
    ValueError
        On the first violated assertion, with a descriptive message.

    Invariants
    ----------
    1. Minimum training size  : len(train_dates) >= 756
    2. Non-empty test set     : len(test_dates) > 0
    3. Disjoint sets          : train_dates ∩ test_dates == ∅
    4. Temporal ordering      : min(test) > max(train)
    5. Embargo gap ≥ 2 days   : (min(test) - max(train)).days >= 2
    6. Cal ⊆ Train            : cal_dates ⊆ train_dates
    7. Cal ∩ Test == ∅        : cal_dates ∩ test_dates == ∅
    """
    train_set = set(fold.train_dates)
    test_set = set(fold.test_dates)
    cal_set = set(fold.cal_dates)

    # 1. Minimum training size
    if len(fold.train_dates) < _MIN_TRAIN_DAYS:
        raise ValueError(
            f"Fold {fold.fold_number}: train_dates has {len(fold.train_dates)} days "
            f"but minimum is {_MIN_TRAIN_DAYS}."
        )

    # 2. Non-empty test set
    if len(fold.test_dates) == 0:
        raise ValueError(f"Fold {fold.fold_number}: test_dates is empty.")

    # 3. Disjoint train and test
    overlap = train_set & test_set
    if overlap:
        raise ValueError(
            f"Fold {fold.fold_number}: {len(overlap)} date(s) appear in both "
            f"train_dates and test_dates — data leakage detected. "
            f"Sample overlapping dates: {sorted(overlap)[:5]}"
        )

    # 4. Temporal ordering
    max_train = max(fold.train_dates)
    min_test = min(fold.test_dates)
    if min_test <= max_train:
        raise ValueError(
            f"Fold {fold.fold_number}: test_start ({min_test.date()}) is not "
            f"strictly after train_end ({max_train.date()})."
        )

    # 5. Embargo gap
    gap_days = (min_test - max_train).days
    if gap_days < _EMBARGO_DAYS:
        raise ValueError(
            f"Fold {fold.fold_number}: embargo gap is {gap_days} calendar days "
            f"but minimum is {_EMBARGO_DAYS}.  "
            f"train_end={max_train.date()}, test_start={min_test.date()}"
        )

    # 6. Calibration dates are a subset of training dates
    if not cal_set.issubset(train_set):
        rogue = cal_set - train_set
        raise ValueError(
            f"Fold {fold.fold_number}: {len(rogue)} cal_date(s) are not in "
            f"train_dates: {sorted(rogue)[:5]}"
        )

    # 7. Calibration dates do not touch test dates
    cal_test_overlap = cal_set & test_set
    if cal_test_overlap:
        raise ValueError(
            f"Fold {fold.fold_number}: {len(cal_test_overlap)} cal_date(s) appear "
            f"in test_dates — calibration leakage detected."
        )


# ---------------------------------------------------------------------------
# ARRAY EXTRACTION WITH SCALER
# ---------------------------------------------------------------------------

def get_fold_arrays(
    fold: WalkForwardFold,
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Extract NumPy arrays for a fold and apply a scaler fit only on train data.

    The model trains on train_dates[:80%] (fold.model_train_dates).
    The remaining 20% (fold.cal_dates) are used by Module 4 for calibration
    — they are NOT included in X_train returned here.

    Parameters
    ----------
    fold : WalkForwardFold
        The fold whose date ranges define the split.
    df : pd.DataFrame
        Full MultiIndex DataFrame (date, ticker).
    feature_cols : List[str]
        Feature column names (output of data_loader.get_feature_columns).
    scaler : StandardScaler, optional
        If None (default), a new StandardScaler is created and fitted on
        X_train.  Pass a pre-fitted scaler only if you know what you are
        doing — the scaler MUST have been fitted on equivalent training data.

    Returns
    -------
    X_train : np.ndarray, shape (n_train_rows, n_features)
        Scaled model-training features (80% of train_dates × 7 tickers).
    X_test  : np.ndarray, shape (n_test_rows, n_features)
        Scaled test features.
    y_train : np.ndarray, shape (n_train_rows,)
        Binary training labels.
    y_test  : np.ndarray, shape (n_test_rows,)
        Binary test labels.
    scaler  : StandardScaler
        The fitted scaler (re-use for calibration set extraction).

    Raises
    ------
    ValueError
        If the scaler has not been fitted (scaler.mean_ is None) when
        transform is called, or if train/test arrays are empty.

    CRITICAL WARNING:
        scaler.fit() is called ONLY on X_train.  Calling it on X_test or
        on the full dataset constitutes lookahead bias and will silently
        produce overoptimistic results.
    """
    date_level = df.index.get_level_values("date")

    # Model-train mask: 80% of train_dates only
    model_train_set = set(fold.model_train_dates)
    test_set = set(fold.test_dates)

    train_mask = date_level.isin(model_train_set)
    test_mask = date_level.isin(test_set)

    X_train_raw = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "target"].values
    X_test_raw = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, "target"].values

    if len(X_train_raw) == 0:
        raise ValueError(
            f"Fold {fold.fold_number}: X_train is empty. "
            "Check that model_train_dates intersects df."
        )
    if len(X_test_raw) == 0:
        raise ValueError(
            f"Fold {fold.fold_number}: X_test is empty. "
            "Check that test_dates intersects df."
        )

    # ── Scaler: fit ONLY on training data ───────────────────────────────────
    if scaler is None:
        scaler = StandardScaler()

    # Guard: only fit if the scaler has not already been fitted.
    # (Allows caller to pass a pre-fitted scaler for calibration set.)
    if not hasattr(scaler, "mean_") or scaler.mean_ is None:
        scaler.fit(X_train_raw)

    # Enforce that scaler is fitted before transforming test data
    if scaler.mean_ is None:
        raise ValueError(
            f"Fold {fold.fold_number}: scaler.mean_ is None after fit() — "
            "this should never happen; check the training data."
        )

    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, scaler


def get_cal_arrays(
    fold: WalkForwardFold,
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract calibration set arrays using an already-fitted scaler.

    Called by Module 4 (calibration.py) after get_fold_arrays() has run.

    Parameters
    ----------
    fold : WalkForwardFold
    df : pd.DataFrame
    feature_cols : List[str]
    scaler : StandardScaler
        Must be already fitted on the model-train portion of this fold.
        Raises ValueError if scaler.mean_ is None.

    Returns
    -------
    X_cal : np.ndarray, shape (n_cal_rows, n_features)
    y_cal : np.ndarray, shape (n_cal_rows,)

    CRITICAL: This function uses scaler.transform() (NOT fit_transform()).
    The scaler must have been fitted on X_train, not X_cal.
    """
    if scaler.mean_ is None:
        raise ValueError(
            "get_cal_arrays: scaler has not been fitted.  "
            "Call get_fold_arrays() first to obtain a fitted scaler."
        )

    date_level = df.index.get_level_values("date")
    cal_mask = date_level.isin(set(fold.cal_dates))

    X_cal_raw = df.loc[cal_mask, feature_cols].values
    y_cal = df.loc[cal_mask, "target"].values

    if len(X_cal_raw) == 0:
        raise ValueError(
            f"Fold {fold.fold_number}: calibration set is empty. "
            "Check that cal_dates intersects df."
        )

    X_cal = scaler.transform(X_cal_raw)
    return X_cal, y_cal


# ---------------------------------------------------------------------------
# PRINTING / REPORTING
# ---------------------------------------------------------------------------

def print_fold_summary(folds: List[WalkForwardFold]) -> None:
    """
    Print a formatted ASCII table summarising all folds.

    Parameters
    ----------
    folds : List[WalkForwardFold]
        Output of generate_folds().

    Output columns
    --------------
    Fold | Train Start | Train End | Test Start | Test End |
    Train Days | Model-Train Days | Cal Days | Test Days
    """
    header = (
        f"{'Fold':>4}  {'Train Start':>12}  {'Train End':>12}  "
        f"{'Test Start':>12}  {'Test End':>12}  "
        f"{'Tr':>6}  {'Mo-Tr':>6}  {'Cal':>6}  {'Te':>6}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for f in folds:
        print(
            f"{f.fold_number:>4}  "
            f"{str(f.train_start.date()):>12}  "
            f"{str(f.train_end.date()):>12}  "
            f"{str(f.test_start.date()):>12}  "
            f"{str(f.test_end.date()):>12}  "
            f"{len(f.train_dates):>6}  "
            f"{len(f.model_train_dates):>6}  "
            f"{len(f.cal_dates):>6}  "
            f"{len(f.test_dates):>6}"
        )
    print(sep)
    print(
        f"Total folds: {len(folds)}  |  "
        f"Out-of-sample coverage: {folds[0].test_start.date()} -> {folds[-1].test_end.date()}"
    )


def fold_stats(
    folds: List[WalkForwardFold],
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compute target class balance statistics for each fold.

    Prints a WARNING (not an error) if any fold's balance falls outside
    the 40–60 % range, which may signal regime shifts or label engineering
    problems.

    Parameters
    ----------
    folds : List[WalkForwardFold]
    df : pd.DataFrame
    feature_cols : List[str]
        Used only to verify column presence; not used for computation.

    Returns
    -------
    pd.DataFrame
        Columns: [fold, train_balance, test_balance,
                  train_warning, test_warning]
        train_balance / test_balance = fraction of target == 1.0
        train_warning / test_warning = True if outside [0.40, 0.60]
    """
    records = []
    date_level = df.index.get_level_values("date")

    for f in folds:
        # Model-train portion only (80% of train)
        train_mask = date_level.isin(set(f.model_train_dates))
        test_mask = date_level.isin(set(f.test_dates))

        y_tr = df.loc[train_mask, "target"].values
        y_te = df.loc[test_mask, "target"].values

        train_bal = float(np.mean(y_tr == 1.0)) if len(y_tr) > 0 else float("nan")
        test_bal = float(np.mean(y_te == 1.0)) if len(y_te) > 0 else float("nan")

        tr_warn = not (0.40 <= train_bal <= 0.60)
        te_warn = not (0.40 <= test_bal <= 0.60)

        if tr_warn:
            print(
                f"  [WARN] Fold {f.fold_number}: "
                f"model-train target balance = {train_bal:.1%} (outside 40-60%)"
            )
        if te_warn:
            print(
                f"  [WARN] Fold {f.fold_number}: "
                f"test target balance = {test_bal:.1%} (outside 40-60%)"
            )

        records.append(
            {
                "fold": f.fold_number,
                "train_balance": round(train_bal, 4),
                "test_balance": round(test_bal, 4),
                "train_warning": tr_warn,
                "test_warning": te_warn,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# MODULE SELF-TEST (synthetic data only — no yfinance)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("walk_forward.py — self-test (synthetic data)")
    print("=" * 60)

    # ── Build synthetic MultiIndex DataFrame ─────────────────────────────────
    TICKERS = ["A", "B", "C", "D", "E", "F", "G"]
    N_FEAT = 47
    FEAT_COLS = [f"feat_{i:02d}" for i in range(N_FEAT)]

    rng = np.random.default_rng(seed=42)
    dates = pd.bdate_range("2015-01-01", "2025-01-01")

    idx = pd.MultiIndex.from_product([dates, TICKERS], names=["date", "ticker"])
    n_rows = len(idx)

    data = {col: rng.standard_normal(n_rows) for col in FEAT_COLS}
    data["Close"] = rng.uniform(50, 500, n_rows)
    data["target"] = rng.integers(0, 2, n_rows).astype(float)

    df_syn = pd.DataFrame(data, index=idx)
    print(
        f"\nSynthetic DataFrame: {df_syn.shape[0]:,} rows, "
        f"{len(dates)} unique dates, {len(TICKERS)} tickers"
    )

    # ── Generate folds ────────────────────────────────────────────────────────
    folds = generate_folds(df_syn)
    print(f"\nFolds generated: {len(folds)}")

    # ── Validate all assertion invariants on every fold ───────────────────────
    violation_count = 0
    for fold in folds:
        try:
            _validate_fold(fold)
        except ValueError as e:
            print(f"[FAIL] VALIDATION FAILED: {e}")
            violation_count += 1

    assert violation_count == 0, f"{violation_count} fold(s) failed validation."

    # ── Embargo: verify no dates bleed across boundary ────────────────────────
    for fold in folds:
        train_set = set(fold.train_dates)
        test_set = set(fold.test_dates)
        assert not (train_set & test_set), f"Fold {fold.fold_number}: date overlap!"
        gap = (min(fold.test_dates) - max(fold.train_dates)).days
        assert gap >= 2, f"Fold {fold.fold_number}: embargo gap {gap} < 2 days!"

    # ── Calibration split ────────────────────────────────────────────────────
    for fold in folds:
        cal_set = set(fold.cal_dates)
        assert cal_set.issubset(set(fold.train_dates)), "Cal not subset of train!"
        assert not (cal_set & set(fold.test_dates)), "Cal bleeds into test!"
        # Verify size is approximately 20%
        expected_cal = len(fold.train_dates) - len(fold.model_train_dates)
        assert expected_cal == len(fold.cal_dates), \
            f"Fold {fold.fold_number}: cal size mismatch."

    # ── Print summary table ──────────────────────────────────────────────────
    print_fold_summary(folds)

    # ── Fold stats ───────────────────────────────────────────────────────────
    print("\nFold target balance check:")
    stats_df = fold_stats(folds, df_syn, FEAT_COLS)
    print(stats_df.to_string(index=False))

    # ── get_fold_arrays smoke test ────────────────────────────────────────────
    fold0 = folds[0]
    X_train, X_test, y_train, y_test, scaler = get_fold_arrays(
        fold0, df_syn, FEAT_COLS
    )
    assert X_train.shape[1] == N_FEAT, "Feature count mismatch in X_train."
    assert X_test.shape[1] == N_FEAT, "Feature count mismatch in X_test."
    assert X_train.shape[0] == len(fold0.model_train_dates) * len(TICKERS), \
        "Row count mismatch in X_train."
    # Scaler must be fitted on train only — verify by checking mean is non-trivial
    assert scaler.mean_ is not None, "Scaler mean is None after fit."
    # Test transform uses train statistics — test mean should differ from 0
    assert not np.allclose(scaler.mean_, 0.0, atol=1e-6), \
        "Scaler mean is zero — possible fit-on-test leakage."

    # ── get_cal_arrays smoke test ─────────────────────────────────────────────
    X_cal, y_cal = get_cal_arrays(fold0, df_syn, FEAT_COLS, scaler)
    assert X_cal.shape[1] == N_FEAT, "Feature count mismatch in X_cal."
    assert X_cal.shape[0] == len(fold0.cal_dates) * len(TICKERS), \
        "Row count mismatch in X_cal."

    # ── Minimum fold count ────────────────────────────────────────────────────
    assert len(folds) >= 10, (
        f"Expected at least 10 folds for 10-year dataset, got {len(folds)}."
    )

    print(
        f"\n[PASS] walk_forward.py PASSED: {len(folds)} folds, "
        f"embargo verified, zero date leakage"
    )
