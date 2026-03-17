"""
Backtesting Engine - Phase 5 (Option A: Pure Inference)
=======================================================
- Uses a frozen CatBoost model (no retraining inside backtest)
- Applies model to full dataset once
- Trades on a single ticker (default: AAPL)
- Compares:
    1) ML Strategy (CatBoost, long-only, probability + direction)
    2) Buy & Hold

Outputs:
- results/predictions/df_pred_inference.csv
- results/figures/backtest_results/cumulative_returns.png
- results/metrics/backtest_metrics.csv
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")


class BacktestingEngine:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.002,   # 0.2%
        slippage: float = 0.0005,          # 0.05%
        confidence_threshold: float = 0.55,
        target_ticker: str = "AAPL"
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.confidence_threshold = confidence_threshold
        self.target_ticker = target_ticker

        self.model = None
        self.scaler = None

    # ------------------------------------------------------
    # 1. LOAD MODEL + SCALER + DATA
    # ------------------------------------------------------
    def load_model_and_data(self) -> pd.DataFrame:
        print("📂 Loading model, scaler, and data...")

        # Model (trained in Phase 4)
        with open("models/catboost_best.pkl", "rb") as f:
            self.model = pickle.load(f)
        print("   ✅ Loaded CatBoost model: models/catboost_best.pkl")

        # Scaler used for ensembles (includes CatBoost training scale)
        scaler_path = Path("models/scaler_ensemble.pkl")
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print("   ✅ Loaded scaler: models/scaler_ensemble.pkl")
        else:
            self.scaler = None
            print("   ⚠️  No scaler_ensemble.pkl found, will use raw features")

        # Data
        df = pd.read_csv("data/final/model_ready_full.csv")
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        # Ensure we have a price column
        if "Close" not in df.columns:
            if "EMA_12" in df.columns:
                df["Close"] = df["EMA_12"]
                print("   ⚠️  No 'Close' column found; using EMA_12 as proxy for Close.")
            else:
                raise ValueError("No 'Close' or 'EMA_12' column available for pricing.")

        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"   Date range: {df['date'].min()} → {df['date'].max()}")

        return df

    # ------------------------------------------------------
    # 2. PURE INFERENCE PREDICTIONS (NO RETRAINING)
    # ------------------------------------------------------
    def generate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("GENERATING PURE-INFERENCE PREDICTIONS (NO RETRAINING)")
        print("=" * 60 + "\n")

        # Restrict to one ticker (simplest clean backtest)
        df = df[df["ticker"] == self.target_ticker].copy().reset_index(drop=True)
        print(f"   Using ticker: {self.target_ticker}")
        print(f"   Samples for this ticker: {len(df)}")

        # Prepare features
        exclude_cols = ["date", "ticker", "movement", "Close"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].values

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Predict once, no CV, no refit
        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)  # direction; trading will add extra threshold

        df_pred = df.copy()
        df_pred["probability"] = probs
        df_pred["prediction"] = preds

        # Compute realized daily returns from Close (per ticker)
        df_pred["daily_return"] = df_pred["Close"].pct_change().fillna(0.0)

        print("\n📊 Prediction summary (inference only):")
        print(df_pred[["prediction", "probability"]].describe())

        # Save for inspection
        os.makedirs("results/predictions", exist_ok=True)
        out_path = "results/predictions/df_pred_inference.csv"
        df_pred.to_csv(out_path, index=False)
        print(f"\n💾 Saved inference predictions to {out_path}")

        return df_pred

    # ------------------------------------------------------
    # 3. SINGLE-TICKER LONG-ONLY STRATEGY
    # ------------------------------------------------------
    def calculate_returns(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        use_predictions: bool = True
    ) -> Tuple[pd.DataFrame, list]:
        """
        Simple long-only strategy on ONE ticker.
        - ML Strategy: long if prob >= threshold AND prediction == 1
        - Buy & Hold: buy on first day, hold to end
        """
        print(f"\n📊 Calculating returns: {strategy_name}")

        portfolio_values = []
        cash = self.initial_capital
        position = 0.0  # number of shares
        trades = []

        for i, row in df.iterrows():
            price = row["Close"]

            # Decide action
            if strategy_name == "Buy and Hold":
                if i == 0 and position == 0:
                    action = "buy"
                else:
                    action = "hold"

            elif use_predictions:
                # ML strategy based on prob + direction
                if (row["prediction"] == 1) and (row["probability"] >= self.confidence_threshold):
                    action = "buy" if position == 0 else "hold"
                else:
                    action = "sell" if position > 0 else "hold_cash"
            else:
                action = "hold_cash"

            # Execute action
            if action == "buy" and position == 0:
                # buy at today's close
                shares = cash / price
                cost = shares * price
                cost *= (1 + self.transaction_cost + self.slippage)

                position = shares
                cash -= cost

                trades.append({
                    "date": row["date"],
                    "action": "BUY",
                    "shares": shares,
                    "price": price,
                    "cash_after": cash
                })

            elif action == "sell" and position > 0:
                proceeds = position * price
                proceeds *= (1 - self.transaction_cost - self.slippage)
                cash += proceeds

                trades.append({
                    "date": row["date"],
                    "action": "SELL",
                    "shares": position,
                    "price": price,
                    "cash_after": cash
                })

                position = 0.0

            # Mark-to-market portfolio value
            if position > 0:
                pv = cash + position * price
            else:
                pv = cash

            portfolio_values.append(pv)

        results = pd.DataFrame({
            "date": df["date"],
            "portfolio_value": portfolio_values
        })

        print(f"   Final Value: ${portfolio_values[-1]:,.2f}")
        print(f"   Total Return: {(portfolio_values[-1] / self.initial_capital - 1) * 100:.2f}%")
        print(f"   Number of Trades: {len(trades)}")

        return results, trades

    # ------------------------------------------------------
    # 4. METRICS
    # ------------------------------------------------------
    def calculate_metrics(self, portfolio_values: pd.Series) -> Dict:
        returns = portfolio_values.pct_change().dropna()

        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100

        n_days = len(portfolio_values)
        years = n_days / 252 if n_days > 1 else 0
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = returns.mean() / downside.std() * np.sqrt(252)
        else:
            sortino = 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0.0

        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0.0
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "calmar_ratio": calmar
        }

    # ------------------------------------------------------
    # 5. RUN BACKTEST (ML vs BUY & HOLD)
    # ------------------------------------------------------
    def run_backtest(self, df_pred: pd.DataFrame) -> Dict:
        print("\n" + "=" * 60)
        print("RUNNING PORTFOLIO BACKTEST (SINGLE TICKER)")
        print("=" * 60 + "\n")

        results = {}

        # ML Strategy
        ml_results, ml_trades = self.calculate_returns(df_pred, "ML Strategy (CatBoost)", use_predictions=True)
        results["ML Strategy"] = {
            "portfolio_values": ml_results,
            "trades": ml_trades,
            "metrics": self.calculate_metrics(ml_results["portfolio_value"])
        }

        # Buy & Hold
        bh_results, bh_trades = self.calculate_returns(df_pred, "Buy and Hold", use_predictions=False)
        results["Buy and Hold"] = {
            "portfolio_values": bh_results,
            "trades": bh_trades,
            "metrics": self.calculate_metrics(bh_results["portfolio_value"])
        }

        return results

    # ------------------------------------------------------
    # 6. PLOTS + METRICS SAVING
    # ------------------------------------------------------
    def plot_cumulative_returns(self, results: Dict, output_dir: str = "results/figures/backtest_results"):
        os.makedirs(output_dir, exist_ok=True)

        print("\n📈 Plotting cumulative returns...")

        plt.figure(figsize=(12, 7))

        for name, data in results.items():
            pv = data["portfolio_values"]["portfolio_value"]
            dates = pd.to_datetime(data["portfolio_values"]["date"])
            cum_ret = (pv / self.initial_capital - 1) * 100
            plt.plot(dates, cum_ret, label=name, linewidth=2)

        plt.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (%)")
        plt.title(f"Cumulative Returns – {self.target_ticker}", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        path = f"{output_dir}/cumulative_returns_{self.target_ticker}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   ✅ Saved: {path}")

    def save_metrics(self, results: Dict, output_dir: str = "results/metrics"):
        os.makedirs(output_dir, exist_ok=True)

        rows = []
        for name, data in results.items():
            row = {"strategy": name}
            row.update(data["metrics"])
            rows.append(row)

        metrics_df = pd.DataFrame(rows)
        path = f"{output_dir}/backtest_metrics_{self.target_ticker}.csv"
        metrics_df.to_csv(path, index=False)

        print("\n💾 Saved backtest metrics to:", path)
        print("\n📊 Performance Comparison:")
        print(metrics_df.to_string(index=False))

        return metrics_df


def main():
    print("\n" + "=" * 60)
    print("PHASE 5 - PORTFOLIO BACKTESTING (PURE INFERENCE)")
    print("=" * 60 + "\n")

    engine = BacktestingEngine(
        initial_capital=100_000.0,
        transaction_cost=0.002,
        slippage=0.0005,
        confidence_threshold=0.55,
        target_ticker="AAPL"   # 👉 change here if you want another ticker
    )

    df = engine.load_model_and_data()
    df_pred = engine.generate_predictions(df)
    results = engine.run_backtest(df_pred)
    engine.plot_cumulative_returns(results)
    engine.save_metrics(results)

    print("\n" + "=" * 60)
    print("✅ BACKTEST (PURE INFERENCE) COMPLETE")
    print("=" * 60)
    print("\nOutputs:")
    print("  • results/predictions/df_pred_inference.csv")
    print("  • results/figures/backtest_results/cumulative_returns_<ticker>.png")
    print("  • results/metrics/backtest_metrics_<ticker>.csv")


if __name__ == "__main__":
    main()
