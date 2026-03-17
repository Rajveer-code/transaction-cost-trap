"""
Temporal Sentiment Decay Analysis - Phase 5 (Day 33-35)
========================================================
Analyze how sentiment impact decays over time.

Analysis:
- Correlation between sentiment and returns at different lags (0-5 days)
- Exponential decay fitting
- Half-life calculation
- Optimal sentiment window identification
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class TemporalAnalyzer:
    """
    Analyze temporal decay of sentiment impact on stock returns.
    """
    
    def __init__(self):
        """Initialize temporal analyzer."""
        self.decay_results = {}
    
    def load_data(self, data_path: str = "data/final/model_ready_full.csv") -> pd.DataFrame:
        """
        Load final dataset with sentiment and returns.
        
        Args:
            data_path: Path to dataset
            
        Returns:
            DataFrame sorted by date
        """
        print("📂 Loading data...")
        df = pd.read_csv(data_path)
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        # ======================================================
        # FIX: Construct missing daily_return from EMA_12 proxy
        # ======================================================
        if 'daily_return' not in df.columns:
            if 'EMA_12' in df.columns:
                print("⚠️  'daily_return' not found — computing from EMA_12 as price proxy...")
                df['daily_return'] = df.groupby('ticker')['EMA_12'].pct_change() * 100
            else:
                raise ValueError("Dataset has no 'daily_return' or price column (Close/EMA_12). Cannot compute temporal decay.")

        print(f"   Loaded {len(df)} observations")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Tickers: {df['ticker'].nunique()}")
        
        return df
    
    def calculate_lagged_correlations(
        self,
        df: pd.DataFrame,
        sentiment_col: str = 'ensemble_sentiment_mean',
        return_col: str = 'daily_return',
        max_lag: int = 5
    ) -> pd.DataFrame:
        """
        Calculate correlation between sentiment and returns at different lags.
        
        Args:
            df: DataFrame with sentiment and returns
            sentiment_col: Name of sentiment column
            return_col: Name of return column
            max_lag: Maximum lag to analyze
            
        Returns:
            DataFrame with correlations and p-values
        """
        print(f"\n{'='*60}")
        print("CALCULATING LAGGED CORRELATIONS")
        print(f"{'='*60}\n")
        
        print(f"Sentiment column: {sentiment_col}")
        print(f"Return column: {return_col}")
        print(f"Max lag: {max_lag} days")
        
        correlations = []
        
        for lag in range(max_lag + 1):
            # Create lagged returns
            df_lag = df.copy()
            df_lag[f'return_lag{lag}'] = df_lag.groupby('ticker')[return_col].shift(-lag)
            
            # Remove NaN values
            df_clean = df_lag[[sentiment_col, f'return_lag{lag}']].dropna()
            
            if len(df_clean) > 0:
                # Calculate correlation
                corr, p_value = pearsonr(
                    df_clean[sentiment_col],
                    df_clean[f'return_lag{lag}']
                )
                
                significant = '✓' if p_value < 0.05 else ''
                
                correlations.append({
                    'lag': lag,
                    'correlation': corr,
                    'p_value': p_value,
                    'n_samples': len(df_clean),
                    'significant': significant
                })
                
                print(f"Lag {lag}: r={corr:+.4f}, p={p_value:.6f} {significant}")
            else:
                correlations.append({
                    'lag': lag,
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'n_samples': 0,
                    'significant': ''
                })
                print(f"Lag {lag}: Insufficient data")
        
        corr_df = pd.DataFrame(correlations)
        
        print(f"\n✅ Correlation analysis complete!")
        
        return corr_df
    
    def fit_exponential_decay(self, corr_df: pd.DataFrame) -> dict:
        """
        Fit exponential decay model to correlation data.
        
        Model: y = a * exp(-b * x) + c
        
        Args:
            corr_df: DataFrame with lag and correlation columns
            
        Returns:
            Dictionary with fit parameters and half-life
        """
        print(f"\n{'='*60}")
        print("FITTING EXPONENTIAL DECAY MODEL")
        print(f"{'='*60}\n")
        
        # Remove NaN values
        clean_data = corr_df.dropna()
        
        if len(clean_data) < 3:
            print("⚠️  Insufficient data for exponential fitting")
            return None
        
        lags = clean_data['lag'].values
        correlations = clean_data['correlation'].values
        
        # Only use positive correlations for decay fitting
        positive_mask = correlations > 0
        if positive_mask.sum() < 3:
            print("⚠️  Insufficient positive correlations for decay fitting")
            return None
        
        lags_pos = lags[positive_mask]
        corr_pos = correlations[positive_mask]
        
        # Define exponential decay function
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        try:
            # Fit curve
            initial_guess = [corr_pos[0], 0.3, 0]
            params, covariance = curve_fit(
                exp_decay,
                lags_pos,
                corr_pos,
                p0=initial_guess,
                maxfev=10000
            )
            
            a, b, c = params
            
            # Calculate half-life: t_half = ln(2) / b
            half_life = np.log(2) / b if b > 0 else np.inf
            
            # Calculate R-squared
            fitted_values = exp_decay(lags_pos, *params)
            ss_res = np.sum((corr_pos - fitted_values) ** 2)
            ss_tot = np.sum((corr_pos - np.mean(corr_pos)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            result = {
                'model': 'Exponential decay: y = a * exp(-b * x) + c',
                'a': float(a),
                'b': float(b),
                'c': float(c),
                'half_life_days': float(half_life),
                'r_squared': float(r_squared)
            }
            
            print(f"Model: y = {a:.4f} * exp(-{b:.4f} * x) + {c:.4f}")
            print(f"Half-life: {half_life:.2f} days")
            print(f"R²: {r_squared:.4f}")
            
            self.decay_results = result
            
            return result
            
        except Exception as e:
            print(f"⚠️  Exponential fitting failed: {str(e)}")
            return None
    
    def plot_decay_curve(
        self,
        corr_df: pd.DataFrame,
        decay_params: dict = None,
        output_dir: str = "results/figures/sentiment_analysis"
    ):
        """
        Plot sentiment decay curve with fitted model.
        
        Args:
            corr_df: DataFrame with correlation data
            decay_params: Exponential decay parameters (optional)
            output_dir: Directory to save plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📈 Creating decay curve plot...")
        
        plt.figure(figsize=(12, 7))
        
        # Plot actual correlations
        plt.scatter(
            corr_df['lag'],
            corr_df['correlation'],
            s=100,
            color='steelblue',
            edgecolor='black',
            linewidth=1.5,
            label='Observed Correlation',
            zorder=3
        )
        
        # Add line connecting points
        plt.plot(
            corr_df['lag'],
            corr_df['correlation'],
            color='steelblue',
            linewidth=1.5,
            alpha=0.5,
            zorder=2
        )
        
        # Plot fitted exponential decay if available
        if decay_params:
            a = decay_params['a']
            b = decay_params['b']
            c = decay_params['c']
            
            x_fit = np.linspace(0, corr_df['lag'].max(), 100)
            y_fit = a * np.exp(-b * x_fit) + c
            
            plt.plot(
                x_fit,
                y_fit,
                color='red',
                linewidth=2,
                linestyle='--',
                label=f'Exponential Fit (Half-life: {decay_params["half_life_days"]:.2f} days)',
                zorder=4
            )
            
            # Add half-life annotation
            half_life = decay_params['half_life_days']
            if 0 < half_life < corr_df['lag'].max():
                y_half = a * np.exp(-b * half_life) + c
                plt.axvline(
                    x=half_life,
                    color='red',
                    linestyle=':',
                    linewidth=1.5,
                    alpha=0.7,
                    label=f'Half-life: {half_life:.2f} days'
                )
                plt.scatter([half_life], [y_half], color='red', s=150, zorder=5, marker='*')
        
        # Add significance markers
        for _, row in corr_df.iterrows():
            if row['significant'] == '✓':
                plt.scatter(
                    row['lag'],
                    row['correlation'],
                    s=200,
                    facecolors='none',
                    edgecolors='green',
                    linewidth=2,
                    zorder=6
                )
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        plt.xlabel('Days After News (Lag)', fontsize=12)
        plt.ylabel('Correlation with Stock Return', fontsize=12)
        plt.title('Temporal Decay of Sentiment Impact', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = f"{output_dir}/sentiment_decay_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path}")
    
    def analyze_by_ticker(
        self,
        df: pd.DataFrame,
        sentiment_col: str = 'ensemble_sentiment_mean',
        return_col: str = 'daily_return',
        output_dir: str = "results/figures/sentiment_analysis"
    ):
        """
        Analyze sentiment decay separately for each ticker.
        
        Args:
            df: DataFrame with data
            sentiment_col: Sentiment column name
            return_col: Return column name
            output_dir: Output directory
        """
        print(f"\n{'='*60}")
        print("TICKER-LEVEL DECAY ANALYSIS")
        print(f"{'='*60}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        tickers = df['ticker'].unique()
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()
        
        ticker_results = []
        
        for idx, ticker in enumerate(tickers):
            df_ticker = df[df['ticker'] == ticker].copy()
            
            print(f"\nAnalyzing {ticker}...")
            
            # Calculate correlations for this ticker
            correlations = []
            for lag in range(6):
                df_ticker[f'return_lag{lag}'] = df_ticker[return_col].shift(-lag)
                df_clean = df_ticker[[sentiment_col, f'return_lag{lag}']].dropna()
                
                if len(df_clean) > 5:
                    corr, p_val = pearsonr(df_clean[sentiment_col], df_clean[f'return_lag{lag}'])
                    correlations.append({'lag': lag, 'correlation': corr, 'p_value': p_val})
                    print(f"  Lag {lag}: r={corr:+.4f}")
            
            corr_df_ticker = pd.DataFrame(correlations)
            
            # Plot for this ticker
            if len(corr_df_ticker) > 0:
                axes[idx].scatter(corr_df_ticker['lag'], corr_df_ticker['correlation'], s=80, color='steelblue')
                axes[idx].plot(corr_df_ticker['lag'], corr_df_ticker['correlation'], linewidth=1.5, color='steelblue', alpha=0.5)
                axes[idx].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
                axes[idx].set_title(ticker, fontweight='bold')
                axes[idx].set_xlabel('Lag (days)', fontsize=9)
                axes[idx].set_ylabel('Correlation', fontsize=9)
                axes[idx].grid(True, alpha=0.3)
                
                # Store max correlation
                max_corr = corr_df_ticker['correlation'].abs().max()
                max_lag = corr_df_ticker.loc[corr_df_ticker['correlation'].abs().idxmax(), 'lag']
                ticker_results.append({
                    'ticker': ticker,
                    'max_correlation': max_corr,
                    'optimal_lag': max_lag
                })
        
        # Hide unused subplots
        for idx in range(len(tickers), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Sentiment Decay by Ticker', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = f"{output_dir}/sentiment_decay_by_ticker.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Ticker analysis complete!")
        print(f"   Saved: {save_path}")
        
        # Save ticker results
        ticker_df = pd.DataFrame(ticker_results)
        ticker_csv_path = f"results/metrics/sentiment_decay_by_ticker.csv"
        ticker_df.to_csv(ticker_csv_path, index=False)
        print(f"   Saved: {ticker_csv_path}")
    
    def save_results(self, corr_df: pd.DataFrame, output_dir: str = "results/metrics"):
        """
        Save correlation and decay results.
        
        Args:
            corr_df: Correlation DataFrame
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving temporal analysis results...")
        
        # Save correlation data
        corr_path = f"{output_dir}/sentiment_decay.csv"
        corr_df.to_csv(corr_path, index=False)
        print(f"   ✅ Saved: {corr_path}")
        
        # Save decay parameters if available
        if self.decay_results:
            import json
            decay_path = f"{output_dir}/sentiment_decay_parameters.json"
            with open(decay_path, 'w') as f:
                json.dump(self.decay_results, f, indent=2)
            print(f"   ✅ Saved: {decay_path}")


def main():
    """
    Main execution for temporal analysis.
    """
    print("\n" + "="*60)
    print("PHASE 5 - DAY 33-35: TEMPORAL SENTIMENT DECAY ANALYSIS")
    print("="*60 + "\n")
    
    # Initialize analyzer
    analyzer = TemporalAnalyzer()
    
    # Load data
    df = analyzer.load_data()
    
    # Calculate lagged correlations
    corr_df = analyzer.calculate_lagged_correlations(df, max_lag=5)
    
    # Fit exponential decay
    decay_params = analyzer.fit_exponential_decay(corr_df)
    
    # Plot decay curve
    analyzer.plot_decay_curve(corr_df, decay_params)
    
    # Ticker-level analysis
    analyzer.analyze_by_ticker(df)
    
    # Save results
    analyzer.save_results(corr_df)
    
    print("\n" + "="*60)
    print("✅ TEMPORAL ANALYSIS COMPLETE!")
    print("="*60)
    print("\nOutputs saved:")
    print("  • results/figures/sentiment_analysis/sentiment_decay_curve.png")
    print("  • results/figures/sentiment_analysis/sentiment_decay_by_ticker.png")
    print("  • results/metrics/sentiment_decay.csv")
    print("  • results/metrics/sentiment_decay_parameters.json")
    print("  • results/metrics/sentiment_decay_by_ticker.csv")


if __name__ == "__main__":
    main()