"""
SHAP Explainer - Phase 5 (Day 29-30)
=====================================
Extended SHAP analysis building on Phase 4's foundation.

New additions:
- SHAP Dependence Plots (feature interactions)
- SHAP Waterfall Charts (individual predictions)
- Feature interaction analysis
- Detailed feature importance CSV
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    Extended SHAP explainability for CatBoost model.
    """
    
    def __init__(self, model_path: str = "models/catboost_best.pkl"):
        """
        Initialize SHAP explainer with trained model.
        
        Args:
            model_path: Path to saved CatBoost model
        """
        print(f"🔄 Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("✅ Model loaded successfully!")
        
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
    
    def prepare_data(self, data_path: str = "data/final/model_ready_full.csv"):
        """
        Load and prepare data for SHAP analysis.
        
        Args:
            data_path: Path to final dataset
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        print(f"\n📂 Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        df = df.sort_values('date').reset_index(drop=True)
        
        # Exclude non-features
        exclude_cols = ['date', 'ticker', 'movement']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['movement'].values
        
        print(f"   Samples: {len(X)}")
        print(f"   Features: {len(feature_cols)}")
        
        return X, y, feature_cols
    
    def compute_shap_values(self, X: np.ndarray, feature_names: list):
        """
        Compute SHAP values for all samples.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
        """
        print(f"\n{'='*60}")
        print("COMPUTING SHAP VALUES")
        print(f"{'='*60}\n")
        
        # Initialize TreeExplainer for CatBoost
        print("🔄 Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
        # Compute SHAP values
        print("🔄 Computing SHAP values (this may take a few minutes)...")
        self.shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output (CatBoost returns array for binary classification)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Use positive class
        
        self.feature_names = feature_names
        
        print(f"✅ SHAP values computed!")
        print(f"   Shape: {self.shap_values.shape}")
    
    def plot_summary(self, X, output_dir: str = "results/figures/shap_plots"):
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📊 Creating SHAP Summary Plot...")

        plt.figure(figsize=(12, 10))

        shap.summary_plot(
            self.shap_values,
            features=X,
            feature_names=self.feature_names,
            max_display=20,
            show=False
        )

        plt.tight_layout()
        save_path = f"{output_dir}/shap_summary_extended.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Saved: {save_path}")

    def plot_dependence(
        self,
        feature_name: str,
        interaction_feature: str = None,
        X: np.ndarray = None,
        output_dir: str = "results/figures/shap_plots"
    ):
        """
        Create SHAP dependence plot showing feature interactions.
        
        Args:
            feature_name: Main feature to analyze
            interaction_feature: Feature to color by (auto-detect if None)
            X: Feature matrix
            output_dir: Directory to save plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 Creating SHAP Dependence Plot: {feature_name}")
        
        feature_idx = self.feature_names.index(feature_name)
        
        plt.figure(figsize=(10, 6))
        
        if interaction_feature:
            interaction_idx = self.feature_names.index(interaction_feature)
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                X,
                feature_names=self.feature_names,
                interaction_index=interaction_idx,
                show=False
            )
        else:
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                X,
                feature_names=self.feature_names,
                show=False
            )
        
        plt.tight_layout()
        safe_name = feature_name.replace('/', '_').replace(' ', '_')
        save_path = f"{output_dir}/shap_dependence_{safe_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path}")
    
    def plot_waterfall(
        self,
        sample_idx: int,
        X: np.ndarray,
        output_dir: str = "results/figures/shap_plots"
    ):
        """
        Create SHAP waterfall plot for individual prediction.
        
        Args:
            sample_idx: Index of sample to explain
            X: Feature matrix
            output_dir: Directory to save plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 Creating SHAP Waterfall Plot: Sample {sample_idx}")
        
        # Create Explanation object for waterfall plot
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.explainer.expected_value,
            data=X[sample_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, show=False)
        
        plt.tight_layout()
        save_path = f"{output_dir}/shap_waterfall_sample{sample_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path}")
    
    def save_feature_importance(self, output_dir: str = "results/metrics"):
        """
        Save detailed feature importance to CSV.
        
        Args:
            output_dir: Directory to save CSV
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving feature importance...")
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Add rank
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        # Save
        save_path = f"{output_dir}/shap_feature_importance.csv"
        importance_df.to_csv(save_path, index=False)
        
        print(f"   ✅ Saved: {save_path}")
        print(f"\n📊 Top 10 Features by SHAP:")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    def analyze_feature_interactions(
        self,
        top_n: int = 5,
        X: np.ndarray = None,
        output_dir: str = "results/figures/shap_plots"
    ):
        """
        Analyze and plot interactions between top features.
        
        Args:
            top_n: Number of top features to analyze
            X: Feature matrix
            output_dir: Directory to save plots
        """
        print(f"\n{'='*60}")
        print("ANALYZING FEATURE INTERACTIONS")
        print(f"{'='*60}\n")
        
        # Get top features
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        top_features = [self.feature_names[i] for i in top_indices]
        
        print(f"Top {top_n} features:")
        for i, feat in enumerate(top_features, 1):
            print(f"  {i}. {feat}")
        
        # Create interaction heatmap
        interaction_matrix = np.zeros((top_n, top_n))
        
        for i, feat_i in enumerate(top_features):
            for j, feat_j in enumerate(top_features):
                if i != j:
                    idx_i = self.feature_names.index(feat_i)
                    idx_j = self.feature_names.index(feat_j)
                    
                    # Approximate interaction strength
                    interaction = np.corrcoef(
                        self.shap_values[:, idx_i],
                        self.shap_values[:, idx_j]
                    )[0, 1]
                    interaction_matrix[i, j] = interaction
        
        # Plot interaction heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            interaction_matrix,
            xticklabels=top_features,
            yticklabels=top_features,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            cbar_kws={'label': 'SHAP Correlation'}
        )
        plt.title('Feature Interaction Matrix (Top Features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = f"{output_dir}/shap_interaction_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Saved interaction heatmap: {save_path}")


def main():
    print("\n" + "="*60)
    print("PHASE 5 - DAY 29-30: EXTENDED SHAP ANALYSIS")
    print("="*60 + "\n")
    
    explainer = SHAPExplainer(model_path="models/catboost_best.pkl")
    
    X, y, feature_names = explainer.prepare_data()
    
    explainer.compute_shap_values(X, feature_names)
    
    # FIX: pass X here
    explainer.plot_summary(X)
    
    importance_df = explainer.save_feature_importance()
    
    top_features = importance_df.head(5)['feature'].tolist()
    
    for feature in top_features:
        explainer.plot_dependence(feature, X=X)
    
    explainer.plot_waterfall(0, X)
    explainer.plot_waterfall(len(X)//2, X)
    explainer.plot_waterfall(len(X)-1, X)

    explainer.analyze_feature_interactions(top_n=5, X=X)

    print("\n" + "="*60)
    print("✅ EXTENDED SHAP ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()