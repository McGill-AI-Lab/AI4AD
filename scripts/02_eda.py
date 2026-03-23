"""
02_eda.py - Exploratory Data Analysis & Preprocessing Checks
============================================================
1. Checks for missing values (NaNs) across the 117 proteins.
2. Applies Median Imputation for missing values.
3. Applies Log2(x + 1) transformation to handle skewed biological distributions.
4. Runs a basic PCA to visualize the high-dimensional space.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from pathlib import Path

# Setup paths
DATA_DIR = Path("data/processed")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("--- 1. LOADING DATA ---")
    df = pd.read_csv(DATA_DIR / "adni_nulisa_cohort.csv")
    print(f"Loaded cohort with shape: {df.shape}")
    
    # Separate features (proteins) from labels (cohort_group)
    # The first column is 'RID', the last (or second) is 'cohort_group'
    # Everything else is a protein.
    meta_cols = ['RID', 'cohort_group']
    protein_cols = [c for c in df.columns if c not in meta_cols]
    
    X = df[protein_cols]
    y = df['cohort_group']
    
    print(f"\n--- 2. MISSINGNESS ANALYSIS ---")
    missing_pct = X.isna().mean() * 100
    print(f"Average missingness across all proteins: {missing_pct.mean():.2f}%")
    print(f"Max missingness for a single protein: {missing_pct.max():.2f}%")
    print(f"Min missingness for a single protein: {missing_pct.min():.2f}%")
    
    # Plot missingness histogram
    plt.figure(figsize=(8, 5))
    plt.hist(missing_pct, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Missing Data across 117 Proteins')
    plt.xlabel('Percentage of Missing Values (%)')
    plt.ylabel('Number of Proteins')
    plt.savefig(PLOT_DIR / 'missingness_histogram.png')
    plt.close()
    print("Saved missingness plot to plots/missingness_histogram.png")

    # print("\n--- 3. IMPUTATION & LOG TRANSFORMATION ---")
    # # Impute missing values with the median of each protein
    # imputer = SimpleImputer(strategy='median')
    # X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=protein_cols)
    
    # # Log2(x + 1) transform to fix skewed biological data
    # X_log = np.log2(X_imputed + 1)
    # print("Missing values imputed (median) and data log2-transformed.")

    print("\n--- 3. MISSINGNESS FILTER & IMPUTATION ---")
    # 1. Proposal Rule: Drop proteins missing in > 20% of participants
    threshold = 0.20
    proteins_to_drop = missing_pct[missing_pct > (threshold * 100)].index.tolist()
    
    print(f"Dropping {len(proteins_to_drop)} proteins missing in >20% of patients:")
    print(f" -> {proteins_to_drop}")
    
    X_filtered = X.drop(columns=proteins_to_drop)
    
    # 2. Impute the remainder with the median
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X_filtered), columns=X_filtered.columns)
    
    # 3. Log2(x + 1) transform to fix skewed biological data
    X_log = np.log2(X_imputed + 1)
    print(f"Remaining {X_filtered.shape[1]} proteins imputed and log2-transformed.")

    print("\n--- 4. PCA VISUALIZATION ---")
    # Run Principal Component Analysis (reduce 117 dimensions to 2)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_log)
    
    # Calculate how much variance the 2D plot explains
    var_exp = pca.explained_variance_ratio_ * 100
    
    # Plot PCA
    plt.figure(figsize=(8, 6))
    colors = {'CN': 'blue', 'MCI': 'orange'}
    
    for group in ['CN', 'MCI']:
        idx = (y == group)
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], 
                    label=group, alpha=0.6, s=30, c=colors[group])
        
    plt.title('PCA of Plasma Proteomics (Log-Transformed)')
    plt.xlabel(f'Principal Component 1 ({var_exp[0]:.1f}% variance)')
    plt.ylabel(f'Principal Component 2 ({var_exp[1]:.1f}% variance)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(PLOT_DIR / 'pca_plot.png')
    plt.close()
    print("Saved PCA plot to plots/pca_plot.png")
    
    print("\n✅ EDA script completed successfully!")

if __name__ == "__main__":
    main()

