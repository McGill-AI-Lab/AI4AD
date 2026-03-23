"""
scripts/04_protein_model.py

MODEL 2: Proteomics-Only Classification (CN vs MCI)
--------------------------------------------------
Goal: Evaluate the predictive power of 113 NULISA plasma proteins.
Benchmark: Clinical Baseline (Model 1) AUC = 0.7935.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# --- Path Configuration ---
NULISA_RAW_PATH = Path("data/raw/BSHRI_PLA_CSF_NULISA_CNS_16Feb2026.csv")
CLINICAL_LABELS_PATH = Path("data/processed/adni_nulisa_cohort.csv")

def main():
    # --- 1. DATA INGESTION & PIVOTING ---
    # Raw NULISA data is in 'long' format (one protein per row).
    # We must pivot it to 'wide' format (one patient per row) for ML.
    df_raw = pd.read_csv(NULISA_RAW_PATH, low_memory=False)
    
    # Standardize RIDs for consistent merging
    df_raw['RID'] = pd.to_numeric(df_raw['RID'], errors='coerce')
    df_raw = df_raw.dropna(subset=['RID'])
    
    # Filter for baseline visits only
    df_bl = df_raw[df_raw['VISCODE'].str.strip().str.lower() == 'bl'].copy()
    
    # Pivot logic: Index is patient (RID), columns are proteins (Target), values are concentrations (NPQ)
    pivot_df = df_bl.pivot_table(
        index='RID', 
        columns='Target', 
        values='NPQ', 
        aggfunc='median'
    )
    
    # --- 2. LOG NORMALIZATION ---
    # Proteomic concentrations are typically skewed. log1p transformation 
    # normalizes the distribution and stabilizes variance.
    pivot_df = np.log1p(pivot_df).reset_index()
    pivot_df.columns.name = None # Remove the 'Target' name from the index
    
    # --- 3. COHORT ALIGNMENT ---
    # Merge pivoted proteins with clinical diagnoses (CN vs MCI)
    labels = pd.read_csv(CLINICAL_LABELS_PATH)
    labels['RID'] = pd.to_numeric(labels['RID'], errors='coerce')
    
    # Inner merge ensures we only analyze patients with both blood data and labels
    final_df = labels[['RID', 'cohort_group']].merge(pivot_df, on='RID', how='inner')
    
    # Convert categorical labels to binary (0=CN, 1=MCI)
    final_df['cohort_group'] = final_df['cohort_group'].str.strip().map({'CN': 0, 'MCI': 1})
    final_df = final_df.dropna(subset=['cohort_group'])
    
    # --- 4. FEATURE SELECTION & ML PIPELINE ---
    protein_cols = [c for c in pivot_df.columns if c != 'RID']
    X = final_df[protein_cols]
    y = final_df['cohort_group'].astype(int)
    
    print(f"Dataset summary: {len(final_df)} patients | {len(protein_cols)} proteins")

    # Pipeline Design:
    # - SimpleImputer: Handles any missing protein measurements using the median.
    # - StandardScaler: Rescales features to mean=0, variance=1 for Logistic Regression.
    # - L1 Penalty (Lasso): Performs automatic feature selection by zeroing out 
    #   non-predictive protein coefficients.
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            C=0.5, 
            random_state=42
        ))
    ])

    # --- 5. CROSS-VALIDATED EVALUATION ---
    # Using 5-Fold Stratified CV to maintain class balance in each fold.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')

    print(f"\n--- MODEL 2 RESULTS: PROTEOMICS ONLY ---")
    print(f"Mean AUC-ROC: {auc_scores.mean():.4f}")
    print(f"Std AUC-ROC:  {auc_scores.std():.4f}")

    # --- 6. TOP FEATURE EXTRACTION ---
    # Fit on the full dataset to identify the most significant biomarkers.
    pipeline.fit(X, y)
    coefficients = pd.Series(pipeline.named_steps['model'].coef_[0], index=protein_cols)
    
    print("\nTop 10 Predictive Biomarkers (Ranked by Absolute Weight):")
    top_10 = coefficients.abs().sort_values(ascending=False).head(10)
    for name, weight in top_10.items():
        actual_coef = coefficients[name]
        direction = "Positive (Risk)" if actual_coef > 0 else "Negative (Protective)"
        print(f"{name:<25} | Coef: {actual_coef:>8.4f} | {direction}")

if __name__ == "__main__":
    main()