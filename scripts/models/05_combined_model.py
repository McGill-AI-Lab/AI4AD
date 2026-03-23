import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# --- RAW PATHS ---
NULISA_PATH = Path("data/raw/BSHRI_PLA_CSF_NULISA_CNS_16Feb2026.csv")
MMSE_PATH = Path("data/raw/MMSE_16Feb2026.csv")
DEMOG_PATH = Path("data/raw/PTDEMOG_16Feb2026.csv")
# This is the file you showed me that has the RID and cohort_group
COHORT_LABELS = Path("data/processed/adni_nulisa_cohort.csv")


def get_clinical_data():
    master = pd.read_csv(COHORT_LABELS)[['RID', 'cohort_group']]
    master['RID'] = pd.to_numeric(master['RID'], errors='coerce')

    # Fix: Get MMSE and filter for Baseline codes
    mmse_df = pd.read_csv(MMSE_PATH, low_memory=False)
    mmse_df['RID'] = pd.to_numeric(mmse_df['RID'], errors='coerce')
    
    # ADNI baseline is typically 'bl' or 'sc' (screening)
    # We filter for these codes and take the first one found for each RID
    mask = mmse_df['VISCODE'].str.strip().str.lower().isin(['bl', 'sc'])
    mmse_bl = mmse_df[mask].groupby('RID').first().reset_index()
    mmse_bl = mmse_bl[['RID', 'MMSCORE']]
    
    # Demographics (PTGENDER: 1=Male, 2=Female)
    demog = pd.read_csv(DEMOG_PATH, low_memory=False)[['RID', 'PTEDUCAT', 'PTGENDER']]
    demog['RID'] = pd.to_numeric(demog['RID'], errors='coerce')
    demog['is_male'] = demog['PTGENDER'].map({1: 1, 2: 0})

    # Merge clinical features onto the master label list
    clinical = master.merge(mmse_bl, on='RID', how='left')
    clinical = clinical.merge(demog[['RID', 'PTEDUCAT', 'is_male']], on='RID', how='left')
    
    return clinical

def get_proteomics_data():
    df = pd.read_csv(NULISA_PATH, low_memory=False)
    df['RID'] = pd.to_numeric(df['RID'], errors='coerce')
    df_bl = df[df['VISCODE'].str.strip().str.lower() == 'bl'].copy()
    
    # Pivot to wide format
    pivot = df_bl.pivot_table(index='RID', columns='Target', values='NPQ', aggfunc='median')
    pivot = np.log1p(pivot).reset_index()
    return pivot

def main():
    print("🚀 Merging raw clinical files with proteomics...")
    clinical = get_clinical_data()
    proteins = get_proteomics_data()
    data = clinical.merge(proteins, on='RID', how='inner')
    
    data['cohort_group'] = data['cohort_group'].str.strip().map({'CN': 0, 'MCI': 1})
    data = data.dropna(subset=['cohort_group'])
    
    X = data.drop(columns=['RID', 'cohort_group'])
    y = data['cohort_group'].astype(int)
    
    # DROP columns that are 100% NaN before training to avoid the alignment error
    X = X.dropna(axis=1, how='all')
    
    print(f"✅ Combined Dataset: {len(data)} patients | {X.shape[1]} features")

    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(penalty='l1', solver='liblinear', C=0.2, random_state=42))
    ])

    # Run CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')

    print(f"\n--- Model 3: Combined Result ---")
    print(f"Mean AUC-ROC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Final Fit & Coefficient alignment fix
    pipe.fit(X, y)
    
    # The fix: Ensure index matches the features the model actually used
    model_features = X.columns
    coefs = pd.Series(pipe.named_steps['model'].coef_[0], index=model_features)
    
    print("\nTop 15 Features by Absolute Impact:")
    print(coefs.abs().sort_values(ascending=False).head(15))

if __name__ == "__main__":
    main()