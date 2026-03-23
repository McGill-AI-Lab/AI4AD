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
APOE_PATH = Path("data/raw/APOERES_20Feb2026.csv")
COHORT_LABELS = Path("data/processed/adni_nulisa_cohort.csv")

def get_clinical_genetics():
    # 1. Master Patient List
    master = pd.read_csv(COHORT_LABELS)[['RID', 'cohort_group']]
    master['RID'] = pd.to_numeric(master['RID'], errors='coerce')

    # 2. MMSE (Cognitive)
    mmse_df = pd.read_csv(MMSE_PATH, low_memory=False)
    mmse_df['RID'] = pd.to_numeric(mmse_df['RID'], errors='coerce')
    mask = mmse_df['VISCODE'].str.strip().str.lower().isin(['bl', 'sc'])
    mmse_bl = mmse_df[mask].groupby('RID').first().reset_index()[['RID', 'MMSCORE']]
    
    # 3. Demographics (Functional)
    demog = pd.read_csv(DEMOG_PATH, low_memory=False)[['RID', 'PTEDUCAT', 'PTGENDER']]
    demog['RID'] = pd.to_numeric(demog['RID'], errors='coerce')
    demog['is_male'] = demog['PTGENDER'].map({1: 1, 2: 0})

    # 4. Genetics (APOE4) - Updated for 'GENOTYPE' column
    apoe_df = pd.read_csv(APOE_PATH, low_memory=False)
    apoe_df['RID'] = pd.to_numeric(apoe_df['RID'], errors='coerce')
    
    # Based on your file, we use the 'GENOTYPE' column and count '4's
    # "3/4" -> 1, "4/4" -> 2, "2/3" -> 0
    apoe_df['APOE4'] = (
        apoe_df['GENOTYPE']
        .astype(str)
        .str.count('4')
        .fillna(0)
    )
    
    # Get the baseline/first entry
    apoe_bl = apoe_df.groupby('RID').first().reset_index()[['RID', 'APOE4']]

    # Final Merge
    clinical = master.merge(mmse_bl, on='RID', how='left')
    clinical = clinical.merge(demog[['RID', 'PTEDUCAT', 'is_male']], on='RID', how='left')
    clinical = clinical.merge(apoe_bl, on='RID', how='left')
    
    return clinical

def get_proteomics_data():
    df = pd.read_csv(NULISA_PATH, low_memory=False)
    df['RID'] = pd.to_numeric(df['RID'], errors='coerce')
    df_bl = df[df['VISCODE'].str.strip().str.lower() == 'bl'].copy()
    pivot = df_bl.pivot_table(index='RID', columns='Target', values='NPQ', aggfunc='median')
    pivot = np.log1p(pivot).reset_index()
    return pivot

def main():
    print("🚀 Merging Clinical, Genetic, and Proteomic data...")
    clinical = get_clinical_genetics()
    proteins = get_proteomics_data()
    data = clinical.merge(proteins, on='RID', how='inner')
    
    data['cohort_group'] = data['cohort_group'].str.strip().map({'CN': 0, 'MCI': 1})
    data = data.dropna(subset=['cohort_group'])
    
    X = data.drop(columns=['RID', 'cohort_group'])
    y = data['cohort_group'].astype(int)
    X = X.dropna(axis=1, how='all')
    
    print(f"✅ Final Dataset: {len(data)} patients | {X.shape[1]} features")

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(penalty='l1', solver='liblinear', C=0.2, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')

    print(f"\n--- MODEL 06: FULL MULTIMODAL RESULT ---")
    print(f"Mean AUC-ROC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    pipe.fit(X, y)
    coefs = pd.Series(pipe.named_steps['model'].coef_[0], index=X.columns)
    print("\n🏆 Top 15 Final Predictors:")
    print(coefs.abs().sort_values(ascending=False).head(15))

if __name__ == "__main__":
    main()