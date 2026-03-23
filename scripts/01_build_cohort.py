"""
ADNI NULISA Plasma Proteomics + Baseline Diagnosis Cohort Builder
===============================================================

What this script produces 
-------------------------
A single "machine learning table" saved to:
  data/processed/adni_nulisa_cohort.csv

Rows: One row per patient (RID) at their baseline visit.
Columns:
  - cohort_group: CN or MCI (baseline)
  - Protein features (wide matrix format: one column per protein)
  - Bookkeeping columns

Conceptual (from COMP 551 ML notes)
------------------------------------------------------
1) Empirical loss vs. generalization error:
   - In ML we can minimize training/empirical loss, but what we care about is
     expected (test) loss on new samples from the true distribution p(x,y),
     i.e. the generalization error.
   - This script is building a clean dataset so your future model training can
     focus on improving generalization, not just "fitting messy joins".

2) Bias–variance + overfitting:
   - Complex feature sets (thousands of proteins) can lead to low training error
     but high test error (overfitting); that's the bias–variance tradeoff picture.
   - Later, you'll likely use regularization to reduce variance by penalizing
     weights (ridge/lasso), which is exactly "add a penalty term to the loss". 

3) Hyperparameters / model selection:
   - Choices like the matching window (180 days) are hyperparameters.
     In the notes, hyperparameters are chosen using validation/cross-validation
     to get the best chance of low generalization error.
   - For now we enforce an exact baseline match ('bl' or 'sc' viscodes) to guarantee
     data alignment, but proximity matching is an alternative approach.

Assumptions / rules applied
---------------------------
- Proteomics file is "long" format. We pivot it to "wide" format.
- We filter SampleMatrixType strictly to 'Plasma'.
- Diagnosis baseline uses the harmonized 'DIAGNOSIS' field where 1=CN, 2=MCI.
"""

import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
# Set up relative paths so this script works on anyone's machine who clones the repo
DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "prot": "BSHRI_PLA_CSF_NULISA_CNS_16Feb2026.csv",
    "dx": "DXSUM_16Feb2026.csv"
}

# Hyperparameter: We drop proteins that are mostly undetected to reduce noise
DETECT_THRESHOLD = 0.50 

# 'bl' is ADNI's standard baseline visit. 'sc' (screening) is sometimes used in ADNI2/GO
BASELINE_VISCODES = {"bl", "sc"}


def process_proteomics():
    """
    Loads raw proteomics, filters for plasma and baseline visits, removes noisy 
    undetected proteins, and pivots the data into a machine-learning-ready feature matrix.
    """
    print("\n--- (1) PROTEOMICS: Filtering & Pivoting ---")
    prot = pd.read_csv(DATA_DIR / FILES["prot"], low_memory=False)
    
    # 1. BIOLOGICAL FILTER: We only want blood plasma. 
    # The 'SampleType' column contains QC flags; 'SampleMatrixType' contains the actual fluid type.
    prot_plasma = prot[prot["SampleMatrixType"].astype(str).str.lower().str.contains("plasma")].copy()
    print(f"  Plasma rows: {len(prot_plasma):,}")

    # 2. TEMPORAL FILTER: Keep only the baseline visit to avoid data leakage 
    # (we want to predict disease state *at* baseline, not use future data).
    prot_bl = prot_plasma[prot_plasma["VISCODE"].astype(str).str.lower().isin(BASELINE_VISCODES)].copy()
    print(f"  Baseline rows: {len(prot_bl):,}")

    # 3. QUALITY CONTROL FILTER: Handle assay detectability limits.
    if "TargetDetectability" in prot_bl.columns:
        before = len(prot_bl)
        
        # Strip the '%' sign, convert to float, and scale to 0.0 - 1.0
        td_clean = prot_bl["TargetDetectability"].astype(str).str.replace("%", "", regex=False)
        td_numeric = pd.to_numeric(td_clean, errors="coerce") / 100.0
        
        # Filter for proteins detected in >= 50% of samples
        prot_bl = prot_bl[td_numeric >= DETECT_THRESHOLD].copy()
            
        print(f"  After detectability filter: {before:,} -> {len(prot_bl):,} rows")
        
        # We are temporarily turning the filter OFF so the rest of the script can run
        # prot_bl = prot_bl[ ... ]


    # 4. IDENTIFY FEATURE NAMES: Find which column holds the protein name
    protein_col = next((c for c in ["ProteinName", "Target", "UniProtID"] if c in prot_bl.columns), None)
    
    # 5. RESHAPE DATA: Pivot from "Long" (many rows per patient) to "Wide" (one row per patient, many columns)
    # This creates our feature matrix `X` for Scikit-Learn.
    prot_wide = prot_bl.pivot_table(
        index="RID",           # Patient ID becomes the row index
        columns=protein_col,   # Protein names become the column headers
        values="NPQ",          # Normalized Protein Quantity becomes the cell values
        aggfunc="first"        # If there are duplicates, take the first one
    ).reset_index()
    
    # Clean up the column names after pivoting
    prot_wide.columns.name = None
    
    # 6. ID STANDARDIZATION: Force RID to be an Integer so it merges perfectly later
    prot_wide['RID'] = pd.to_numeric(prot_wide['RID'], errors='coerce').astype('Int64')
    prot_wide = prot_wide.dropna(subset=['RID'])
    
    print(f"  Wide matrix shape: {prot_wide.shape} (patients x proteins)")
    return prot_wide


def build_baseline_cohort():
    """
    Extracts clinical diagnoses at baseline and maps them to clean binary labels (CN vs MCI)
    to serve as the target variable (y) for our ML models.
    """
    print("\n--- (2) DIAGNOSIS: Extracting Baseline Labels ---")
    dx = pd.read_csv(DATA_DIR / FILES["dx"], low_memory=False)

    # 1. TEMPORAL FILTER: Get baseline diagnoses only
    dx_bl = dx[dx["VISCODE"].astype(str).str.lower().isin(BASELINE_VISCODES)].copy()

    # 2. LABEL ENGINEERING: Convert ADNI's complex phase-specific codes into clean binary classes
    def get_label(row):
        # First, try the harmonized DIAGNOSIS column (used in ADNI2/3/4)
        try:
            d = int(float(row.get("DIAGNOSIS", -1)))
            if d == 1: return "CN"
            if d == 2: return "MCI"
            # Note: We exclude d == 3 (Alzheimer's Disease) as per our proposal scope
        except: 
            pass
        
        # Fallback for ADNI1 data where the DIAGNOSIS column might be empty
        if row.get("DXMCI", 0) == 1: return "MCI"
        if row.get("DXNORM", 0) == 1: return "CN"
        return "exclude"

    # Apply the logic to create our target variable 'y'
    dx_bl["cohort_group"] = dx_bl.apply(get_label, axis=1)
    
    # Drop anyone who isn't CN or MCI
    cohort = dx_bl[dx_bl["cohort_group"] != "exclude"][["RID", "cohort_group"]].copy()
    
    # Ensure strict 1-to-1 mapping by keeping only the earliest baseline record per patient
    cohort = cohort.drop_duplicates(subset="RID", keep="first")
    
    # ID STANDARDIZATION: Force RID to Integer
    cohort['RID'] = pd.to_numeric(cohort['RID'], errors='coerce').astype('Int64')
    cohort = cohort.dropna(subset=['RID'])

    print(f"  Clean baseline labels: {len(cohort):,} patients")
    print(cohort["cohort_group"].value_counts().to_string())
    return cohort


def main():
    # 1. Create Feature Matrix (X)
    prot_wide = process_proteomics()
    
    # 2. Create Target Labels (y)
    cohort = build_baseline_cohort()

    print("\n--- (3) MERGE: Joining Proteomics with Labels ---")
    # 3. Inner Join: Keep only patients who have BOTH a diagnosis AND proteomics data
    final_df = pd.merge(cohort, prot_wide, on="RID", how="inner")
    
    # 4. Save to disk
    out_path = OUT_DIR / "adni_nulisa_cohort.csv"
    final_df.to_csv(out_path, index=False)
    
    print(f"\n✅ SUCCESS: Saved final cohort to {out_path}")
    print(f"  Final Shape: {final_df.shape} (Patients: {len(final_df)})")
    print("\nGroup Balance:")
    print(final_df['cohort_group'].value_counts().to_string())

if __name__ == "__main__":
    main()