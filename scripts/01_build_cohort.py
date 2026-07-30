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
import glob

DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "prot": "BSHRI_PLA_CSF_NULISA_CNS",
    "dx": "DXSUM",
    "demog": "PTDEMOG",
    "apoe": "APOERES",
    "mmse": "MMSE",
}

DETECT_THRESHOLD = 0.50  # drop proteins detected in < 50% of samples
BASELINE_VISCODES = {"bl", "sc"}  # 'sc' (screening) used in ADNI2/GO

def get_file(filename):
    '''
    Returns first csv matching prefix of filename.
    '''
    matches = list(DATA_DIR.glob(f'{filename}*.csv'))
    
    if len(matches) > 1:
        print(f'Warning: Multiple files found for {filename}. Using {matches[0].name}')

    return matches[0]


def process_proteomics():
    """
    Loads raw proteomics, filters for plasma and baseline visits, removes noisy 
    undetected proteins, and pivots the data into a machine-learning-ready feature matrix.
    """
    print("\n--- (1) PROTEOMICS: Filtering & Pivoting ---")
    prot = pd.read_csv(get_file(FILES["prot"]), low_memory=False)

    # Filter to plasma only (file also contains CSF rows)
    prot_plasma = prot[prot["SampleMatrixType"].astype(str).str.lower().str.contains("plasma")].copy()
    print(f"  Plasma rows: {len(prot_plasma):,}")

    # Keep baseline visits only
    prot_bl = prot_plasma[prot_plasma["VISCODE"].astype(str).str.lower().isin(BASELINE_VISCODES)].copy()
    print(f"  Baseline rows: {len(prot_bl):,}")

    # Drop proteins below detectability threshold
    if "TargetDetectability" in prot_bl.columns:
        before = len(prot_bl)
        td_clean = prot_bl["TargetDetectability"].astype(str).str.replace("%", "", regex=False)
        td_numeric = pd.to_numeric(td_clean, errors="coerce") / 100.0
        prot_bl = prot_bl[td_numeric >= DETECT_THRESHOLD].copy()
        print(f"  After detectability filter: {before:,} -> {len(prot_bl):,} rows")

    # Find which column holds the protein name
    protein_col = next((c for c in ["ProteinName", "Target", "UniProtID"] if c in prot_bl.columns), None)

    # Pivot from long (many rows per patient) to wide (one row per patient, one col per protein)
    prot_wide = prot_bl.pivot_table(
        index="RID",
        columns=protein_col,
        values="NPQ",
        aggfunc="first"
    ).reset_index()

    # Carry baseline exam date through for age calculation
    exam_dates = prot_bl[["RID", "EXAMDATE"]].drop_duplicates(subset="RID")
    prot_wide = prot_wide.merge(exam_dates, on="RID", how="left")
    prot_wide['EXAMDATE'] = pd.to_datetime(prot_wide['EXAMDATE'])

    prot_wide.columns.name = None
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
    dx = pd.read_csv(get_file(FILES["dx"]), low_memory=False)

    dx_bl = dx[dx["VISCODE"].astype(str).str.lower().isin(BASELINE_VISCODES)].copy()

    def get_label(row):
        # Try harmonized DIAGNOSIS column first (ADNI2/3/4)
        try:
            d = int(float(row.get("DIAGNOSIS", -1)))
            if d == 1: return "CN"
            if d == 2: return "MCI"
        except:
            pass
        # Fallback for ADNI1 (uses flag columns instead)
        if row.get("DXMCI", 0) == 1: return "MCI"
        if row.get("DXNORM", 0) == 1: return "CN"
        return "exclude"

    dx_bl["cohort_group"] = dx_bl.apply(get_label, axis=1)

    cohort = dx_bl[dx_bl["cohort_group"] != "exclude"][["RID", "PHASE", "cohort_group"]].copy()
    cohort = cohort.drop_duplicates(subset="RID", keep="first")
    cohort['RID'] = pd.to_numeric(cohort['RID'], errors='coerce').astype('Int64')
    cohort = cohort.dropna(subset=['RID'])

    print(f"  Clean baseline labels: {len(cohort):,} patients")
    print(cohort["cohort_group"].value_counts().to_string())
    return cohort


def load_demographics():
    """
    Loads PTDEMOG and returns one row per patient with sex and education.
    PTGENDER: 1 = Male, 2 = Female (kept as-is; downstream scripts recode to 0/1).
    PTEDUCAT: years of education.
    """
    print("\n--- (3) DEMOGRAPHICS ---")
    demog = pd.read_csv(get_file(FILES["demog"]), low_memory=False)

    # Date of birth: PTDOB is 'MM/YYYY' (month precision); PTDOBYY is a full date
    # STRING like '1931-01-01', not a year integer. Running to_numeric() on PTDOBYY
    # coerces every value to NaN, which is what silently wiped out AGE. Parse both as
    # dates and prefer PTDOB, since PTDOBYY's month/day are placeholders (always 01-01).
    dob = pd.to_datetime(demog["PTDOB"], format="%m/%Y", errors="coerce")
    dob_fallback = pd.to_datetime(demog["PTDOBYY"], errors="coerce")
    demog["DOB"] = dob.fillna(dob_fallback)

    demog['RID'] = pd.to_numeric(demog['RID'], errors='coerce').astype('Int64')
    demog = demog.dropna(subset=['RID'])
    # groupby.first() skips NaN per column, so a patient missing DOB on one visit
    # still picks up the value from another.
    demog = demog.groupby("RID", as_index=False)[["PTGENDER", "PTEDUCAT", "DOB"]].first()
    print(f"  Demographics for {len(demog):,} patients "
          f"({demog['DOB'].notna().sum():,} with usable DOB)")

    return demog


def load_apoe():
    """
    Loads APOERES and computes the APOE ε4 allele count (0, 1, or 2).
    The GENOTYPE column is formatted as e.g. '3/4', '4/4'.
    We count how many '4' alleles appear.
    """
    print("\n--- (4) APOE ---")
    apoe = pd.read_csv(get_file(FILES["apoe"]), low_memory=False)
    apoe['RID'] = pd.to_numeric(apoe['RID'], errors='coerce').astype('Int64')
    apoe = apoe.dropna(subset=['RID', 'GENOTYPE'])

    apoe["apoe4_count"] = apoe["GENOTYPE"].astype(str).str.count("4")
    apoe = apoe.groupby("RID", as_index=False)["apoe4_count"].max()
    # ASCII only: the Windows console codec (cp1252) cannot encode a literal epsilon.
    print(f"  APOE e4 counts for {len(apoe):,} patients")
    return apoe


def load_mmse():
    """
    Loads MMSE scores filtered to the baseline visit.
    MMSCORE: Mini-Mental State Exam score (0–30, higher = better).
    """
    print("\n--- (5) MMSE ---")
    mmse = pd.read_csv(get_file(FILES["mmse"]), low_memory=False)
    mmse['RID'] = pd.to_numeric(mmse['RID'], errors='coerce').astype('Int64')
    mmse = mmse.dropna(subset=['RID'])

    mmse = mmse[mmse["VISCODE"].astype(str).str.strip().str.lower().isin(BASELINE_VISCODES)].copy()
    mmse = mmse.groupby("RID", as_index=False)[["MMSCORE"]].first()
    print(f"  Baseline MMSE for {len(mmse):,} patients")
    return mmse


def main():
    # 1. Create Feature Matrix (X)
    prot_wide = process_proteomics()

    # 2. Create Target Labels (y)
    cohort = build_baseline_cohort()

    # 3-5. Load covariates
    demog = load_demographics()
    apoe = load_apoe()
    mmse = load_mmse()

    print("\n--- (6) MERGE: Joining Everything by RID ---")
    final_df = pd.merge(cohort, prot_wide, on="RID", how="inner")
    final_df = final_df.merge(demog, on="RID", how="left")
    final_df = final_df.merge(apoe, on="RID", how="left")
    final_df = final_df.merge(mmse, on="RID", how="left")

    # Age at baseline exam, in years with month precision
    final_df['AGE'] = (final_df['EXAMDATE'] - final_df['DOB']).dt.days / 365.25
    final_df['AGE'] = final_df['AGE'].round(1)
    final_df = final_df.drop(columns=['EXAMDATE', 'DOB'])

    out_path = OUT_DIR / "adni_nulisa_cohort.csv"
    final_df.to_csv(out_path, index=False)

    print(f"\nSaved final cohort to {out_path}")
    print(f"  Final Shape: {final_df.shape} (Patients: {len(final_df)})")
    print(f"  Covariate coverage:")
    for col in ["AGE", "PTGENDER", "PTEDUCAT", "apoe4_count", "MMSCORE"]:
        n = final_df[col].notna().sum()
        print(f"    {col}: {n}/{len(final_df)} ({100*n/len(final_df):.0f}%)")
    print("\nGroup Balance:")
    print(final_df['cohort_group'].value_counts().to_string())

if __name__ == "__main__":
    main()