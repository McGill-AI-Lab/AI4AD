"""
03_model.py - Clinical-only baseline model (Model 1) for CN vs MCI.

This script establishes our H_1 Baseline. Before we test if the 113 NULISA 
proteins actually predict Mild Cognitive Impairment (MCI), we need to see 
how well standard clinical data does on its own.
"""

from __future__ import annotations

from pathlib import Path

# Data manipulation libraries
import pandas as pd

# Scikit-learn machine learning tools
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- 1. FILE PATHS & CONFIGURATION ---
# Define exactly where the raw ADNI data and our processed cohort live
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

COHORT_FILE = PROCESSED_DIR / "adni_nulisa_cohort.csv"
PTDEMOG_FILE = RAW_DIR / "PTDEMOG_16Feb2026.csv"
APOE_FILE = RAW_DIR / "APOERES_20Feb2026.csv"
MMSE_FILE = RAW_DIR / "MMSE_16Feb2026.csv"

# These are the 4 specific clinical features we are using for our baseline
FEATURE_COLUMNS = ["PTEDUCAT", "apoe4_count", "MMSCORE", "is_male"]


# --- 2. DATA CLEANING FUNCTIONS ---
def standardize_rid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the 'RID' (Roster ID) column.
    Because ADNI tables sometimes format IDs differently (e.g., ' 002' vs '2'),
    we force them all to be integers so our pandas merges don't fail.
    """
    out = df.copy()
    # Convert to numeric, turn errors into NaN, and cast as Int64
    out["RID"] = pd.to_numeric(out["RID"], errors="coerce").astype("Int64")
    # Drop any rows where the RID didn't convert properly
    out = out.dropna(subset=["RID"])
    return out


def build_apoe_feature(apoe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the APOE epsilon-4 allele count (0, 1, or 2) for each patient.
    Parses the 'GENOTYPE' column which is formatted as '3/4', '4/4', etc.
    """
    apoe = standardize_rid(apoe)

    # Drop rows where GENOTYPE is missing
    apoe = apoe.dropna(subset=['GENOTYPE'])

    # Count how many '4's appear in the string (e.g., '3/4' has one, '4/4' has two)
    apoe["apoe4_count"] = apoe["GENOTYPE"].astype(str).str.count('4')
    
    # If a patient has multiple records, just grab their maximum count
    apoe_rid = apoe.groupby("RID", as_index=False)["apoe4_count"].max()
    return apoe_rid


# --- 3. MERGING PIPELINE ---
def load_and_merge() -> pd.DataFrame:
    """Load cohort + clinical covariates and return merged dataframe."""
    cohort = standardize_rid(pd.read_csv(COHORT_FILE, low_memory=False))
    ptdemog = standardize_rid(pd.read_csv(PTDEMOG_FILE, low_memory=False))
    apoe = pd.read_csv(APOE_FILE, low_memory=False)
    mmse = standardize_rid(pd.read_csv(MMSE_FILE, low_memory=False))

    ptdemog_first = (
        ptdemog.groupby("RID", as_index=False)[["PTGENDER", "PTEDUCAT"]].first()
    )
    apoe_feature = build_apoe_feature(apoe)
    
    # FIX: ADNI uses 'sc' (Screening) or 'bl' (Baseline) for the first visit
    valid_viscodes = ["bl", "sc"]
    mmse_bl = mmse[mmse["VISCODE"].astype(str).str.strip().str.lower().isin(valid_viscodes)].copy()
    mmse_bl = mmse_bl.groupby("RID", as_index=False)[["MMSCORE"]].first()

    merged = cohort.merge(ptdemog_first, on="RID", how="left")
    merged = merged.merge(apoe_feature, on="RID", how="left")
    merged = merged.merge(mmse_bl, on="RID", how="left")
    return merged


# --- 4. MACHINE LEARNING PREPARATION ---
def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features X and labels y from merged dataframe."""
    out = df.copy()
    
    # FIX: Force to numeric, then map 1 -> 1 (Male) and 2 -> 0 (Female)
    ptgender_num = pd.to_numeric(out["PTGENDER"], errors="coerce")
    out["is_male"] = ptgender_num.map({1: 1, 2: 0})

    out["cohort_group"] = (
        out["cohort_group"].astype(str).str.strip().str.upper().map({"CN": 0, "MCI": 1})
    )
    out = out.dropna(subset=["cohort_group"])
    y = out["cohort_group"].astype(int)
    X = out[FEATURE_COLUMNS].copy()
    return X, y


# --- 5. MODEL TRAINING & EVALUATION ---
def evaluate_model(X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    """
    Train an L2 Regularized Logistic Regression using Stratified 5-Fold CV.
    We use a scikit-learn Pipeline to guarantee ZERO data leakage!
    """
    # Create the pipeline
    pipeline = Pipeline(
        steps=[
            # STEP 1: Impute missing clinical data using the Median 
            ("imputer", SimpleImputer(strategy="median")),
            
            # STEP 2: Standardize data (mean=0, variance=1) so the L2 penalty works correctly
            ("scaler", StandardScaler()),
            
            # STEP 3: Train the L2 Logistic Regression
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    # Setup Stratified K-Fold to maintain the CN vs MCI class balance in every fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Run the cross-validation and compute the Area Under the Curve (AUC)
    auc_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=make_scorer(roc_auc_score, response_method="predict_proba"),
    )
    
    # Return the mean AUC and its standard deviation
    return float(auc_scores.mean()), float(auc_scores.std())


# --- 6. MAIN EXECUTION ---
def main() -> None:
    print("--- Loading and merging cohort + clinical covariates ---")
    merged = load_and_merge()
    print(f"Merged shape: {merged.shape}")

    print("--- Preparing model features ---")
    X, y = prepare_xy(merged)
    print(f"X shape: {X.shape}")
    print(f"Class balance (0=CN, 1=MCI):\n{y.value_counts().sort_index().to_string()}")

    print("--- Running Stratified 5-Fold CV (Logistic Regression) ---")
    mean_auc, std_auc = evaluate_model(X, y)

    print("\nModel 1 Clinical Baseline Performance")
    print(f"Mean AUC-ROC: {mean_auc:.4f}")
    print(f"Std AUC-ROC:  {std_auc:.4f}")


# This block ensures the code only runs when you execute the script directly
if __name__ == "__main__":
    main()