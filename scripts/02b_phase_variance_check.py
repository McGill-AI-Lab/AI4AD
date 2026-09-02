"""
02b_phase_variance_check.py - Independent PCA phase-variance check

Rebuilds the phase-variance PCA check from the Phase 1 cohort using the same
preprocessing choices documented in reports/phase2_decisions.md, while saving
the outputs to separate filenames.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols

COHORT_PATH = Path("data/processed/adni_nulisa_cohort.csv")
OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)

META_COLS = ["RID", "PHASE", "cohort_group", "AGE", "PTGENDER",
             "PTEDUCAT", "apoe4_count", "MMSCORE"]

MISSING_CUTOFF = 0.20
FLOOR = 5.0
N_PCS = 10

def variance_explained_by_group(y, group):
    return ols("y ~ C(g)", data=pd.DataFrame({"y": y, "g": group})).fit().rsquared


df = pd.read_csv(COHORT_PATH)
protein_cols = [c for c in df.columns if c not in META_COLS]
print(f"Loaded {df.shape[0]} patients, {len(protein_cols)} protein columns")

if "PHASE" not in df.columns:
    raise ValueError("No PHASE column in this file, cannot run the check.")

print("Phase counts:", dict(df["PHASE"].value_counts()))
print("\nPhase vs diagnosis:")
print(pd.crosstab(df["PHASE"], df["cohort_group"]))

# Floor values to NaN before filtering or imputation.
X = df[protein_cols].copy()
floored = (X < FLOOR).sum().sum()
print(f"\n{floored} values below the {FLOOR} floor, setting these to NaN")
X = X.mask(X < FLOOR)

# Drop a protein if it fails the cutoff in any single phase.
print(f"\nChecking missingness per phase (cutoff {int(MISSING_CUTOFF*100)}%)...")
phases = df["PHASE"].values
missing_by_phase = {}
for phase in df["PHASE"].unique():
    phase_mask = phases == phase
    missing_by_phase[phase] = X.loc[phase_mask].isna().mean()

missing_by_phase_df = pd.DataFrame(missing_by_phase)
missing_by_phase_df["all_rows"] = X.isna().mean()
fails_any_phase = (missing_by_phase_df.drop(columns="all_rows") > MISSING_CUTOFF).any(axis=1)
drop_cols = missing_by_phase_df[fails_any_phase].index.tolist()

print(f"Dropping {len(drop_cols)} proteins that fail the cutoff in at least one phase:")
print(missing_by_phase_df.loc[drop_cols].round(3))

X = X.drop(columns=drop_cols)
X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=X.index)
print(f"\n{X.shape[1]} proteins remaining after filter + imputation")

# No log transform: NPQ values are already on the log2 scale.

X_scaled = StandardScaler().fit_transform(X)

n_pcs = min(N_PCS, X_scaled.shape[1], X_scaled.shape[0])
pca = PCA(n_components=n_pcs)
scores = pca.fit_transform(X_scaled)
pc_names = [f"PC{i+1}" for i in range(n_pcs)]

scores_df = pd.DataFrame(scores, columns=pc_names, index=df.index)
scores_df["RID"] = df["RID"].values
scores_df["PHASE"] = df["PHASE"].values
scores_df["cohort_group"] = df["cohort_group"].values

print(f"\nTop {n_pcs} PCs explain this much variance each:")
for name, v in zip(pc_names, pca.explained_variance_ratio_):
    print(f"  {name}: {v*100:.2f}%")

phase_r2, diag_r2 = [], []
for pc in pc_names:
    phase_r2.append(variance_explained_by_group(scores_df[pc], scores_df["PHASE"]))
    diag_r2.append(variance_explained_by_group(scores_df[pc], scores_df["cohort_group"]))

variance_table = pd.DataFrame({
    "PC": pc_names,
    "pct_of_total_variance": pca.explained_variance_ratio_ * 100,
    "phase_r2": phase_r2,
    "diagnosis_r2": diag_r2,
})
variance_table["phase_share"] = variance_table["pct_of_total_variance"] * variance_table["phase_r2"]
variance_table["diagnosis_share"] = variance_table["pct_of_total_variance"] * variance_table["diagnosis_r2"]

print("\n", variance_table.round(3).to_string(index=False))

phase_total = variance_table["phase_share"].sum()
diag_total = variance_table["diagnosis_share"].sum()

print(f"\nPhase explains roughly {phase_total:.2f}% of total variance")
print(f"Diagnosis explains roughly {diag_total:.2f}% of total variance, for comparison")
print("(Phase 2 writeup: 2.79% phase, 0.38% diagnosis)")

if phase_total > 10:
    print("-> Over the 10% cutoff, so ComBat should be applied per the proposal")
else:
    print("-> Under 10%, so ComBat is not indicated per the proposal")

if phase_total > 5 and diag_total > 5:
    print(
        "\nBoth numbers are non-trivial here; it may be worth checking phase "
        "variance within CN and MCI separately before locking in the ComBat "
        "call, since ADNIGO is 100% MCI."
    )

scores_df.to_csv(OUT_DIR / "pca_scores_cedric_check.csv", index=False)
variance_table.to_csv(OUT_DIR / "pca_variance_cedric_check.csv", index=False)
print("\nSaved reports/pca_scores_cedric_check.csv and reports/pca_variance_cedric_check.csv")
print("(Saved separately from the merged Phase 2 exports.)")
