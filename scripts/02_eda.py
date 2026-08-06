"""
02_eda.py - Phase 2: EDA & Preprocessing
========================================
Takes the locked Phase 1 cohort (data/processed/adni_nulisa_cohort.csv) and
produces the modelling feature matrix, plus the numbers behind the Phase 2
decisions writeup (reports/phase2_decisions.md).

Pipeline (reasoning for each step is in reports/phase2_decisions.md):

  1. Drop `Apolipoprotein E`                                  [D4]
  2. Below-detection (NPQ < 5) -> NaN                         [D3]
  3. Missingness filter, WITHIN-PHASE >20% not pooled         [D2]
  4. Median imputation, global column median
  5. NO log2 transform                                        [D1]
  6. Covariate prep, sex 1/2 -> 0/1, scaling deferred         [D5]
  7. Batch check, PCA variance explained by PHASE             [D6]

Outputs
  data/processed/phase2_feature_matrix.csv  locked, UNSCALED modelling matrix
  reports/pca_scores.csv                    RID, PC1-PC5, PHASE, cohort_group
  reports/pca_variance.csv                  per-PC variance + phase/dx R^2
  reports/phase2_preprocessing_summary.json machine-readable run summary
  plots/*.png                               working plots (report figures are
                                            Cedric's and Jessica's branches)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Setup paths
DATA_DIR = Path("data/processed")
PLOT_DIR = Path("plots")
REPORT_DIR = Path("reports")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Warn (not fail) if the cohort shape shifts, it would mean a different cohort.
EXPECTED_N_PROTEINS = 116
EXPECTED_SHAPE = (728, 124)

# Preprocessing constants (see the writeup before changing any of these) 

DROP_PROTEINS = ["Apolipoprotein E"]  # [D4] genotype proxy, duplicates apoe4_count
DETECTION_FLOOR = 5.0                 # [D3] below this is undetected, not measured
MISSING_THRESHOLD = 0.20              # [D2] applied per PHASE, not pooled
COMBAT_TRIGGER = 0.10                 # [D6] phase variance above this -> ComBat
N_PCS_EXPORT = 5                      # PCs written out for the figure work

ID_COLS = ["RID", "PHASE"]
LABEL_COL = "cohort_group"
COVARIATE_COLS = ["AGE", "PTGENDER", "PTEDUCAT", "apoe4_count", "MMSCORE"]


def variance_explained_by_group(scores, groups):
    """Fraction of a vector's variance lying between groups (one-way ANOVA R^2).

    Applied per PC and weighted by that PC's variance share, this gives the
    share of total protein variance explained by the grouping.
    """
    scores = np.asarray(scores, dtype=float)
    grand_mean = scores.mean()
    total_ss = ((scores - grand_mean) ** 2).sum()
    if total_ss == 0:
        return 0.0
    between_ss = sum(
        len(scores[groups == g]) * (scores[groups == g].mean() - grand_mean) ** 2
        for g in pd.unique(groups)
    )
    return float(between_ss / total_ss)


def main():
    print("--- 1. LOADING DATA ---")
    df = pd.read_csv(DATA_DIR / "adni_nulisa_cohort.csv")
    print(f"Loaded cohort with shape: {df.shape}")
    if df.shape != EXPECTED_SHAPE:
        print(f"  WARNING: expected shape {EXPECTED_SHAPE}. This is a different "
              "cohort than the Phase 2 brief was verified against -- compare "
              "before trusting anything downstream.")

    # These must stay OUT of protein_cols or the filters and PCA treat them as
    # proteins.
    meta_cols = ID_COLS + [LABEL_COL] + COVARIATE_COLS
    missing_meta = [c for c in meta_cols if c not in df.columns]
    assert not missing_meta, (
        f"Cohort CSV is missing expected non-protein columns: {missing_meta}. "
        "Re-run 01_build_cohort.py -- an older version of it did not emit covariates."
    )

    protein_cols = [c for c in df.columns if c not in meta_cols]
    X = df[protein_cols].copy()
    y = df[LABEL_COL]
    phase = df["PHASE"]

    # Guard against a covariate (or any string column) leaking into the protein matrix.
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    assert not non_numeric, (
        f"Non-numeric columns leaked into the protein matrix: {non_numeric}. "
        "Add them to meta_cols before transforming."
    )

    print(f"Protein features: {len(protein_cols)} | covariates held out: {COVARIATE_COLS}")
    if len(protein_cols) != EXPECTED_N_PROTEINS:
        print(f"  WARNING: expected {EXPECTED_N_PROTEINS} proteins, found {len(protein_cols)}. "
              "Confirm this is intentional (detectability filter changed?) and update "
              "EXPECTED_N_PROTEINS.")

    print("\n--- 2. RAW MISSINGNESS ---")
    missing_pct = X.isna().mean() * 100
    print(f"Average missingness across all proteins: {missing_pct.mean():.2f}%")
    print(f"Max missingness for a single protein:    {missing_pct.max():.2f}%")
    print(f"Min missingness for a single protein:    {missing_pct.min():.2f}%")

    row_missing = X.isna().mean(axis=1) * 100
    print(f"Worst single sample: {row_missing.max():.2f}% missing "
          f"| samples over 20%: {(row_missing > 20).sum()}")

    plt.figure(figsize=(8, 5))
    plt.hist(missing_pct, bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of Missing Data across {len(protein_cols)} Proteins")
    plt.xlabel("Percentage of Missing Values (%)")
    plt.ylabel("Number of Proteins")
    plt.savefig(PLOT_DIR / "missingness_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/missingness_histogram.png")

    print("\n--- 3. DECISION 4: DROP GENOTYPE-PROXY PROTEINS ---")
    dropped_proxy = [c for c in DROP_PROTEINS if c in X.columns]
    if dropped_proxy:
        # Log the genotype association the drop rests on.
        for col in dropped_proxy:
            if col == "Apolipoprotein E":
                below = X[col] < DETECTION_FLOOR
                rates = below.groupby(df["apoe4_count"]).mean() * 100
                detail = ", ".join(f"e4={int(k)}: {v:.1f}%" for k, v in rates.items())
                print(f"  '{col}' fraction below detection by genotype -> {detail}")
        X = X.drop(columns=dropped_proxy)
    print(f"Dropped {len(dropped_proxy)} protein(s) as genotype proxies: {dropped_proxy}")

    print("\n--- 4. DECISION 3: BELOW-DETECTION VALUES -> NaN ---")
    below_floor = (X < DETECTION_FLOOR) & X.notna()
    n_floor = int(below_floor.to_numpy().sum())
    n_zeros = int((X == 0).to_numpy().sum())
    print(f"Values below NPQ {DETECTION_FLOOR}: {n_floor} (of which exact zeros: {n_zeros}) "
          f"across {int((below_floor.sum() > 0).sum())} proteins")
    # Per-protein counts go to the JSON summary, not the console.
    floor_counts = below_floor.sum()
    floor_counts = {k: int(v) for k, v in floor_counts[floor_counts > 0]
                    .sort_values(ascending=False).items()}
    X = X.mask(below_floor)
    print("Masked to NaN so they cannot drag the imputation medians down.")

    print("\n--- 5. DECISION 2: WITHIN-PHASE MISSINGNESS FILTER ---")
    # A protein must clear the threshold in EVERY phase to survive.
    per_phase_missing = X.isna().groupby(phase).mean()
    worst_phase_missing = per_phase_missing.max()
    pooled_missing = X.isna().mean()

    drop_within = worst_phase_missing[worst_phase_missing > MISSING_THRESHOLD].index.tolist()
    drop_pooled = pooled_missing[pooled_missing > MISSING_THRESHOLD].index.tolist()
    only_within = [c for c in drop_within if c not in drop_pooled]

    print(f"A pooled >20% filter would drop {len(drop_pooled)} proteins.")
    print(f"The within-phase >20% filter drops  {len(drop_within)} proteins "
          f"({len(only_within)} of them caught only by the within-phase rule).")
    # Per-phase rates behind that difference -> JSON summary.
    only_within_detail = {
        col: {"pooled": round(float(pooled_missing[col]) * 100, 1),
              **{p: round(float(per_phase_missing.loc[p, col]) * 100, 1)
                 for p in per_phase_missing.index}}
        for col in only_within
    }

    X_filtered = X.drop(columns=drop_within)
    print(f"Proteins retained: {X_filtered.shape[1]}")

    print("\n--- 6. MEDIAN IMPUTATION ---")
    # Global, not per-phase: a per-phase median would write phase -- and via
    # ADNIGO, diagnosis -- into the imputed values.
    n_imputed = int(X_filtered.isna().to_numpy().sum())
    medians = X_filtered.median()
    X_imputed = X_filtered.fillna(medians)
    assert not X_imputed.isna().any().any(), "Imputation left NaNs behind."
    print(f"Imputed {n_imputed} cells "
          f"({n_imputed / X_imputed.size * 100:.2f}% of the retained matrix) "
          "with the global column median.")

    print("\n--- 7. DECISION 1: NO LOG2 TRANSFORM ---")
    desc = X_imputed.to_numpy().ravel()
    print("NPQ value distribution (retained proteins, post-imputation):")
    print(f"    min {np.min(desc):.2f} | p1 {np.percentile(desc, 1):.2f} | "
          f"median {np.median(desc):.2f} | p99 {np.percentile(desc, 99):.2f} | "
          f"max {np.max(desc):.2f}")
    print("Already log2 (NULISA NPQ is defined that way), so log2(x+1) is SKIPPED.")

    print("\n--- 8. COVARIATE PREP ---")
    covars = df[COVARIATE_COLS].copy()
    # ADNI codes PTGENDER as 1 = Male, 2 = Female.
    assert set(covars["PTGENDER"].dropna().unique()) <= {1.0, 2.0}, (
        f"Unexpected PTGENDER codes: {sorted(covars['PTGENDER'].dropna().unique())}"
    )
    covars["SEX_FEMALE"] = (covars["PTGENDER"] == 2).astype(int)
    covars = covars.drop(columns=["PTGENDER"])
    print(f"Recoded PTGENDER 1/2 -> SEX_FEMALE 0/1 "
          f"(female n={int(covars['SEX_FEMALE'].sum())}, "
          f"male n={int((1 - covars['SEX_FEMALE']).sum())})")

    coverage = covars.notna().mean() * 100
    print("Covariate coverage: " + ", ".join(f"{c} {v:.1f}%" for c, v in coverage.items()))
    assert (coverage == 100).all(), (
        "A covariate has missing values; Phase 1 promised 100% coverage. "
        "Re-run 01_build_cohort.py and check the AGE parse."
    )

    # [D5] Not applied to the export -- Modelling scales in-fold. These stats are
    # for tables and figures only.
    scaling_stats = {
        col: {"mean": float(covars[col].mean()), "std": float(covars[col].std(ddof=0))}
        for col in ["AGE", "MMSCORE"]
    }
    print("Continuous covariates left UNSCALED in the export (see decision 5).")
    for col, s in scaling_stats.items():
        print(f"    {col}: mean {s['mean']:.2f}, sd {s['std']:.2f} "
              "(full-cohort, descriptive only -- fit the scaler in-fold)")

    print("\n--- 9. DECISION 6: BATCH CHECK (PCA) ---")
    # Z-score first, or PC1 just tracks whichever protein has the widest range.
    protein_means = X_imputed.mean()
    protein_sds = X_imputed.std(ddof=0).replace(0, 1.0)
    X_z = (X_imputed - protein_means) / protein_sds

    pca = PCA()
    scores = pca.fit_transform(X_z.to_numpy())
    evr = pca.explained_variance_ratio_

    phase_arr = phase.to_numpy()
    dx_arr = y.to_numpy()
    r2_phase = np.array([variance_explained_by_group(scores[:, k], phase_arr)
                         for k in range(scores.shape[1])])
    r2_dx = np.array([variance_explained_by_group(scores[:, k], dx_arr)
                      for k in range(scores.shape[1])])

    total_phase_var = float((r2_phase * evr).sum())
    total_dx_var = float((r2_dx * evr).sum())

    print(f"PC1 {evr[0] * 100:.1f}% | PC2 {evr[1] * 100:.1f}% | "
          f"PC3 {evr[2] * 100:.1f}% | first 5 cumulative {evr[:5].sum() * 100:.1f}%")
    # Per-PC numbers are written to reports/pca_variance.csv, not printed.
    print(f"TOTAL protein variance explained by PHASE:     {total_phase_var * 100:.2f}%")
    print(f"TOTAL protein variance explained by DIAGNOSIS: {total_dx_var * 100:.2f}%")
    print(f"ComBat trigger threshold: {COMBAT_TRIGGER * 100:.0f}%")
    combat_triggered = total_phase_var > COMBAT_TRIGGER
    if combat_triggered:
        print("  -> OVER THRESHOLD. ComBat indicated. Save BOTH corrected and")
        print("     uncorrected matrices -- Aim 4 measures what ComBat removes.")
    else:
        print("  -> UNDER THRESHOLD. No ComBat; keep the uncorrected matrix.")

    # Exported so figure work does not recompute the PCA.
    pca_scores = pd.DataFrame(
        scores[:, :N_PCS_EXPORT],
        columns=[f"PC{k + 1}" for k in range(N_PCS_EXPORT)],
    )
    pca_scores.insert(0, "RID", df["RID"].to_numpy())
    pca_scores["PHASE"] = phase_arr
    pca_scores["cohort_group"] = dx_arr
    pca_scores.to_csv(REPORT_DIR / "pca_scores.csv", index=False)

    pca_variance = pd.DataFrame({
        "PC": [f"PC{k + 1}" for k in range(len(evr))],
        "explained_variance_ratio": evr,
        "cumulative_variance_ratio": np.cumsum(evr),
        "r2_phase": r2_phase,
        "r2_diagnosis": r2_dx,
    })
    pca_variance.to_csv(REPORT_DIR / "pca_variance.csv", index=False)
    print("Saved reports/pca_scores.csv and reports/pca_variance.csv")

    # Working plots only -- the report figures are Cedric's.
    plt.figure(figsize=(8, 6))
    for group, color in [("ADNI1", "tab:blue"), ("ADNIGO", "tab:green"), ("ADNI3", "tab:orange")]:
        idx = phase_arr == group
        if idx.any():
            plt.scatter(scores[idx, 0], scores[idx, 1], label=group, alpha=0.6, s=25, c=color)
    plt.title(f"Protein PCA by ADNI phase (phase explains {total_phase_var * 100:.1f}% of variance)")
    plt.xlabel(f"PC1 ({evr[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({evr[1] * 100:.1f}% variance)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(PLOT_DIR / "pca_by_phase.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    for group, color in [("CN", "tab:blue"), ("MCI", "tab:orange")]:
        idx = dx_arr == group
        plt.scatter(scores[idx, 0], scores[idx, 1], label=group, alpha=0.6, s=25, c=color)
    plt.title(f"Protein PCA by diagnosis (diagnosis explains {total_dx_var * 100:.1f}% of variance)")
    plt.xlabel(f"PC1 ({evr[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({evr[1] * 100:.1f}% variance)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(PLOT_DIR / "pca_by_diagnosis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/pca_by_phase.png and plots/pca_by_diagnosis.png")

    print("\n--- 10. WRITING FEATURE MATRIX ---")
    feature_matrix = pd.concat(
        [df[["RID", "PHASE", LABEL_COL]].reset_index(drop=True),
         covars.reset_index(drop=True),
         X_imputed.reset_index(drop=True)],
        axis=1,
    )
    assert len(feature_matrix) == len(df), "Row count changed during assembly."
    assert feature_matrix["RID"].is_unique, "Duplicate RIDs in the feature matrix."
    out_path = DATA_DIR / "phase2_feature_matrix.csv"
    feature_matrix.to_csv(out_path, index=False)
    n_meta_out = feature_matrix.shape[1] - X_imputed.shape[1]
    print(f"Wrote {out_path} -- shape {feature_matrix.shape} "
          f"({X_imputed.shape[1]} proteins + {n_meta_out} id/label/covariate columns)")
    print("Values are UNSCALED. Standardise inside the CV pipeline (decision 5).")

    summary = {
        "cohort_shape": list(df.shape),
        "n_proteins_input": len(protein_cols),
        "n_proteins_retained": int(X_imputed.shape[1]),
        "dropped_genotype_proxy": dropped_proxy,
        "detection_floor": DETECTION_FLOOR,
        "n_values_masked_below_floor": n_floor,
        "n_values_below_floor_by_protein": floor_counts,
        "missing_threshold": MISSING_THRESHOLD,
        "dropped_within_phase": drop_within,
        "dropped_pooled_would_be": drop_pooled,
        "dropped_only_by_within_phase_rule": only_within,
        "missingness_pct_of_only_within_drops": only_within_detail,
        "n_cells_imputed": n_imputed,
        "log2_transform_applied": False,
        "covariates": list(covars.columns),
        "scaling_applied": False,
        "descriptive_scaling_stats": scaling_stats,
        "pca": {
            "evr_top5": [float(v) for v in evr[:5]],
            "variance_explained_by_phase": total_phase_var,
            "variance_explained_by_diagnosis": total_dx_var,
            "combat_threshold": COMBAT_TRIGGER,
            "combat_triggered": bool(combat_triggered),
        },
        "output_matrix": str(out_path).replace("\\", "/"),
    }
    with open(REPORT_DIR / "phase2_preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved reports/phase2_preprocessing_summary.json")


    print("\n[OK] Phase 2 preprocessing completed successfully.")


if __name__ == "__main__":
    main()
