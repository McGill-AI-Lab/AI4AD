"""
02c_eda_figures.py - Stage 2 EDA figures
========================================

Builds the Stage 2 PCA and protein distribution figures from the merged Phase 2
preprocessing logic without writing participant-level derived tables to disk.

Outputs
  reports/figures/stage2_pca_comparison.png
  reports/figures/stage2_pca_by_phase.png
  reports/figures/stage2_pca_by_cohort_group.png
  reports/figures/stage2_protein_distribution_grid.png
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import textwrap
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "ai4ad-matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
COHORT_PATH = ROOT / "data/processed/adni_nulisa_cohort.csv"
FEATURE_MATRIX_PATH = ROOT / "data/processed/phase2_feature_matrix.csv"
PCA_SCORES_PATH = ROOT / "reports/pca_scores.csv"
PCA_VARIANCE_PATH = ROOT / "reports/pca_variance.csv"
SUMMARY_PATH = ROOT / "reports/phase2_preprocessing_summary.json"
OUT_DIR = ROOT / "reports/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PCA_COMPARISON_FIG = OUT_DIR / "stage2_pca_comparison.png"
PCA_PHASE_FIG = OUT_DIR / "stage2_pca_by_phase.png"
PCA_COHORT_FIG = OUT_DIR / "stage2_pca_by_cohort_group.png"
PROTEIN_GRID_FIG = OUT_DIR / "stage2_protein_distribution_grid.png"

PCA_PHASE_TITLE = "PCA of Plasma Protein Profiles by ADNI Phase"
PCA_DIAGNOSIS_TITLE = "PCA of Plasma Protein Profiles by Diagnosis"
PCA_MARKER_SIZE = 26
PCA_MARKER_ALPHA = 0.62
PCA_MARKER_EDGEWIDTH = 0.3

PHASE_PALETTE = {
    "ADNI1": "#4C78A8",
    "ADNIGO": "#F58518",
    "ADNI3": "#54A24B",
}

COHORT_PALETTE = {
    "CN": "#4C78A8",
    "MCI": "#E45756",
}

DEFAULT_FINAL_COVARIATES = ["AGE", "PTEDUCAT", "apoe4_count", "MMSCORE", "SEX_FEMALE"]

# Preset proteins for the distribution grid.
REPRESENTATIVE_PROTEINS = [
    "Neurofilament light polypeptide",
    "Glial fibrillary acidic protein",
    "Microtubule-associated protein tau",
    "Neurogranin",
    "Neuronal pentraxin-2",
    "Growth/differentiation factor 15",
    "Chitotriosidase-1",
    "Vascular cell adhesion protein 1",
    "C-reactive protein",
]


def load_phase2_module():
    """Load scripts/02_eda.py so this script follows the merged constants."""
    module_path = ROOT / "scripts/02_eda.py"
    spec = importlib.util.spec_from_file_location("phase2_eda", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load preprocessing module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_summary():
    if not SUMMARY_PATH.exists():
        return {}
    return json.loads(SUMMARY_PATH.read_text())


def final_covariate_cols(summary):
    return summary.get("covariates", DEFAULT_FINAL_COVARIATES)


def build_phase2_feature_matrix_in_memory(eda_module):
    """Rebuild the final Phase 2 matrix in memory using 02_eda.py rules."""
    if not COHORT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {COHORT_PATH}. Run scripts/01_build_cohort.py first."
        )

    df = pd.read_csv(COHORT_PATH)
    meta_cols = eda_module.ID_COLS + [eda_module.LABEL_COL] + eda_module.COVARIATE_COLS
    missing_meta = [col for col in meta_cols if col not in df.columns]
    if missing_meta:
        raise ValueError(
            "Cohort CSV is missing expected non-protein columns: "
            f"{missing_meta}. Re-run scripts/01_build_cohort.py."
        )

    protein_cols = [col for col in df.columns if col not in meta_cols]
    x = df[protein_cols].copy()
    phase = df["PHASE"]

    dropped_proxy = [col for col in eda_module.DROP_PROTEINS if col in x.columns]
    if dropped_proxy:
        x = x.drop(columns=dropped_proxy)

    below_floor = (x < eda_module.DETECTION_FLOOR) & x.notna()
    x = x.mask(below_floor)

    per_phase_missing = x.isna().groupby(phase).mean()
    worst_phase_missing = per_phase_missing.max()
    drop_within = worst_phase_missing[
        worst_phase_missing > eda_module.MISSING_THRESHOLD
    ].index.tolist()

    x_filtered = x.drop(columns=drop_within)
    x_imputed = x_filtered.fillna(x_filtered.median())
    if x_imputed.isna().any().any():
        raise ValueError("In-memory Phase 2 matrix still has NaNs after median imputation.")

    covars = df[eda_module.COVARIATE_COLS].copy()
    unexpected_gender = set(covars["PTGENDER"].dropna().unique()) - {1.0, 2.0}
    if unexpected_gender:
        raise ValueError(f"Unexpected PTGENDER codes: {sorted(unexpected_gender)}")
    covars["SEX_FEMALE"] = (covars["PTGENDER"] == 2).astype(int)
    covars = covars.drop(columns=["PTGENDER"])

    feature_matrix = pd.concat(
        [
            df[["RID", "PHASE", eda_module.LABEL_COL]].reset_index(drop=True),
            covars.reset_index(drop=True),
            x_imputed.reset_index(drop=True),
        ],
        axis=1,
    )

    return feature_matrix


def load_or_build_feature_matrix(eda_module):
    """Use the locked Phase 2 matrix when present, otherwise rebuild in memory."""
    if FEATURE_MATRIX_PATH.exists():
        print(f"Using existing final matrix: {FEATURE_MATRIX_PATH.relative_to(ROOT)}")
        return pd.read_csv(FEATURE_MATRIX_PATH), "disk"

    print(
        "Final matrix missing locally; rebuilding Phase 2 matrix in memory from "
        f"{COHORT_PATH.relative_to(ROOT)} using scripts/02_eda.py rules."
    )
    return build_phase2_feature_matrix_in_memory(eda_module), "in_memory"


def protein_feature_cols(feature_matrix, label_col, covariate_cols, id_cols):
    meta_cols = id_cols + [label_col] + list(covariate_cols)
    return [col for col in feature_matrix.columns if col not in meta_cols]


def pca_artifacts_match(feature_matrix, scores):
    required_cols = ["RID", "PC1", "PC2", "PHASE", "cohort_group"]
    if any(col not in scores.columns for col in required_cols):
        return False

    cohort_view = feature_matrix[["RID", "PHASE", "cohort_group"]].reset_index(drop=True)
    score_view = scores[["RID", "PHASE", "cohort_group"]].reset_index(drop=True)
    return cohort_view.equals(score_view)


def compute_pca_outputs(feature_matrix, eda_module, covariate_cols):
    """Reproduce the 02_eda.py PCA logic without writing scores to disk."""
    proteins = protein_feature_cols(
        feature_matrix=feature_matrix,
        label_col=eda_module.LABEL_COL,
        covariate_cols=covariate_cols,
        id_cols=eda_module.ID_COLS,
    )

    x = feature_matrix[proteins]
    protein_means = x.mean()
    protein_sds = x.std(ddof=0).replace(0, 1.0)
    x_z = (x - protein_means) / protein_sds

    pca = PCA()
    scores = pca.fit_transform(x_z.to_numpy())
    evr = pca.explained_variance_ratio_

    phase_arr = feature_matrix["PHASE"].to_numpy()
    dx_arr = feature_matrix[eda_module.LABEL_COL].to_numpy()
    r2_phase = np.array(
        [
            eda_module.variance_explained_by_group(scores[:, idx], phase_arr)
            for idx in range(scores.shape[1])
        ]
    )
    r2_dx = np.array(
        [
            eda_module.variance_explained_by_group(scores[:, idx], dx_arr)
            for idx in range(scores.shape[1])
        ]
    )

    n_export = min(getattr(eda_module, "N_PCS_EXPORT", 5), scores.shape[1])
    pca_scores = pd.DataFrame(
        scores[:, :n_export],
        columns=[f"PC{idx + 1}" for idx in range(n_export)],
    )
    pca_scores.insert(0, "RID", feature_matrix["RID"].to_numpy())
    pca_scores["PHASE"] = phase_arr
    pca_scores["cohort_group"] = dx_arr

    pca_variance = pd.DataFrame(
        {
            "PC": [f"PC{idx + 1}" for idx in range(len(evr))],
            "explained_variance_ratio": evr,
            "cumulative_variance_ratio": np.cumsum(evr),
            "r2_phase": r2_phase,
            "r2_diagnosis": r2_dx,
        }
    )

    return pca_scores, pca_variance


def load_or_compute_pca(feature_matrix, eda_module, covariate_cols):
    """Prefer the merged PCA exports so figure coordinates stay exact when possible."""
    if PCA_SCORES_PATH.exists() and PCA_VARIANCE_PATH.exists():
        scores = pd.read_csv(PCA_SCORES_PATH)
        variance = pd.read_csv(PCA_VARIANCE_PATH)
        if pca_artifacts_match(feature_matrix, scores):
            print(
                "Using existing merged PCA outputs: "
                f"{PCA_SCORES_PATH.relative_to(ROOT)} and "
                f"{PCA_VARIANCE_PATH.relative_to(ROOT)}"
            )
            return scores, variance, "disk"

        print(
            "Existing PCA outputs do not align with the current cohort ordering; "
            "recomputing PCA in memory with scripts/02_eda.py logic."
        )
    else:
        print("Merged PCA outputs not fully present; recomputing PCA in memory.")

    scores, variance = compute_pca_outputs(feature_matrix, eda_module, covariate_cols)
    return scores, variance, "in_memory"


def padded_limits(values, pad_fraction=0.05):
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    if span == 0:
        span = 1.0
    pad = span * pad_fraction
    return lower - pad, upper + pad


def explained_variance_label(variance, pc_name):
    match = variance.loc[variance["PC"] == pc_name, "explained_variance_ratio"]
    if match.empty:
        raise KeyError(f"Missing explained variance ratio for {pc_name}")
    return float(match.iloc[0]) * 100.0


def draw_pca_scatter(ax, scores, group_col, palette):
    for group in palette:
        group_mask = scores[group_col] == group
        if not group_mask.any():
            continue
        ax.scatter(
            scores.loc[group_mask, "PC1"],
            scores.loc[group_mask, "PC2"],
            s=PCA_MARKER_SIZE,
            alpha=PCA_MARKER_ALPHA,
            color=palette[group],
            edgecolors="white",
            linewidth=PCA_MARKER_EDGEWIDTH,
            label=group,
        )


def style_pca_axis(ax, variance, title, xlim, ylim, legend_title):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel(f"PC1 ({explained_variance_label(variance, 'PC1'):.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance_label(variance, 'PC2'):.1f}% variance)")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(title=legend_title, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_pca(
    scores,
    variance,
    group_col,
    palette,
    title,
    output_path,
    xlim,
    ylim,
    legend_title,
):
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    draw_pca_scatter(ax=ax, scores=scores, group_col=group_col, palette=palette)
    style_pca_axis(
        ax=ax,
        variance=variance,
        title=title,
        xlim=xlim,
        ylim=ylim,
        legend_title=legend_title,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.relative_to(ROOT)}")


def plot_pca_comparison(scores, variance, xlim, ylim, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharex=True, sharey=True)

    draw_pca_scatter(
        ax=axes[0],
        scores=scores,
        group_col="PHASE",
        palette=PHASE_PALETTE,
    )
    style_pca_axis(
        ax=axes[0],
        variance=variance,
        title=PCA_PHASE_TITLE,
        xlim=xlim,
        ylim=ylim,
        legend_title="ADNI Phase",
    )

    draw_pca_scatter(
        ax=axes[1],
        scores=scores,
        group_col="cohort_group",
        palette=COHORT_PALETTE,
    )
    style_pca_axis(
        ax=axes[1],
        variance=variance,
        title=PCA_DIAGNOSIS_TITLE,
        xlim=xlim,
        ylim=ylim,
        legend_title="Diagnosis",
    )

    for label, ax in zip(["A", "B"], axes):
        ax.text(
            -0.12,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.relative_to(ROOT)}")


def select_representative_proteins(proteins, n=9):
    selected = [protein for protein in REPRESENTATIVE_PROTEINS if protein in proteins]
    if len(selected) < n:
        fillers = [protein for protein in proteins if protein not in selected]
        selected.extend(fillers[: n - len(selected)])
    return selected[:n]


def plot_protein_distribution_grid(feature_matrix, label_col, proteins, output_path):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharey=False)
    axes = axes.flatten()

    for ax, protein in zip(axes, proteins):
        for group, color in COHORT_PALETTE.items():
            subset = feature_matrix.loc[feature_matrix[label_col] == group, protein].dropna()
            if subset.nunique() < 2:
                continue
            density = gaussian_kde(subset.to_numpy())
            x_grid = np.linspace(subset.min(), subset.max(), 250)
            y_grid = density(x_grid)
            ax.plot(x_grid, y_grid, color=color, linewidth=1.6)
            ax.fill_between(x_grid, 0, y_grid, color=color, alpha=0.18)

        ax.set_title(textwrap.fill(protein, width=28), fontsize=11, fontweight="bold")
        ax.set_xlabel("NPQ (log2 scale)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(proteins) :]:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color=color, lw=2.4, label=group)
        for group, color in COHORT_PALETTE.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        "Selected Phase 2 Protein Distributions by Cohort Group",
        fontsize=16,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.relative_to(ROOT)}")


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    eda_module = load_phase2_module()
    summary = load_summary()
    covariate_cols = final_covariate_cols(summary)

    feature_matrix, feature_source = load_or_build_feature_matrix(eda_module)
    scores, variance, pca_source = load_or_compute_pca(
        feature_matrix=feature_matrix,
        eda_module=eda_module,
        covariate_cols=covariate_cols,
    )

    proteins = protein_feature_cols(
        feature_matrix=feature_matrix,
        label_col=eda_module.LABEL_COL,
        covariate_cols=covariate_cols,
        id_cols=eda_module.ID_COLS,
    )
    selected_proteins = select_representative_proteins(proteins)

    xlim = padded_limits(scores["PC1"])
    ylim = padded_limits(scores["PC2"])

    plot_pca(
        scores=scores,
        variance=variance,
        group_col="PHASE",
        palette=PHASE_PALETTE,
        title=PCA_PHASE_TITLE,
        output_path=PCA_PHASE_FIG,
        xlim=xlim,
        ylim=ylim,
        legend_title="ADNI Phase",
    )
    plot_pca(
        scores=scores,
        variance=variance,
        group_col="cohort_group",
        palette=COHORT_PALETTE,
        title=PCA_DIAGNOSIS_TITLE,
        output_path=PCA_COHORT_FIG,
        xlim=xlim,
        ylim=ylim,
        legend_title="Diagnosis",
    )
    plot_pca_comparison(
        scores=scores,
        variance=variance,
        xlim=xlim,
        ylim=ylim,
        output_path=PCA_COMPARISON_FIG,
    )
    plot_protein_distribution_grid(
        feature_matrix=feature_matrix,
        label_col=eda_module.LABEL_COL,
        proteins=selected_proteins,
        output_path=PROTEIN_GRID_FIG,
    )

    print()
    print(f"Feature matrix source: {feature_source}")
    print(f"PCA source: {pca_source}")
    print("Selected proteins plotted:")
    for protein in selected_proteins:
        print(f"  - {protein}")
    print("Generated figure files:")
    for output_path in [
        PCA_PHASE_FIG,
        PCA_COHORT_FIG,
        PCA_COMPARISON_FIG,
        PROTEIN_GRID_FIG,
    ]:
        print(f"  - {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
