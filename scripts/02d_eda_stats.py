"""
02d_eda_stats.py - Phase 2 descriptive figures and tables 
===================================================================

Descriptive half of roles-doc job 5. Reads only the locked Phase 1 cohort
(data/processed/adni_nulisa_cohort.csv), so it does not depend on the
preprocessing pipeline in 02_eda.py and can run before or after it.

Three deliverables:

  1. Missing-data heatmap, samples x proteins, overall AND per phase.
     Three-state: observed / below detection floor / missing. The floor state
     is shown because Phase 2 decision D3 masks NPQ < 5 to NaN before the
     missingness filter runs -- the raw NaN picture alone understates it.
  2. Class balance summary, CN/MCI overall and broken down by phase.
  3. Table 1, CN vs MCI on age, sex, education, APOE e4 and MMSE, with
     p-values.

Outputs
  reports/figures/stage2_missingness_heatmap.png
  reports/figures/stage2_missingness_heatmap_by_phase.png
  reports/figures/stage2_class_balance.png
  reports/table1_cn_vs_mci.md
  reports/table1_cn_vs_mci.csv
  reports/class_balance.csv
  reports/missingness_by_protein.csv
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "ai4ad-matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
COHORT_PATH = ROOT / "data/processed/adni_nulisa_cohort.csv"
FIG_DIR = ROOT / "reports/figures"
REPORT_DIR = ROOT / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MISSING_FIG = FIG_DIR / "stage2_missingness_heatmap.png"
MISSING_PHASE_FIG = FIG_DIR / "stage2_missingness_heatmap_by_phase.png"
CLASS_BALANCE_FIG = FIG_DIR / "stage2_class_balance.png"
TABLE1_MD = REPORT_DIR / "table1_cn_vs_mci.md"
TABLE1_CSV = REPORT_DIR / "table1_cn_vs_mci.csv"
CLASS_BALANCE_CSV = REPORT_DIR / "class_balance.csv"
MISSINGNESS_CSV = REPORT_DIR / "missingness_by_protein.csv"

# Mirrors 02_eda.py. Duplicated rather than imported so this script stays
# runnable off the Phase 1 cohort alone; asserted against the CSV below.
ID_COLS = ["RID", "PHASE"]
LABEL_COL = "cohort_group"
COVARIATE_COLS = ["AGE", "PTGENDER", "PTEDUCAT", "apoe4_count", "MMSCORE"]
DETECTION_FLOOR = 5.0     # [D3] NPQ below this is undetected, not measured
MISSING_THRESHOLD = 0.20  # [D2] applied per phase, not pooled
DROP_PROTEINS = ["Apolipoprotein E"]  # [D4] genotype proxy, removed before D2 runs

EXPECTED_SHAPE = (728, 124)
EXPECTED_N_PROTEINS = 116

PHASE_ORDER = ["ADNI1", "ADNIGO", "ADNI3"]
PHASE_PALETTE = {"ADNI1": "#4C78A8", "ADNIGO": "#F58518", "ADNI3": "#54A24B"}
GROUP_ORDER = ["CN", "MCI"]
GROUP_PALETTE = {"CN": "#4C78A8", "MCI": "#E45756"}

# Heatmap states: 0 observed, 1 below detection floor, 2 missing.
STATE_COLORS = ["#E8E8E8", "#F58518", "#B02418"]
STATE_LABELS = [
    "Observed",
    "Below detection floor (NPQ < %g)" % DETECTION_FLOOR,
    "Missing (NaN)",
]
DPI = 300


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_cohort():
    if not COHORT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {COHORT_PATH}. Run scripts/01_build_cohort.py first."
        )
    df = pd.read_csv(COHORT_PATH)
    print(f"Loaded cohort: {df.shape}")
    if df.shape != EXPECTED_SHAPE:
        print(f"  WARNING: expected shape {EXPECTED_SHAPE}. This is a different "
              "cohort than Phase 2 was verified against -- compare before "
              "trusting any of these numbers.")

    meta_cols = ID_COLS + [LABEL_COL] + COVARIATE_COLS
    missing_meta = [c for c in meta_cols if c not in df.columns]
    assert not missing_meta, (
        f"Cohort CSV is missing expected non-protein columns: {missing_meta}. "
        "Re-run scripts/01_build_cohort.py."
    )

    # AGE was silently empty in an earlier build; Table 1 is meaningless without it.
    for col in COVARIATE_COLS:
        n_present = int(df[col].notna().sum())
        if n_present < len(df):
            print(f"  WARNING: {col} present for only {n_present}/{len(df)} rows. "
                  "Table 1 reports the reduced n for that row.")

    protein_cols = [c for c in df.columns if c not in meta_cols]
    non_numeric = df[protein_cols].select_dtypes(exclude="number").columns.tolist()
    assert not non_numeric, (
        f"Non-numeric columns leaked into the protein matrix: {non_numeric}."
    )
    if len(protein_cols) != EXPECTED_N_PROTEINS:
        print(f"  WARNING: expected {EXPECTED_N_PROTEINS} proteins, found "
              f"{len(protein_cols)}.")
    print(f"Proteins: {len(protein_cols)} | covariates held out: {COVARIATE_COLS}")
    return df, protein_cols


def present_phases(df):
    seen = [p for p in PHASE_ORDER if p in set(df["PHASE"])]
    extra = sorted(set(df["PHASE"]) - set(PHASE_ORDER))
    if extra:
        print(f"  NOTE: unexpected PHASE values present, appended: {extra}")
    return seen + extra


# --------------------------------------------------------------------------- #
# 1. Missing-data heatmap
# --------------------------------------------------------------------------- #

def state_matrix(x):
    """0 = observed, 1 = below detection floor, 2 = missing."""
    values = x.to_numpy(dtype=float)
    states = np.zeros(values.shape, dtype=np.int8)
    states[values < DETECTION_FLOOR] = 1
    states[np.isnan(values)] = 2
    return states


def unusable_mask(x):
    """Cells the pipeline cannot use: NaN, or below the detection floor."""
    return x.isna() | (x < DETECTION_FLOOR)


def protein_order(x, phase):
    """Worst per-phase unusable rate first, which is the D2 filter's own view."""
    worst_per_phase = unusable_mask(x).groupby(phase).mean().max()
    return worst_per_phase.sort_values(ascending=False).index.tolist()


def short_label(name, width=42):
    return name if len(name) <= width else name[: width - 3] + "..."


def affected_proteins(x, phase, ordered):
    """Split proteins into those carrying any unusable cell and the clean rest.

    99 of 116 are clean in every phase. Drawing them as 99 near-identical gray
    rows spends most of the canvas on "nothing happened here" and squeezes the
    protein names down to unreadable rotated ticks, so the clean ones are
    summarised in one band instead.
    """
    worst = unusable_mask(x).groupby(phase).mean().max()
    affected = [p for p in ordered if worst[p] > 0]
    return affected, len(ordered) - len(affected)


def draw_states(ax, states, cmap):
    """states is proteins x samples: proteins read as rows, so names stay level."""
    ax.imshow(states, aspect="auto", interpolation="nearest", cmap=cmap,
              vmin=0, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])


def label_protein_rows(ax, proteins):
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels([short_label(p) for p in proteins], fontsize=8)
    ax.tick_params(axis="y", length=0, pad=3)


def legend_handles():
    return [Patch(facecolor=c, edgecolor="none", label=lab)
            for c, lab in zip(STATE_COLORS, STATE_LABELS)]


def sample_order(block):
    """Worst samples to the left within a phase, so structure reads left-right."""
    rates = unusable_mask(block).mean(axis=1)
    return rates.sort_values(ascending=False, kind="stable").index.tolist()


def clean_band(ax, n_clean):
    ax.set_facecolor(STATE_COLORS[0])
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.5, 0.5, f"{n_clean} further proteins: every value observed, "
                      "in all three phases",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.5,
            color="#555555")


def stacked_rate_bars(ax, x, proteins):
    """Split each protein's unusable rate into below-floor vs missing."""
    missing = (x[proteins].isna().mean() * 100).to_numpy()
    floor = ((x[proteins] < DETECTION_FLOOR).sum().to_numpy() / len(x)) * 100
    rows = np.arange(len(proteins))
    gap = 0.6  # surface gap between the two fills, in percentage points

    ax.barh(rows, missing, height=0.72, color=STATE_COLORS[2])
    ax.barh(rows, floor, height=0.72, left=np.where(floor > 0, missing + gap, missing),
            color=STATE_COLORS[1])
    for row, total in zip(rows, missing + floor):
        # A sub-1% rate is not zero; rounding it to "0" would read as clean.
        text = f"{total:.0f}" if total >= 1 else "<1"
        ax.text(total + 2, row, text, va="center", fontsize=7, color="#555555")

    ax.set_xlim(0, 118)
    ax.set_ylim(len(proteins) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100%"], fontsize=7.5)
    ax.tick_params(axis="x", length=0, pad=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.set_title("% of samples", fontsize=8, color="#555555", pad=6)


def plot_missingness_overall(df, ordered, phases):
    """Every sample, grouped by phase, with a per-protein rate bar alongside."""
    cmap = ListedColormap(STATE_COLORS)
    proteins, n_clean = affected_proteins(df[ordered], df["PHASE"], ordered)

    columns, bounds = [], []
    for ph in phases:
        columns.extend(sample_order(df.loc[df["PHASE"] == ph, proteins]))
        bounds.append(len(columns))
    states = state_matrix(df.loc[columns, proteins]).T

    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(
        3, 2, width_ratios=[4.6, 1], height_ratios=[0.07, 1, 0.11],
        wspace=0.015, hspace=0.06,
    )
    ax_phase, ax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_bar, ax_clean = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 0])

    start = 0
    for ph, end in zip(phases, bounds):
        ax_phase.axvspan(start, end, color=PHASE_PALETTE.get(ph, "#888888"), lw=0)
        ax_phase.text((start + end) / 2, 0.5, f"{ph}  n={end - start}",
                      ha="center", va="center", fontsize=8.5, color="white",
                      fontweight="bold")
        start = end
    ax_phase.set_xlim(0, len(columns))
    ax_phase.set_ylim(0, 1)
    ax_phase.axis("off")

    draw_states(ax, states, cmap)
    for b in bounds[:-1]:
        ax.axvline(b - 0.5, color="white", linewidth=1.4)
    label_protein_rows(ax, proteins)
    ax.set_xlabel(f"Participants (n={len(columns)}), sorted within phase",
                  fontsize=9, labelpad=6)

    stacked_rate_bars(ax_bar, df, proteins)
    clean_band(ax_clean, n_clean)
    fig.add_subplot(gs[0, 1]).axis("off")
    fig.legend(handles=legend_handles(), loc="lower left",
               bbox_to_anchor=(0.795, 0.0), frameon=False, fontsize=8.5)

    fig.suptitle("Missing and below-detection plasma protein measurements\n"
                 "ADNI NULISA baseline cohort, before imputation",
                 fontsize=12, y=1.04)
    fig.savefig(MISSING_FIG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {MISSING_FIG.relative_to(ROOT)}")


def plot_missingness_by_phase(df, ordered, phases):
    """Phases side by side on shared protein rows, so rows compare directly."""
    cmap = ListedColormap(STATE_COLORS)
    proteins, n_clean = affected_proteins(df[ordered], df["PHASE"], ordered)
    counts = [int((df["PHASE"] == ph).sum()) for ph in phases]

    fig, axes = plt.subplots(
        1, len(phases), figsize=(12, 5.2), sharey=True,
        gridspec_kw={"width_ratios": counts, "wspace": 0.02},
    )
    axes = np.atleast_1d(axes)

    for ax, ph, n in zip(axes, phases, counts):
        block = df.loc[df["PHASE"] == ph, proteins]
        block = block.loc[sample_order(block)]
        draw_states(ax, state_matrix(block).T, cmap)
        # Rate over all proteins, not just the rows drawn, so it matches the CSV.
        rate = unusable_mask(df.loc[df["PHASE"] == ph, ordered]).to_numpy().mean() * 100
        ax.set_title(f"{ph}\nn={n}  |  {rate:.1f}% of all cells unusable", fontsize=9,
                     color=PHASE_PALETTE.get(ph, "#333333"), fontweight="bold",
                     pad=6)

    label_protein_rows(axes[0], proteins)
    fig.legend(handles=legend_handles(), loc="upper left",
               bbox_to_anchor=(0.905, 0.86), frameon=False, fontsize=8.5)
    fig.suptitle("Protein data quality by ADNI phase (shared protein rows)",
                 fontsize=12, y=1.06)
    fig.text(0.5, -0.02,
             f"{n_clean} further proteins are fully observed in all three phases "
             "and are not shown.", ha="center", fontsize=8.5, color="#555555")
    fig.savefig(MISSING_PHASE_FIG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {MISSING_PHASE_FIG.relative_to(ROOT)}")


def missingness_table(df, protein_cols, phases):
    """Per protein: NaN rate and unusable rate, overall and per phase."""
    x = df[protein_cols]
    unusable = unusable_mask(x)
    out = pd.DataFrame({
        "protein": protein_cols,
        "pct_missing_overall": (x.isna().mean() * 100).reindex(protein_cols).to_numpy(),
        "pct_unusable_overall": (unusable.mean() * 100).reindex(protein_cols).to_numpy(),
    })
    per_phase_nan = x.isna().groupby(df["PHASE"]).mean() * 100
    per_phase_unusable = unusable.groupby(df["PHASE"]).mean() * 100
    for ph in phases:
        out[f"pct_missing_{ph}"] = per_phase_nan.loc[ph].reindex(protein_cols).to_numpy()
        out[f"pct_unusable_{ph}"] = per_phase_unusable.loc[ph].reindex(protein_cols).to_numpy()

    out["worst_phase_unusable"] = out[[f"pct_unusable_{ph}" for ph in phases]].max(axis=1)
    out["over_D2_threshold"] = out["worst_phase_unusable"] > MISSING_THRESHOLD * 100
    # 02_eda.py removes the D4 proteins before D2 runs, so they never reach the
    # filter. Flagging them is what reconciles this count with its 7.
    out["removed_by_D4_first"] = out["protein"].isin(DROP_PROTEINS)
    out["dropped_by_D2_filter"] = out["over_D2_threshold"] & ~out["removed_by_D4_first"]
    out = out.sort_values("worst_phase_unusable", ascending=False).reset_index(drop=True)
    out.round(3).to_csv(MISSINGNESS_CSV, index=False)
    print(f"  wrote {MISSINGNESS_CSV.relative_to(ROOT)}")
    return out


# --------------------------------------------------------------------------- #
# 2. Class balance
# --------------------------------------------------------------------------- #

def class_balance_table(df, phases):
    counts = (
        pd.crosstab(df["PHASE"], df[LABEL_COL])
        .reindex(index=phases, fill_value=0)
        .reindex(columns=GROUP_ORDER, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    counts.loc["All"] = counts.sum()
    counts["n"] = counts[GROUP_ORDER].sum(axis=1)
    counts["pct_MCI"] = (counts["MCI"] / counts["n"] * 100).round(1)
    counts.index.name = "PHASE"
    counts.reset_index().to_csv(CLASS_BALANCE_CSV, index=False)
    print(f"  wrote {CLASS_BALANCE_CSV.relative_to(ROOT)}")
    return counts


def plot_class_balance(counts, phases):
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 2.1]}
    )

    overall = counts.loc["All"]
    total = int(overall["n"])
    bars = ax_left.bar(GROUP_ORDER, [overall[g] for g in GROUP_ORDER],
                       color=[GROUP_PALETTE[g] for g in GROUP_ORDER], width=0.62)
    for bar, g in zip(bars, GROUP_ORDER):
        ax_left.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + total * 0.012,
                     f"{int(overall[g])}\n{overall[g] / total * 100:.1f}%",
                     ha="center", va="bottom", fontsize=9)
    ax_left.set_ylim(0, total * 0.75)
    ax_left.set_ylabel("Participants")
    ax_left.set_title(f"Overall class balance (n={total})", fontsize=10.5)

    width = 0.38
    positions = np.arange(len(phases))
    for offset, g in zip((-width / 2, width / 2), GROUP_ORDER):
        values = [int(counts.loc[ph, g]) for ph in phases]
        bars = ax_right.bar(positions + offset, values, width=width,
                            color=GROUP_PALETTE[g], label=g)
        for bar, v in zip(bars, values):
            ax_right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                          str(v), ha="center", va="bottom", fontsize=8.5)
    ax_right.set_xticks(positions)
    ax_right.set_xticklabels(
        [f"{ph}\nn={int(counts.loc[ph, 'n'])}, {counts.loc[ph, 'pct_MCI']:.0f}% MCI"
         for ph in phases]
    )
    tallest = int(counts.loc[phases, GROUP_ORDER].to_numpy().max())
    ax_right.set_ylim(0, max(tallest * 1.22, 1))
    ax_right.set_title("Class balance by ADNI phase", fontsize=10.5)
    ax_right.legend(frameon=False, fontsize=9)

    for ax in (ax_left, ax_right):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    fig.suptitle("CN vs MCI class balance, ADNI NULISA baseline cohort",
                 fontsize=12, y=1.02)
    if "ADNIGO" in counts.index and int(counts.loc["ADNIGO", "CN"]) == 0:
        fig.text(0.5, -0.06,
                 "ADNIGO enrolled early-MCI participants only (no new controls), "
                 "so phase and diagnosis are confounded.",
                 ha="center", fontsize=8.5, color="#444444")
    fig.savefig(CLASS_BALANCE_FIG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {CLASS_BALANCE_FIG.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# 3. Table 1
# --------------------------------------------------------------------------- #


def fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def row(label, cn="", mci="", test="", p=np.nan, p_welch=np.nan, p_mw=np.nan,
        smd=np.nan, n_cn=0, n_mci=0):
    return {"variable": label, "cn": cn, "mci": mci, "test": test, "p_value": p,
            "p_welch_t": p_welch, "p_mann_whitney": p_mw, "smd": smd,
            "n_cn": n_cn, "n_mci": n_mci}


def continuous_row(label, cn, mci, primary="welch"):
    """mean +/- SD per group, with Welch and Mann-Whitney p, plus an SMD."""
    cn, mci = pd.Series(cn).dropna(), pd.Series(mci).dropna()
    p_welch = float(stats.ttest_ind(cn, mci, equal_var=False).pvalue)
    p_mw = float(stats.mannwhitneyu(cn, mci, alternative="two-sided").pvalue)
    pooled_sd = np.sqrt((cn.var(ddof=1) + mci.var(ddof=1)) / 2)
    return row(
        label,
        cn=f"{cn.mean():.2f} +/- {cn.std(ddof=1):.2f}",
        mci=f"{mci.mean():.2f} +/- {mci.std(ddof=1):.2f}",
        test="Welch t-test" if primary == "welch" else "Mann-Whitney U",
        p=p_welch if primary == "welch" else p_mw,
        p_welch=p_welch,
        p_mw=p_mw,
        smd=(mci.mean() - cn.mean()) / pooled_sd if pooled_sd > 0 else np.nan,
        n_cn=len(cn),
        n_mci=len(mci),
    )


def categorical_row(label, cn_flag, mci_flag):
    """n (%) positive per group, chi-square on the 2x2 table."""
    cn_flag = pd.Series(cn_flag).dropna().astype(bool)
    mci_flag = pd.Series(mci_flag).dropna().astype(bool)
    table = np.array([
        [cn_flag.sum(), len(cn_flag) - cn_flag.sum()],
        [mci_flag.sum(), len(mci_flag) - mci_flag.sum()],
    ])
    prop_cn, prop_mci = cn_flag.mean(), mci_flag.mean()
    pooled = np.sqrt((prop_cn * (1 - prop_cn) + prop_mci * (1 - prop_mci)) / 2)
    return row(
        label,
        cn=f"{int(cn_flag.sum())} ({prop_cn * 100:.1f}%)",
        mci=f"{int(mci_flag.sum())} ({prop_mci * 100:.1f}%)",
        test="Chi-square",
        p=float(stats.chi2_contingency(table, correction=False).pvalue),
        smd=(prop_mci - prop_cn) / pooled if pooled > 0 else np.nan,
        n_cn=len(cn_flag),
        n_mci=len(mci_flag),
    )


def phase_rows(cn, mci, phases):
    counts = np.array([
        [int((cn["PHASE"] == ph).sum()) for ph in phases],
        [int((mci["PHASE"] == ph).sum()) for ph in phases],
    ])
    seen = counts.sum(axis=0) > 0
    p = float(stats.chi2_contingency(counts[:, seen], correction=False).pvalue)
    return [
        row(f"  {ph}, n (%)",
            cn=f"{counts[0, i]} ({counts[0, i] / len(cn) * 100:.1f}%)",
            mci=f"{counts[1, i]} ({counts[1, i] / len(mci) * 100:.1f}%)",
            test="Chi-square (3x2)" if i == 0 else "",
            p=p if i == 0 else np.nan,
            n_cn=len(cn), n_mci=len(mci))
        for i, ph in enumerate(phases)
    ]


def build_table1(df, phases):
    cn = df[df[LABEL_COL] == "CN"]
    mci = df[df[LABEL_COL] == "MCI"]
    unexpected_sex = set(df["PTGENDER"].dropna().unique()) - {1.0, 2.0}
    assert not unexpected_sex, f"Unexpected PTGENDER codes: {sorted(unexpected_sex)}"

    rows = [
        continuous_row("Age, years", cn["AGE"], mci["AGE"]),
        categorical_row("Female, n (%)", cn["PTGENDER"] == 2, mci["PTGENDER"] == 2),
        continuous_row("Education, years", cn["PTEDUCAT"], mci["PTEDUCAT"]),
        continuous_row("APOE e4 alleles", cn["apoe4_count"], mci["apoe4_count"],
                       primary="mannwhitney"),
        categorical_row("APOE e4 carrier (>=1 allele), n (%)",
                        cn["apoe4_count"] >= 1, mci["apoe4_count"] >= 1),
        continuous_row("MMSE", cn["MMSCORE"], mci["MMSCORE"], primary="mannwhitney"),
        row("ADNI phase", n_cn=len(cn), n_mci=len(mci)),
        *phase_rows(cn, mci, phases),
    ]

    table = pd.DataFrame(rows)
    table.to_csv(TABLE1_CSV, index=False)
    print(f"  wrote {TABLE1_CSV.relative_to(ROOT)}")
    return table, len(cn), len(mci)


NOTES_HEADING = "## Notes"


def write_table1_markdown(table, n_cn, n_mci):
    lines = [
        "# Table 1. Baseline characteristics, CN vs MCI",
        "",
        f"ADNI NULISA plasma baseline cohort, n = {n_cn + n_mci} "
        f"({n_cn} CN, {n_mci} MCI). Continuous variables are mean +/- SD. "
        "Generated by `scripts/02d_eda_stats.py` from "
        "`data/processed/adni_nulisa_cohort.csv`.",
        "",
        f"| Variable | CN (n={n_cn}) | MCI (n={n_mci}) | Test | p | SMD |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        p = "" if not np.isfinite(r["p_value"]) else fmt_p(r["p_value"])
        smd = "" if not np.isfinite(r["smd"]) else f"{r['smd']:+.2f}"
        # Sub-rows are indented with spaces in the CSV; markdown collapses those.
        name = r["variable"]
        indent = len(name) - len(name.lstrip(" "))
        name = "&nbsp;" * (indent * 2) + name.lstrip(" ")
        lines.append(
            f"| {name} | {r['cn']} | {r['mci']} | {r['test']} | {p} | {smd} |"
        )

    # Everything from the "## Notes" heading down is hand-written; carry it
    # through so a rerun refreshes the table without deleting the commentary.
    if TABLE1_MD.exists():
        existing = TABLE1_MD.read_text(encoding="utf-8")
        if NOTES_HEADING in existing:
            lines += ["", NOTES_HEADING + existing.split(NOTES_HEADING, 1)[1].rstrip()]

    TABLE1_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {TABLE1_MD.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #

def main():
    print("--- 1. LOADING COHORT ---")
    df, protein_cols = load_cohort()
    phases = present_phases(df)

    print("\n--- 2. MISSING-DATA HEATMAPS ---")
    ordered = protein_order(df[protein_cols], df["PHASE"])
    plot_missingness_overall(df, ordered, phases)
    plot_missingness_by_phase(df, ordered, phases)
    miss = missingness_table(df, protein_cols, phases)
    nan_rate = df[protein_cols].isna().to_numpy().mean() * 100
    unusable_rate = unusable_mask(df[protein_cols]).to_numpy().mean() * 100
    print(f"  cells missing (NaN): {nan_rate:.2f}% | missing or below floor: "
          f"{unusable_rate:.2f}%")
    n_over = int(miss["over_D2_threshold"].sum())
    n_d4 = int((miss["over_D2_threshold"] & miss["removed_by_D4_first"]).sum())
    print(f"  proteins over the {MISSING_THRESHOLD:.0%} per-phase threshold: {n_over} "
          f"({n_over - n_d4} dropped by D2; {n_d4} already removed by D4)")
    print("  worst five (worst per-phase unusable %):")
    for _, r in miss.head(5).iterrows():
        print(f"    {r['protein'][:46]:<46} {r['worst_phase_unusable']:5.1f}%")

    print("\n--- 3. CLASS BALANCE ---")
    counts = class_balance_table(df, phases)
    plot_class_balance(counts, phases)
    for ph in list(counts.index):
        row = counts.loc[ph]
        print(f"  {ph:<7} CN {int(row['CN']):>3} | MCI {int(row['MCI']):>3} | "
              f"n {int(row['n']):>3} | {row['pct_MCI']:5.1f}% MCI")

    print("\n--- 4. TABLE 1 ---")
    table, n_cn, n_mci = build_table1(df, phases)
    write_table1_markdown(table, n_cn, n_mci)
    for _, r in table.iterrows():
        if r["cn"] == "" and r["mci"] == "":
            continue
        p = "" if not np.isfinite(r["p_value"]) else fmt_p(r["p_value"])
        print(f"  {r['variable'][:38]:<38} CN {r['cn']:<16} MCI {r['mci']:<16} p={p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
