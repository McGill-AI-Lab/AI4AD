# Phase 2 Preprocessing Decisions

What happens between the Phase 1 cohort (`data/processed/adni_nulisa_cohort.csv`,
728 x 124) and the file we model on (`data/processed/phase2_feature_matrix.csv`,
728 x 116). All of it is in `scripts/02_eda.py`.

116 proteins in, 108 out. 1 dropped because it repeats the APOE gene data, 7
dropped for too much missing data. No participants dropped. 75 empty cells filled
in (0.10%).

| # | Decision | What we did |
|---|---|---|
| 1 | log2(x+1) transform | not applied, values are already log2 |
| 2 | Missing data filter | run per phase, not on all rows at once |
| 3 | Very low values | anything under 5 treated as missing |
| 4 | `Apolipoprotein E` | dropped |
| 5 | Scaling age and MMSE | not done here, Modeling scales inside the folds |
| 6 | ComBat | not applied, phase only explains 2.79% |

Decisions 1, 2 and 5 do not match the roles doc. Reasons below.

---

## 1. No log2 transform

NULISA NPQ values are already log2. The data shows it: min 5.07, median 12.51,
max 24.58. Untransformed protein levels do not look like that.

Taking the log again would turn 12.5 into 3.75 and 18.0 into 4.25. A gap of 5.5
becomes a gap of 0.5, so the differences between participants get squashed, and
those differences are what the model needs.

For the report: a change of 1 means the protein level doubled.

## 2. Missing data filter run per phase

The roles doc says drop any protein missing in more than 20% of participants. We
check that 20% inside each phase instead, and drop the protein if it fails in any
one of them. Checking all rows together drops 3 proteins. Per phase drops 7. The
4 extra ones:

| Protein | all rows | ADNI1 | ADNIGO | ADNI3 |
|---|---|---|---|---|
| Amyloid-beta precursor protein | 10.0% | 0.0% | 82.0% | 0.0% |
| Synaptosomal-associated protein 25 | 9.9% | 0.0% | 80.9% | 0.0% |
| Tyrosine 3-mono./tryptophan 5-mono. activation protein gamma | 12.2% | 0.0% | 41.6% | 15.3% |
| Hemoglobin subunit alpha | 16.8% | 12.0% | 12.4% | 22.1% |

The first three are the real problem. APP and SNAP25 are fully there in ADNI1 and
ADNI3 but about 80% missing in ADNIGO. Across all rows that averages to about 10%,
so they pass the 20% check. We then fill those gaps with the median, which puts
the same number into 80% of ADNIGO rows. Everyone in ADNIGO is MCI. So the model
could learn "APP equals the median" means "this person is MCI" - a shortcut we
created ourselves that would raise our AUC and mean nothing.

Dropping APP and SNAP25 costs us two proteins that matter in AD, and that belongs
in the report as a limitation. Keeping a protein whose missing values give away
the answer is worse.

Hemoglobin subunit alpha only just fails, in ADNI3, and only after decision 3. It
is a blood sample quality marker rather than a brain protein, so we lose nothing.

Both versions of the filter also drop Pleiotrophin (93.7% missing), 14-3-3 protein
zeta/delta (90.4%) and Interleukin-1 beta (25.0%).

## 3. Very low values treated as missing

The protein columns sit around 12.5, so a 0.0 does not mean there was no protein.
It means the test could not detect it. Left in, those values drag down the median
we use to fill gaps.

Anything under 5.0 is set to missing first: 229 values across 10 proteins, mostly
Hemoglobin subunit alpha (122), Pleiotrophin (33), GDF-15 (32) and
Chitotriosidase-1 (30). The cutoff of 5 sits below the 1st percentile of the data
we keep (7.99), so it only catches undetected values and does not cut into real
low readings.

This runs before the filter in decision 2, which is why Hemoglobin subunit alpha
ends up failing in ADNI3.

## 4. Apolipoprotein E dropped

This column has two separate clumps of values (0 to 3, and 10 to 16), and which
clump someone falls into depends on their APOE gene. Share of samples below the
detection cutoff by number of e4 alleles: 49.0% with 0, 3.8% with 1, 7.7% with 2.
The test only picks the protein up properly in e4 carriers, so the column is a
messier version of `apoe4_count`, which we already have from APOERES.

Two reasons to drop it. It repeats a column we already have in cleaner form, and
in Phase 4 SHAP would split the credit between the two. The pathway writeup could
then report ApoE as a top protein finding when it is really the gene result we had
before any protein data.

`apoe4_count` stays in, so we do not lose the genetic signal.

## 5. Age and MMSE left unscaled

The file we hand over is not scaled. Scaling with all 728 rows before
cross-validation builds the mean and standard deviation partly from rows that end
up in the test fold, so test data leaks into training. Small for two covariates,
not small for 108 proteins, and easy to avoid by scaling inside the pipeline:

```python
Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(penalty="l2"))])
```

L2 needs the proteins on the same scale somewhere, so this is about where it
happens, not whether. For tables and figures: AGE 72.62 +/- 7.12, MMSCORE
28.24 +/- 1.80.

Sex is recoded in the file, since that is just renaming: `PTGENDER` 1/2 becomes
`SEX_FEMALE` 0/1 (345 female, 383 male).

## 6. No ComBat

Phase explains 2.79% of the variation in the protein data. The roles doc sets the
cutoff at 10%, so we do not correct.

How it was measured: put the proteins on the same scale, run PCA, and for each
component check how much of its spread is explained by which phase a sample came
from. Weight each of those by how much of the total variation that component
carries, then add them up. Looking only at PC1 and PC2 would miss batch effects
sitting further down.

| PC | variation | explained by phase | explained by diagnosis |
|---|---|---|---|
| PC1 | 17.3% | 6.5% | 0.5% |
| PC2 | 12.5% | 3.8% | 0.2% |
| PC3 | 3.5% | 4.0% | 0.1% |
| PC4 | 3.4% | 4.5% | 0.2% |
| PC5 | 2.8% | 2.6% | 2.5% |
| all components, weighted | | 2.79% | 0.38% |

Every component is in `reports/pca_variance.csv`, per-sample scores in
`reports/pca_scores.csv`. Cedric is checking the 2.79% on his own, since the
ComBat call rests on it.

Two things to keep in mind even though the answer was no. Everyone in ADNIGO is
MCI, so correcting for phase partly means correcting for diagnosis. And Aim 4
tests whether a model trained on ADNI1+GO still works on ADNI3, which is the same
difference ComBat would remove. If we revisit this, save both versions of the file
and run Aim 4 on the uncorrected one.

---

## Other notes

Gaps are filled with the median of the whole column, not the median of that
person's phase. A per-phase median would write phase into the numbers, and through
ADNIGO that means writing diagnosis into them.

There is no ADNI2 in this data. Aim 4's "train on ADNI1+2, test on ADNI3" is really
ADNI1+GO (72% MCI) tested on ADNI3 (37% MCI). Those two groups have very different
amounts of MCI to begin with, which moves the AUC on its own, so a drop does not
automatically mean the proteins differ between phases.

## What the output file looks like

`data/processed/phase2_feature_matrix.csv`, 728 rows x 116 columns, one row per
participant:

| Column | Notes |
|---|---|
| `RID` | participant id |
| `PHASE` | ADNI1 / ADNIGO / ADNI3, for Aim 4 splits, not a model feature |
| `cohort_group` | CN / MCI |
| `AGE` | years, unscaled |
| `PTEDUCAT` | years |
| `apoe4_count` | 0 / 1 / 2 |
| `MMSCORE` | 0-30, unscaled |
| `SEX_FEMALE` | 0 male, 1 female |
| 108 protein columns | NPQ, log2 scale, no gaps |

No missing values anywhere. Rebuild with `python scripts/02_eda.py`; you get the
same file every time.
