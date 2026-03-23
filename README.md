# AI4AD: Predicting Mild Cognitive Impairment from Plasma Proteomics

**McGill AI Lab** · February 2026

A machine learning pipeline to classify **Cognitively Normal (CN)** vs **Mild Cognitive Impairment (MCI)** participants using plasma protein levels from the ADNI NULISA CNS panel. The goal is to evaluate whether blood-based biomarkers can improve early MCI screening beyond standard clinical assessments.

---

## What This Project Does

We build and compare three models:

1. **Clinical Baseline** — logistic regression on age, sex, APOE ε4, and MMSE score
2. **Proteomics Model** — logistic regression on ~113 plasma protein features
3. **Multimodal Model** — clinical covariates + proteomics combined

Models are evaluated using stratified 5-fold cross-validation with AUC-ROC as the primary metric. We also plan to run a survival analysis (Cox model) on MCI-to-AD conversion and a SHAP-based interpretability analysis to identify the most biologically relevant proteins.

---

## Project Structure

```
alzheimer-s-detection/
├── notebooks/              # Exploratory notebooks
├── plots/                  # Generated plots
├── reports/
│   └── figures/            # Final figures
├── scripts/
│   ├── models/             # Model scripts
│   ├── utils/              # Utility functions
│   ├── 00_inspect_columns.py
│   ├── 00_sanity_check.py
│   ├── 01_build_cohort.py
│   ├── 02_eda.py
│   └── 03_model.py
├── src/                    # Shared source modules
├── .gitignore
├── requirements.txt
└── README.md
```

> Raw ADNI data goes in a `data/raw/` folder locally — it is not committed to the repo.

---

## Getting Started

```bash
git clone <repo-url>
cd alzheimer-s-detection
pip install -r requirements.txt
```

Place your raw ADNI CSVs in `data/raw/`, then run the scripts in order starting from `01_build_cohort.py`.

---

*Data accessed under the ADNI data use agreement. Raw data is not stored in this repository.*
