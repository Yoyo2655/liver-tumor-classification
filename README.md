# Liver Tumor Classification from Radiomic Features

**CentraleSupélec academic ML group project (2025–2026)**


## Overview

This group project explores machine-learning methods for distinguishing liver tumor classes from **tabular radiomic features** derived from medical imaging. The repository contains selected original Python research scripts, including exploratory analyses, classification, patient-level diagnostics, and evaluation plots. It does **not** contain raw medical images, patient records, feature tables, fitted models, or patient-specific reports.

This is **not** an end-to-end image classifier and does not use speech recognition or LLM-based medical report generation.

## Methods in the original code

- Tabular radiomic-feature preprocessing and exploratory analysis.
- Logistic regression with class balancing and a multilayer perceptron (scikit-learn).
- Patient-grouped stratified train/test splitting (`StratifiedGroupKFold`) to avoid mixing slices from the same patient across splits.
- Standardization fitted on training data; evaluation with accuracy, classification reports, confusion matrices, and other diagnostic plots.
- Additional research scripts for patient-level diagnostic aggregation and visual analysis.

**Evaluation caveat:** the training scripts select one split from a five-fold splitter; this is not equivalent to reporting five-fold cross-validation. No externally validated clinical performance is claimed.

## Repository structure

```text
code/              Original group research scripts (not refactored)
requirements.txt   Python dependencies identified from imports
.gitignore         Excludes data, model artifacts and generated outputs
```

## Running locally

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The scripts are **not runnable out of the box**: several expect restricted source spreadsheets in `dataset/` and write trained artifacts to `models/`. Those files are intentionally excluded. Consult each script for its expected input filename and schema. No synthetic replacement dataset or verified end-to-end execution is provided.

## Privacy, authorship and publication

This is a **collaborative CentraleSupélec student project**, not an individual project. 

No license is granted here; reuse rights require confirmation from the rights holders.
