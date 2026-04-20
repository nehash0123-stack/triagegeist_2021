# Triagegeist 🏥
### AI-Assisted Emergency Triage Prediction & Bias Analysis

> Submitted to the Laitinen-Fredriksson Foundation Triagegeist Competition (April 2026)

---

## Overview

This project builds a hybrid ML + NLP pipeline to predict ESI triage acuity levels from structured emergency department intake data, and surfaces systematic undertriage patterns across demographic and temporal subgroups.

**Dataset:** NHAMCS 2021 Emergency Department module (CDC) — freely available, no registration required.

**Key Results:**

| Metric | Value |
|--------|-------|
| Macro F1 | 0.384 |
| Adjacent Accuracy (±1 ESI) | 91.8% |
| Critical Sensitivity (ESI 1–2) | 55.1% |
| Overall Undertriage Rate | 8.1% |

**Core Finding:** Vital sign distributions overlap almost completely across ESI levels 1–5 — empirically demonstrating why structured-data triage AI has an inherent performance ceiling and why human inter-rater variability persists.

---

## Repository Structure

```
triagegeist/
├── README.md
├── triagegeist.py   ← Main Kaggle notebook
└── requirements.txt
```

---

## Setup Instructions

### 1. Dataset Download

Download the NHAMCS 2021 ED dataset from the CDC:

```
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS/
```

File needed: `ed2021_sas.sas7bdat` (~78MB)

No registration or credentialed access required.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

Open `triagegeist.py` in Kaggle or Jupyter and run all cells top to bottom.

Update the filepath in Cell 1 to point to your downloaded dataset:

```python
df = pd.read_sas('your/path/to/ed2021_sas.sas7bdat',
                 format='sas7bdat', encoding='latin1')
```

---

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
lightgbm>=4.0
xgboost>=2.0
optuna>=3.0
shap>=0.45
imbalanced-learn>=0.11
matplotlib>=3.7
seaborn>=0.12
pyreadstat>=1.2
```

---

## Pipeline Summary

```
NHAMCS 2021 ED (16,207 visits)
        ↓
  Clean → keep ESI 1–5 → 10,495 rows
        ↓
  Feature Engineering (531 total features)
  ├── Structured vitals + demographics (23)
  ├── Clinical risk scores: MEWS, Shock Index,
  │   Pulse Pressure, binary flags (8)
  └── TF-IDF on RFV complaint codes (500)
        ↓
  LightGBM + XGBoost Ensemble
  (class-balanced weights, Optuna tuning)
        ↓
  Evaluation: F1, Adjacent Accuracy,
  Critical Sensitivity, Bias Analysis, SHAP
```

---

## Key Findings

### 1. Vital Signs Overlap Across ESI Levels
Boxplot analysis shows near-complete distributional overlap of all vitals (HR, BP, RR, O2, Temp) across ESI 1–5. This is not a modeling failure — it is a data reality that explains the performance ceiling and validates why clinical gestalt matters in triage.

### 2. Systematic Undertriage Patterns

| Subgroup | Undertriage Rate |
|----------|-----------------|
| Elderly (65+) | 10.9% |
| Infants (0–2) | 10.0% |
| Daytime (8–16h) | 8.9% |
| Female | 8.2% |
| Male | 7.9% |

### 3. SHAP Feature Importance (ESI-1 Critical)
Top predictors: `WAITTIME`, `AGE×MEWS`, `AGE`, `PULSE_PRESSURE`, `HOUR`
Composite and contextual features outperform raw vitals — no single measurement reliably separates critical from non-urgent patients.

---

## Reproducibility

- All random seeds set to `42`
- `train_test_split` uses `stratify=y`
- Optuna tuning: minor variation across runs expected (±0.01 F1)
- Python 3.12, all package versions in `requirements.txt`

---

## Data Citation

National Center for Health Statistics. *National Hospital Ambulatory Medical Care Survey: 2021 Emergency Department Summary Tables.*
Available: https://www.cdc.gov/nchs/ahcd/

---

## License

This project is released for research and educational purposes.
Competition submission for Triagegeist — Laitinen-Fredriksson Foundation, April 2026.
