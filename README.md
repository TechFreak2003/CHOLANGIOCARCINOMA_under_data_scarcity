# CCA Risk Prediction — Machine Learning Module

The machine-learning core for **Module 1 (questionnaire-based risk estimation)** of the
Cholangiocarcinoma Risk Prediction system. Trains and compares five classifiers, calibrates
their probabilities, explains predictions with SHAP, and scores individual patients.

> **Data note.** By default this trains on a **synthetic** cohort whose risk structure follows
> CCA epidemiology. These are *pipeline-validation* results, **not clinical findings**. To use
> real data, pass a CSV/TSV with the same schema (see `schema.py`).

## Files

| File | Purpose |
|---|---|
| `schema.py` | Single source of truth for the feature set and risk bands |
| `data.py` | Synthetic generator (`make_synthetic`) + real-data loader (`load_real`) |
| `preprocessing.py` | Imputation + scaling pipeline (serialised with each model) |
| `models.py` | The five candidate classifiers with their Table 4.1 configs |
| `train.py` | Fit, calibrate, 5-fold CV, evaluate, save artifacts, print Table 4.5 |
| `explain.py` | SHAP explanations for a single patient |
| `predict.py` | `predict(patient)` — scores one patient (feeds the future API) |
| `test_pipeline.py` | pytest suite (15 tests) |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train on synthetic data and print the metrics table:**
```bash
python train.py
```

**Train on real data:**
```bash
python train.py --data path/to/your_dataset.csv
```
The loader auto-detects CSV vs TSV and fails loudly if any schema column is missing.

**Score one patient:**
```bash
python predict.py          # runs a high-risk and low-risk demo
```
Or from Python / a future API endpoint:
```python
from predict import predict

result = predict({
    "age": 68, "sex": 1, "psc": 1, "alp": 340, "ggt": 290,
    "bilirubin": 4.2, "ca19_9": 310,   # partial input is fine — rest imputed
})
# -> {"probability": ..., "risk_band": ..., "model": ...,
#     "all_models": {...}, "model_agreement": ..., "explanation": [...]}
```

**Run the tests:**
```bash
pytest -q
```

## Design notes

- **Selection vs. deployment.** Model selection and the headline metrics use the *base*
  (un-calibrated) model, so the operating point — and therefore recall/F1 — is not distorted
  by calibration. The primary model is the highest-F1 candidate, tie-broken by ROC-AUC.
- **Calibration** is fit separately (isotonic) and its Brier-score improvement is reported in
  the `Brier(cal)` column. It is saved as `<model>_calibrated.joblib` for calibration analysis;
  the deployed inference model is the base model, so metrics/artifact/`predict()` all agree.
- **`predict()` runs all five models** and returns each one's probability plus a `model_agreement`
  flag — the comparison view the website shows. The primary model's result is the headline.
- **Graceful degradation.** Missing feature keys in `predict()` are imputed by the pipeline, so a
  partial questionnaire still returns an estimate.
- **Schema safety.** Training snapshots the feature order to `artifacts/feature_columns.json`;
  everything imports `FEATURES` from `schema.py`, so train-time and inference-time columns can't drift.

## Artifacts (written to `./artifacts/`)

- `<model>.joblib` — deployed (base) pipeline, one per model
- `<model>_calibrated.joblib` — calibrated variant, for analysis
- `primary_model.txt` — name of the selected model
- `metrics.json` — full metrics table
- `feature_columns.json` — schema snapshot
