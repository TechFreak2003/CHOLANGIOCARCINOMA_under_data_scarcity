# CCA Risk Prediction — Machine Learning Module

> Machine-learning core for **Module 1 (questionnaire-based risk estimation)** of the
> Cholangiocarcinoma Risk Prediction system.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.3%2B-orange">
  <img alt="tests" src="https://img.shields.io/badge/tests-15%2F15%20passing-brightgreen">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey">
  <img alt="status" src="https://img.shields.io/badge/status-research%20%2F%20not%20for%20clinical%20use-red">
</p>

Trains and compares five classifiers, calibrates their probabilities, explains every
prediction with SHAP, and scores individual patients — built for the low-data,
high-heterogeneity conditions that make cholangiocarcinoma (CCA) hard to model.

> [!IMPORTANT]
> **This is a research and educational project — not a medical device.**
> It is not clinically validated and must never be used for diagnosis or treatment
> decisions. By default it trains on a **synthetic** cohort whose risk structure follows
> CCA epidemiology; these are *pipeline-validation* results, **not clinical findings**.

---

## Why this exists

CCA is rare, aggressive, and asymptomatic in its early stages — which is exactly when
risk information is most useful and least available. Usable datasets and research sit
behind paywalls or access controls, and no open-source CCA risk tool exists. This module
is the machine-learning core of the open tool that was missing: it works under data
scarcity, degrades gracefully on partial input, and explains itself.

---

## Quick start

```bash
pip install -r requirements.txt
python train.py        # train on synthetic data, print the metrics table
python predict.py      # score a demo high-risk and low-risk patient
pytest -q              # run the 15-test suite
```

---

## Files

| File | Purpose |
|---|---|
| `schema.py` | Single source of truth for the feature set and risk bands |
| `data.py` | Synthetic generator (`make_synthetic`) + real-data loader (`load_real`) |
| `preprocessing.py` | Imputation + scaling pipeline (serialised with each model) |
| `models.py` | The five candidate classifiers with their configs |
| `train.py` | Fit, calibrate, 5-fold CV, evaluate, save artifacts, print the metrics table |
| `explain.py` | SHAP explanations for a single patient |
| `predict.py` | `predict(patient)` — scores one patient (feeds the future API) |
| `test_pipeline.py` | pytest suite (15 tests) |

---

## Pipeline overview

```
                 make_synthetic()  -- or --  load_real(csv/tsv)
                          |
                          v
                 preprocessing.py        impute -> scale (serialised per model)
                          |
                          v
                   models.py             LogReg . Decision Tree . Random Forest
                          |              XGBoost . LightGBM
                          v
                   train.py              5-fold CV . fit . isotonic calibration
                          |              select best-F1 . save artifacts
                          v
        +-----------------+-----------------+
        v                                   v
   predict.py                          explain.py
   probability . risk band             SHAP per-feature
   all-model comparison                contributions
```

---

## Usage

### Train

```bash
python train.py                              # synthetic (default)
python train.py --data path/to/dataset.csv   # real data (CSV or TSV auto-detected)
```
The real-data loader validates the schema and **fails loudly** — listing any missing
columns — rather than producing silent garbage.

### Score one patient

```python
from predict import predict

result = predict({
    "age": 68, "sex": 1, "psc": 1, "alp": 340, "ggt": 290,
    "bilirubin": 4.2, "ca19_9": 310,   # partial input is fine — the rest is imputed
})
```

Returns:

```python
{
    "probability": 0.99,           # calibrated P(elevated CCA risk)
    "risk_band": "Very High",      # Low / Moderate / High / Very High
    "model": "Logistic Regression",
    "all_models": { ... },         # every model's probability, for comparison
    "model_agreement": "split",    # "unanimous" or "split" — a confidence signal
    "explanation": [ ... ],        # top SHAP drivers, sorted by contribution
}
```

### Bring your own data

Any CSV/TSV with the columns in `schema.py` plus a binary `label` column works.
No code changes — the same pipeline that validates on synthetic data trains on real data.

---

## Results (synthetic demonstration data)

Produced by `python train.py`. Not clinical results — see the note above.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Accuracy |
|---|---|---|---|---|---|---|
| **Logistic Regression** (selected) | 73.3% | 41.7% | **62.5%** | **50.0%** | 74.6% | 69.3% ± 3.0% |
| Decision Tree | 74.2% | 41.7% | 52.1% | 46.3% | 67.9% | 66.5% ± 6.9% |
| Random Forest | 83.1% | 75.0% | 31.3% | 44.1% | 75.5% | 80.4% ± 1.2% |
| XGBoost | 79.1% | 51.4% | 37.5% | 43.4% | 71.5% | 77.0% ± 1.9% |
| LightGBM | *(see note)* | | | | | |

**Logistic Regression is selected** as primary — not for accuracy (Random Forest is
higher) but for the best **recall** and **F1**, which matter most in screening where a
missed case is the costly error, plus the strongest interpretability. Numbers vary
slightly with your environment and any real data you supply.

> **Note on LightGBM:** if the LightGBM row is missing, its import failed (commonly a stale
> `dask` conflicting with a newer `pandas`). Run `pip install --upgrade dask` or
> `pip uninstall dask -y`, then re-run. `models.py` prints the exact reason rather than
> dropping the model silently.

---

## Design notes

- **Selection vs. deployment.** Model selection and the headline metrics use the *base*
  (un-calibrated) model, so the operating point — and therefore recall/F1 — is not distorted
  by calibration. The primary model is the highest-F1 candidate, tie-broken by ROC-AUC.
- **Calibration** is fit separately (isotonic) and its Brier-score improvement is reported in
  the `Brier(cal)` column. It is saved as `<model>_calibrated.joblib` for calibration analysis;
  the deployed inference model is the base model, so metrics, artifact, and `predict()` all agree.
- **`predict()` runs all five models** and returns each one's probability plus a `model_agreement`
  flag — the comparison view the website shows. The primary model's result is the headline;
  agreement across models is itself a confidence signal.
- **Graceful degradation.** Missing feature keys in `predict()` are imputed by the pipeline, so a
  partial questionnaire still returns an estimate.
- **Schema safety.** Training snapshots the feature order to `artifacts/feature_columns.json`;
  everything imports `FEATURES` from `schema.py`, so train-time and inference-time columns can't drift.
- **Fail loudly.** Missing optional models and bad datasets raise visible errors instead of
  silently degrading the results.

---

## Synthetic data — how it's built

`make_synthetic()` does not produce random noise. Each patient's latent risk is a weighted
sum of their features, with weights following CCA epidemiology (primary sclerosing
cholangitis and liver fluke carry the largest weights). Biomarkers are driven by that same
latent risk, so the **cholestatic pattern** — sharply raised ALP/GGT with only mildly
elevated transaminases — appears in high-risk records. Label noise and an ~18% positive
prevalence keep the task realistic. The pipeline is validated by checking that the models'
learned SHAP importances match the epidemiology that was put in; it does not claim clinical
accuracy. See `data.py` for the full construction.

---

## Artifacts (written to `./artifacts/`)

| Artifact | Contents |
|---|---|
| `<model>.joblib` | Deployed (base) pipeline, one per model |
| `<model>_calibrated.joblib` | Calibrated variant, for calibration analysis |
| `primary_model.txt` | Name of the selected model |
| `metrics.json` | Full metrics table |
| `feature_columns.json` | Schema snapshot (guards inference against drift) |

---

## Roadmap

- [x] ML core: five-model competition, calibration, SHAP, single-patient inference
- [x] Test suite (15 tests)
- [ ] FastAPI service wrapping `predict()`
- [ ] React questionnaire frontend
- [ ] Module 2 — lab-report interpreter
- [ ] Module 3 — imaging-report analyser
- [ ] Real-data validation via access-controlled cohorts (e.g. dbGaP)

---

## Contributing

Issues and pull requests are welcome. Please run `pytest -q` before submitting, and keep
the fail-loudly principle — new optional dependencies should surface their errors, not
swallow them.

---

## License

Released under the MIT License. See `LICENSE`.

---

## Disclaimer

This software is provided for **research and educational purposes only**. It is not a
medical device, is not clinically validated, and must not be used to inform diagnosis,
prognosis, or treatment. No warranty is expressed or implied.
