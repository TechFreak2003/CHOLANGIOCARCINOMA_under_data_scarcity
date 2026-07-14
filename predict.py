"""
predict.py — score one patient. This is the function the future FastAPI
endpoint will call.

predict(patient: dict, model=None) -> dict with:
  probability   calibrated P(elevated CCA risk)
  risk_band     Low / Moderate / High / Very High
  model         which model produced it
  explanation   top SHAP feature contributions (list)
  all_models    every model's probability (for the site's comparison view)

Missing feature keys are allowed — they're filled with NaN and imputed by the
pipeline, so a partial questionnaire still yields an estimate (graceful
degradation, per the report).
"""

import os
import json
import glob

import numpy as np
import pandas as pd
import joblib

from schema import FEATURES, risk_band
from data import make_synthetic
from explain import explain_patient

ARTIFACT_DIR = "artifacts"
_CACHE = {}


def _load_all():
    """Load every saved model + the primary name. Cached after first call."""
    if _CACHE:
        return _CACHE
    primary_path = os.path.join(ARTIFACT_DIR, "primary_model.txt")
    if not os.path.exists(primary_path):
        raise FileNotFoundError(
            "No artifacts found. Run `python train.py` first.")
    with open(primary_path) as f:
        primary = f.read().strip()

    models = {}
    for path in glob.glob(os.path.join(ARTIFACT_DIR, "*.joblib")):
        base = os.path.splitext(os.path.basename(path))[0]
        if base.endswith("_calibrated"):
            continue  # calibrated variants are for analysis, not the model list
        models[_unslug(base)] = joblib.load(path)

    _CACHE["models"] = models
    _CACHE["primary"] = primary
    # small background sample for non-tree SHAP explainers
    _CACHE["background"] = make_synthetic(n=100, seed=7)[FEATURES]
    return _CACHE


def _unslug(slug):
    return {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
    }.get(slug, slug)


def _row(patient: dict) -> pd.DataFrame:
    """Build a single-row DataFrame, NaN for anything the caller omitted."""
    data = {f: [patient.get(f, np.nan)] for f in FEATURES}
    return pd.DataFrame(data)[FEATURES]


def predict(patient: dict, model: str = None, explain: bool = True) -> dict:
    ctx = _load_all()
    models = ctx["models"]
    primary = model or ctx["primary"]
    row = _row(patient)

    # every model's probability, for the comparison view
    all_probs = {}
    for name, m in models.items():
        all_probs[name] = float(m.predict_proba(row)[:, 1][0])

    p = all_probs[primary]
    result = {
        "probability": round(p, 4),
        "risk_band": risk_band(p),
        "model": primary,
        "all_models": {k: round(v, 4) for k, v in all_probs.items()},
        "model_agreement": _agreement(all_probs),
    }
    if explain:
        result["explanation"] = explain_patient(
            models[primary], row, background=ctx["background"])[:8]
    return result


def _agreement(all_probs):
    """Report whether models agree on the band — a confidence signal."""
    bands = {risk_band(p) for p in all_probs.values()}
    return "unanimous" if len(bands) == 1 else "split"


if __name__ == "__main__":
    # demo: a high-risk and a low-risk patient
    high = dict(age=68, sex=1, bmi=29, se_asian=1, alcohol_units=20, smoker=1,
                psc=1, liver_fluke=0, hbv=0, hcv=0, cirrhosis=1, gallstones=1,
                t2dm=1, fam_bileduct=1,
                alp=340, ggt=290, alt=45, ast=48, bilirubin=4.2,
                ca19_9=310, albumin=3.2)
    low = dict(age=35, sex=0, bmi=22, se_asian=0, alcohol_units=1, smoker=0,
               psc=0, liver_fluke=0, hbv=0, hcv=0, cirrhosis=0, gallstones=0,
               t2dm=0, fam_bileduct=0,
               alp=75, ggt=25, alt=20, ast=22, bilirubin=0.6,
               ca19_9=12, albumin=4.5)
    for label, pt in [("HIGH-RISK", high), ("LOW-RISK", low)]:
        r = predict(pt)
        print(f"\n{label}: p={r['probability']}  band={r['risk_band']}  "
              f"model={r['model']}  agreement={r['model_agreement']}")
        print("  all models:", r["all_models"])
        print("  top drivers:")
        for e in r["explanation"][:4]:
            arrow = "↑" if e["shap"] > 0 else "↓"
            print(f"    {arrow} {e['label']}: {e['value']}  (shap {e['shap']:+.3f})")
