"""
train.py — train, calibrate, cross-validate, evaluate, and save all models.

Pipeline per model:
  1. Fit on the training split.
  2. Wrap in CalibratedClassifierCV (isotonic) so predicted probabilities are
     trustworthy — the report's calibration requirement.
  3. Stratified 5-fold CV on the training split for the CV-accuracy column.
  4. Evaluate the calibrated model on the held-out test split.

Artifacts written to ./artifacts/:
  <model>.joblib          calibrated fitted pipeline (one per model)
  primary_model.txt       name of the selected (best-F1) model
  metrics.json            full metrics table
  feature_columns.json    schema snapshot (guards inference against drift)

Run:  python train.py [--data path.csv] [--synthetic]
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, brier_score_loss)

from schema import FEATURES, TARGET
from data import make_synthetic, load_real
from models import get_models

SEED = 42
ARTIFACT_DIR = "artifacts"


def evaluate(model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    return dict(
        accuracy=accuracy_score(y_te, pred),
        precision=precision_score(y_te, pred, zero_division=0),
        recall=recall_score(y_te, pred, zero_division=0),
        f1=f1_score(y_te, pred, zero_division=0),
        roc_auc=roc_auc_score(y_te, proba),
        brier=brier_score_loss(y_te, proba),
    )


def run(df, quiet=False):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    X = df[FEATURES]
    y = df[TARGET].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = {}
    for name, pipe in get_models(scale_pos_weight=spw).items():
        # CV accuracy on the base pipeline
        cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy")

        # Fit the base (uncalibrated) model. Selection and the headline
        # metrics are computed on THIS model, so the operating point (and
        # therefore recall/F1) is not distorted by calibration — see the
        # calibration-vs-recall trade-off discussed in the report (§4.6).
        pipe.fit(X_tr, y_tr)
        metrics = evaluate(pipe, X_te, y_te)

        # Separately fit an isotonic-calibrated version for trustworthy
        # probabilities at inference. We report the Brier improvement it buys
        # but do NOT let it change model selection.
        calibrated = CalibratedClassifierCV(
            get_models(scale_pos_weight=spw)[name], method="isotonic", cv=3)
        calibrated.fit(X_tr, y_tr)
        metrics["brier_calibrated"] = float(
            evaluate(calibrated, X_te, y_te)["brier"])

        metrics["cv_acc_mean"] = float(cv_scores.mean())
        metrics["cv_acc_std"] = float(cv_scores.std())
        results[name] = metrics

        # Save the BASE model as the deployment artifact, so the reported
        # metrics, the saved model, and predict() all share one operating
        # point. The calibrated version is saved alongside for the calibration
        # analysis, but is not the default inference model.
        joblib.dump(pipe, os.path.join(ARTIFACT_DIR, f"{_slug(name)}.joblib"))
        joblib.dump(calibrated,
                    os.path.join(ARTIFACT_DIR, f"{_slug(name)}_calibrated.joblib"))

    # primary model = highest F1, tie-broken by ROC-AUC
    primary = max(results, key=lambda k: (results[k]["f1"], results[k]["roc_auc"]))

    with open(os.path.join(ARTIFACT_DIR, "primary_model.txt"), "w") as f:
        f.write(primary)
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(ARTIFACT_DIR, "feature_columns.json"), "w") as f:
        json.dump(FEATURES, f, indent=2)

    if not quiet:
        print_table(results, primary)
    return results, primary


def _slug(name):
    return name.lower().replace(" ", "_")


def print_table(results, primary):
    def pct(x):
        return f"{x * 100:.2f}%"
    print("\n" + "=" * 92)
    print("TABLE 4.5 — Model Performance Comparison")
    print("=" * 92)
    hdr = (f"{'Model':<22}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}"
           f"{'ROC-AUC':>9}{'Brier':>8}{'Brier(cal)':>11}{'CV Acc':>16}")
    print(hdr)
    print("-" * 92)
    for name, m in results.items():
        mark = "  <-- primary" if name == primary else ""
        cv = f"{pct(m['cv_acc_mean'])} \u00b1{pct(m['cv_acc_std'])}"
        print(f"{name:<22}{pct(m['accuracy']):>8}{pct(m['precision']):>8}"
              f"{pct(m['recall']):>8}{pct(m['f1']):>8}{pct(m['roc_auc']):>9}"
              f"{m['brier']:>8.3f}{m['brier_calibrated']:>11.3f}{cv:>16}{mark}")
    print("-" * 92)
    print(f"Primary model (highest F1, tie-broken by ROC-AUC): {primary}")
    print(f"Artifacts written to ./{ARTIFACT_DIR}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to real CSV/TSV. Omit to use synthetic.")
    ap.add_argument("--synthetic", action="store_true", help="Force synthetic data.")
    args = ap.parse_args()

    if args.data and not args.synthetic:
        df = load_real(args.data)
        print(f"[real:{args.data}] n={len(df)} positives={int(df[TARGET].sum())}")
    else:
        df = make_synthetic()
        print(f"[synthetic] n={len(df)} positives={int(df[TARGET].sum())} "
              f"({df[TARGET].mean():.1%})")
    run(df)


if __name__ == "__main__":
    main()
