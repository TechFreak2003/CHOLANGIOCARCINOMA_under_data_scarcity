import os
import numpy as np
import pytest

from schema import FEATURES, TARGET, risk_band
from data import make_synthetic, load_real
from models import get_models
import train
import predict as predict_mod


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train once on synthetic data into a temp artifact dir."""
    d = tmp_path_factory.mktemp("artifacts")
    train.ARTIFACT_DIR = str(d)
    predict_mod.ARTIFACT_DIR = str(d)
    predict_mod._CACHE.clear()
    df = make_synthetic(n=600, seed=42)
    results, primary = train.run(df, quiet=True)
    return results, primary


# ---------- data ----------

def test_synthetic_shape():
    df = make_synthetic(n=500, seed=1)
    assert list(df.columns) == FEATURES + [TARGET]
    assert len(df) == 500
    assert df[TARGET].isin([0, 1]).all()

def test_synthetic_has_both_classes():
    df = make_synthetic(n=500, seed=1)
    assert df[TARGET].nunique() == 2, "need both classes for a classifier"

def test_synthetic_prevalence_in_range():
    df = make_synthetic(n=2000, prevalence=0.18, seed=3)
    assert 0.10 < df[TARGET].mean() < 0.28

def test_load_real_rejects_bad_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("age,foo\n1,2\n")
    with pytest.raises(ValueError):
        load_real(str(bad))

def test_load_real_roundtrip(tmp_path):
    df = make_synthetic(n=50, seed=2)
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    loaded = load_real(str(p))
    assert list(loaded.columns) == FEATURES + [TARGET]
    assert len(loaded) == 50


# ---------- models ----------

def test_registry_has_five_models():
    m = get_models()
    assert len(m) == 5, f"expected 5 candidates, got {list(m)}"

def test_models_include_required():
    m = get_models()
    for name in ["Logistic Regression", "Decision Tree", "Random Forest",
                 "XGBoost", "LightGBM"]:
        assert name in m


# ---------- training ----------

def test_training_produces_all_metrics(trained):
    results, primary = trained
    assert len(results) == 5
    for name, m in results.items():
        for key in ["accuracy", "precision", "recall", "f1",
                    "roc_auc", "brier", "cv_acc_mean", "cv_acc_std"]:
            assert key in m
            assert not np.isnan(m[key])

def test_metrics_in_valid_ranges(trained):
    results, _ = trained
    for m in results.values():
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert 0.0 <= m[key] <= 1.0
        assert 0.0 <= m["brier"] <= 1.0

def test_primary_is_best_f1(trained):
    results, primary = trained
    best = max(results, key=lambda k: (results[k]["f1"], results[k]["roc_auc"]))
    assert primary == best

def test_artifacts_written(trained):
    assert os.path.exists(os.path.join(train.ARTIFACT_DIR, "primary_model.txt"))
    assert os.path.exists(os.path.join(train.ARTIFACT_DIR, "metrics.json"))


# ---------- prediction ----------

def test_predict_returns_contract(trained):
    predict_mod._CACHE.clear()
    pt = dict(age=68, sex=1, bmi=29, se_asian=1, alcohol_units=20, smoker=1,
              psc=1, liver_fluke=0, hbv=0, hcv=0, cirrhosis=1, gallstones=1,
              t2dm=1, fam_bileduct=1, alp=340, ggt=290, alt=45, ast=48,
              bilirubin=4.2, ca19_9=310, albumin=3.2)
    r = predict_mod.predict(pt)
    assert 0.0 <= r["probability"] <= 1.0
    assert r["risk_band"] in ["Low", "Moderate", "High", "Very High"]
    assert len(r["all_models"]) == 5
    assert len(r["explanation"]) > 0

def test_predict_handles_partial_input(trained):
    predict_mod._CACHE.clear()
    # only a few fields supplied; rest imputed
    r = predict_mod.predict(dict(age=70, psc=1, alp=300))
    assert 0.0 <= r["probability"] <= 1.0

def test_high_risk_scores_above_low_risk(trained):
    predict_mod._CACHE.clear()
    high = dict(age=70, sex=1, bmi=30, se_asian=1, alcohol_units=25, smoker=1,
                psc=1, liver_fluke=1, hbv=1, hcv=0, cirrhosis=1, gallstones=1,
                t2dm=1, fam_bileduct=1, alp=400, ggt=350, alt=50, ast=52,
                bilirubin=6.0, ca19_9=500, albumin=3.0)
    low = dict(age=30, sex=0, bmi=21, se_asian=0, alcohol_units=0, smoker=0,
               psc=0, liver_fluke=0, hbv=0, hcv=0, cirrhosis=0, gallstones=0,
               t2dm=0, fam_bileduct=0, alp=70, ggt=20, alt=18, ast=20,
               bilirubin=0.5, ca19_9=8, albumin=4.6)
    assert predict_mod.predict(high, explain=False)["probability"] > \
           predict_mod.predict(low, explain=False)["probability"]


# ---------- schema ----------

def test_risk_bands():
    assert risk_band(0.05) == "Low"
    assert risk_band(0.25) == "Moderate"
    assert risk_band(0.55) == "High"
    assert risk_band(0.90) == "Very High"
