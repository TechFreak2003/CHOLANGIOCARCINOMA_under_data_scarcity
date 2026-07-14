"""
explain.py — SHAP explanations for a single patient.

The primary model is a CalibratedClassifierCV wrapping a Pipeline(pre -> clf).
SHAP needs the preprocessed feature matrix and the underlying classifier, so
we reach through the calibration/pipeline wrappers to get both.

explain_patient(row_df) -> list of {feature, label, value, shap} sorted by
absolute contribution, so the UI can show the top drivers of a prediction.
"""

import numpy as np
import shap

from schema import FEATURES, FEATURE_LABELS


def _unwrap(model):
    """Return (preprocessor, classifier) from either a Pipeline or a
    CalibratedClassifierCV wrapping a Pipeline."""
    if hasattr(model, "named_steps"):            # plain Pipeline (deployed model)
        pipe = model
    elif hasattr(model, "calibrated_classifiers_"):   # CalibratedClassifierCV
        pipe = model.calibrated_classifiers_[0].estimator
    else:
        raise TypeError(f"Cannot unwrap model of type {type(model)}")
    return pipe.named_steps["pre"], pipe.named_steps["clf"]


def explain_patient(calibrated, row_df, background=None):
    """Explain one patient's prediction with SHAP.

    calibrated : fitted CalibratedClassifierCV (the loaded primary model)
    row_df     : single-row DataFrame with FEATURES columns
    background : optional DataFrame for the SHAP background distribution;
                 required for linear/kernel explainers, ignored by TreeExplainer
    """
    pre, clf = _unwrap(calibrated)
    x = pre.transform(row_df[FEATURES])

    # Choose an explainer appropriate to the model family
    model_name = type(clf).__name__
    try:
        if model_name in ("RandomForestClassifier", "XGBClassifier",
                           "LGBMClassifier", "DecisionTreeClassifier"):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(x)
            vals = _positive_class(sv)
        else:
            # linear / other: use a background sample
            bg = pre.transform(background[FEATURES]) if background is not None else x
            explainer = shap.LinearExplainer(clf, bg)
            vals = np.asarray(explainer.shap_values(x))
    except Exception:
        # robust fallback: model-agnostic permutation explainer
        bg = pre.transform(background[FEATURES]) if background is not None else x
        explainer = shap.Explainer(clf.predict_proba, bg)
        vals = explainer(x).values[..., 1]

    vals = np.asarray(vals).reshape(-1)[: len(FEATURES)]
    raw = row_df[FEATURES].iloc[0]

    out = []
    for i, feat in enumerate(FEATURES):
        out.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "value": float(raw[feat]),
            "shap": float(vals[i]),
        })
    out.sort(key=lambda d: abs(d["shap"]), reverse=True)
    return out


def _positive_class(shap_values):
    """TreeExplainer returns different shapes across versions/models."""
    arr = np.asarray(shap_values)
    if arr.ndim == 3:            # (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
        # take positive class
        if arr.shape[0] == 2:
            return arr[1]
        return arr[..., -1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        return np.asarray(shap_values[1])
    return arr
