"""
models.py — the five candidate classifiers from Table 4.1.

Each is returned wrapped in a Pipeline(preprocessor -> classifier) so that
preprocessing travels with the model. get_models() returns an ordered dict
name -> unfitted Pipeline.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocessing import make_preprocessor

SEED = 42

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False


def _wrap(clf):
    return Pipeline([("pre", make_preprocessor()), ("clf", clf)])


def get_models(scale_pos_weight: float = 1.0) -> dict:
    """Return name -> unfitted Pipeline for every available candidate.

    scale_pos_weight is passed to the boosters to handle class imbalance;
    it is computed from the training split by the caller (train.py).
    """
    models = {
        "Logistic Regression": _wrap(LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=SEED)),
        "Decision Tree": _wrap(DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=SEED)),
        "Random Forest": _wrap(RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=SEED)),
    }
    if HAVE_XGB:
        models["XGBoost"] = _wrap(XGBClassifier(
            tree_method="hist", n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=SEED))
    if HAVE_LGB:
        models["LightGBM"] = _wrap(LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, is_unbalance=True,
            random_state=SEED, verbose=-1))
    return models
