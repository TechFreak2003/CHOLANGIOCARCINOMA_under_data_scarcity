"""
preprocessing.py — reusable preprocessing pipeline.

Numeric features: median imputation + standardisation.
Binary features:  most-frequent imputation (0/1 kept as-is otherwise).

Returned as a single sklearn ColumnTransformer so the exact same transform
that was fit on training data is serialised with the model and reapplied at
inference. A model separated from its scaler is silently wrong.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from schema import NUMERIC_FEATURES, BINARY_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("bin", binary, BINARY_FEATURES),
    ])
