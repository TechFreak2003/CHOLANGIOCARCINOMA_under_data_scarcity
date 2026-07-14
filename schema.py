"""
schema.py — the single source of truth for the feature set.

Every module (data generation, preprocessing, training, inference) imports
FEATURES from here, so the column order can never drift between training and
prediction. TARGET is the binary label column.
"""

# Numeric continuous features
NUMERIC_FEATURES = [
    "age", "bmi", "alcohol_units",
    "alp", "ggt", "alt", "ast", "bilirubin", "ca19_9", "albumin",
]

# Binary / categorical (0/1) features
BINARY_FEATURES = [
    "sex", "se_asian", "smoker",
    "psc", "liver_fluke", "hbv", "hcv",
    "cirrhosis", "gallstones", "t2dm", "fam_bileduct",
]

FEATURES = NUMERIC_FEATURES + BINARY_FEATURES
TARGET = "label"

# Human-readable labels for SHAP output / UI
FEATURE_LABELS = {
    "age": "Age",
    "bmi": "Body-mass index",
    "alcohol_units": "Alcohol (units/week)",
    "alp": "Alkaline phosphatase (ALP)",
    "ggt": "Gamma-glutamyl transferase (GGT)",
    "alt": "Alanine aminotransferase (ALT)",
    "ast": "Aspartate aminotransferase (AST)",
    "bilirubin": "Total bilirubin",
    "ca19_9": "CA 19-9",
    "albumin": "Albumin",
    "sex": "Male sex",
    "se_asian": "Southeast Asian ethnicity",
    "smoker": "Smoker",
    "psc": "Primary sclerosing cholangitis",
    "liver_fluke": "Liver fluke exposure",
    "hbv": "Hepatitis B",
    "hcv": "Hepatitis C",
    "cirrhosis": "Chronic liver disease / cirrhosis",
    "gallstones": "Gallstones",
    "t2dm": "Type 2 diabetes",
    "fam_bileduct": "Family history of bile duct cancer",
}

RISK_BANDS = [
    (0.00, 0.15, "Low"),
    (0.15, 0.40, "Moderate"),
    (0.40, 0.70, "High"),
    (0.70, 1.01, "Very High"),
]


def risk_band(p: float) -> str:
    """Map a probability to a categorical risk band."""
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "Very High"
