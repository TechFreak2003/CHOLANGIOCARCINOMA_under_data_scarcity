import numpy as np
import pandas as pd

from schema import FEATURES, TARGET


def make_synthetic(n: int = 900, prevalence: float = 0.18, seed: int = 42) -> pd.DataFrame:
    """Synthetic CCA cohort whose risk structure follows CCA epidemiology.

    NOT real patient data. Risk weights follow the literature ordering:
    PSC and liver fluke carry the largest weights; biomarkers are driven by
    the same latent risk so the cholestatic pattern (high ALP/GGT, mild
    transaminase rise) appears in high-risk records.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(58, 12, n).clip(20, 90)
    sex = rng.integers(0, 2, n)
    bmi = rng.normal(26, 4.5, n).clip(15, 45)
    se_asian = rng.binomial(1, 0.30, n)
    alcohol_units = rng.gamma(2.0, 3.0, n).clip(0, 60)
    smoker = rng.binomial(1, 0.35, n)

    psc = rng.binomial(1, 0.06, n)
    liver_fluke = rng.binomial(1, 0.05 + 0.10 * se_asian, n)
    hbv = rng.binomial(1, 0.08, n)
    hcv = rng.binomial(1, 0.07, n)
    cirrhosis = rng.binomial(1, 0.09, n)
    gallstones = rng.binomial(1, 0.14, n)
    t2dm = rng.binomial(1, 0.16, n)
    fam_bileduct = rng.binomial(1, 0.04, n)

    logit = (-2.35
             + 0.030 * (age - 55)
             + 0.10 * (bmi - 25) / 5
             + 0.25 * se_asian
             + 0.020 * alcohol_units
             + 0.20 * smoker
             + 2.10 * psc
             + 2.00 * liver_fluke
             + 1.05 * hbv
             + 1.00 * hcv
             + 1.15 * cirrhosis
             + 0.55 * gallstones
             + 0.50 * t2dm
             + 1.10 * fam_bileduct)

    risk_p = 1 / (1 + np.exp(-logit))
    alp = rng.normal(90 + 140 * risk_p, 35, n).clip(30, 700)
    ggt = rng.normal(35 + 180 * risk_p, 30, n).clip(5, 900)
    alt = rng.normal(30 + 20 * risk_p, 12, n).clip(5, 250)
    ast = rng.normal(25 + 22 * risk_p, 12, n).clip(5, 250)
    bilirubin = rng.normal(0.7 + 3.0 * risk_p, 0.6, n).clip(0.1, 20)
    ca19_9 = rng.normal(20 + 260 * risk_p, 60, n).clip(0, 1500)
    albumin = rng.normal(4.3 - 0.9 * risk_p, 0.4, n).clip(2, 5.5)

    # bisection on an intercept shift to hit target prevalence
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() > prevalence:
            hi = mid
        else:
            lo = mid
    shift = (lo + hi) / 2

    noisy_logit = logit + shift + rng.normal(0, 0.45, n)
    label = rng.binomial(1, 1 / (1 + np.exp(-noisy_logit)))

    df = pd.DataFrame(dict(
        age=age, bmi=bmi, alcohol_units=alcohol_units,
        alp=alp, ggt=ggt, alt=alt, ast=ast, bilirubin=bilirubin,
        ca19_9=ca19_9, albumin=albumin,
        sex=sex, se_asian=se_asian, smoker=smoker,
        psc=psc, liver_fluke=liver_fluke, hbv=hbv, hcv=hcv,
        cirrhosis=cirrhosis, gallstones=gallstones, t2dm=t2dm,
        fam_bileduct=fam_bileduct,
        label=label.astype(int),
    ))
    return df[FEATURES + [TARGET]]


def load_real(path: str) -> pd.DataFrame:
    """Load a real dataset (CSV or TSV) with the schema columns + label.

    Delimiter is auto-detected. Raises a clear error listing any missing
    columns so schema mismatches fail loudly, not silently.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset at {path} is missing required columns: {sorted(missing)}. "
            f"Expected schema: {FEATURES + [TARGET]}"
        )
    return df[FEATURES + [TARGET]]
