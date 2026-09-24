"""Loading, synthesis and preprocessing of the Q-CHAT-10 toddler screening data.

The expected input is the public "Autism screening data for toddlers" dataset
(Thabtah, 2018; 1054 records), usually distributed as
``Toddler Autism dataset July 2018.csv``. Its columns are:

    Case_No, A1..A10, Age_Mons, Qchat-10-Score, Sex, Ethnicity, Jaundice,
    Family_mem_with_ASD, Who completed the test, Class/ASD Traits

A1..A10 are the Q-CHAT-10 items already coded so that 1 means the answer
indicates an ASD trait. A toddler is labelled "Yes" when the item sum
(Qchat-10-Score) is greater than 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ITEM_COLS = [f"A{i}" for i in range(1, 11)]
NUMERIC_COLS = ["Age_Mons"]
BINARY_COLS = ["Sex", "Jaundice", "Family_mem_with_ASD"]
CATEGORICAL_COLS = ["Ethnicity", "Who completed the test"]
TARGET_COL = "Class/ASD Traits"

# Column aliases seen in different copies of the dataset.
_ALIASES = {
    "class/asd traits ": TARGET_COL,
    "class/asd traits": TARGET_COL,
    "class": TARGET_COL,
    "asd_traits": TARGET_COL,
    "who_completed_the_test": "Who completed the test",
    "who completed the test": "Who completed the test",
    "family_mem_with_asd": "Family_mem_with_ASD",
    "jaundice": "Jaundice",
    "sex": "Sex",
    "ethnicity": "Ethnicity",
    "age_mons": "Age_Mons",
    "qchat-10-score": "Qchat-10-Score",
    "case_no": "Case_No",
}

ETHNICITIES = [
    "middle eastern", "White European", "Hispanic", "black", "asian",
    "south asian", "Native Indian", "Others", "Latino", "mixed", "Pacifica",
]
RESPONDENTS = ["family member", "Health Care Professional", "Health care professional", "Self", "Others"]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in _ALIASES:
            rename[col] = _ALIASES[key]
        elif key.upper() in ITEM_COLS:
            rename[col] = key.upper()
        else:
            rename[col] = col.strip()
    return df.rename(columns=rename)


def load_csv(path: str) -> pd.DataFrame:
    """Read the toddler dataset and normalise its column names."""
    df = _normalise_columns(pd.read_csv(path))
    missing = [c for c in ITEM_COLS + NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS + [TARGET_COL]
               if c not in df.columns]
    if missing:
        raise ValueError(f"CSV at {path} is missing columns: {missing}")
    return df


def make_synthetic(n: int = 1054, seed: int = 42) -> pd.DataFrame:
    """Generate a dataset with the same schema and labelling rule as the real one.

    Each toddler gets a latent ASD propensity; item answers, family history and
    jaundice are drawn conditional on it, and the label follows the published
    rule (Q-CHAT-10 score > 3). This lets the whole pipeline run end to end
    without the real CSV. Results on it are NOT clinical evidence.
    """
    rng = np.random.default_rng(seed)
    latent = rng.random(n) < 0.69  # ~69% "Yes" in the real data
    item_p = np.where(latent[:, None], rng.uniform(0.45, 0.85, (n, 10)), rng.uniform(0.03, 0.30, (n, 10)))
    items = (rng.random((n, 10)) < item_p).astype(int)
    score = items.sum(axis=1)
    df = pd.DataFrame(items, columns=ITEM_COLS)
    df.insert(0, "Case_No", np.arange(1, n + 1))
    df["Age_Mons"] = rng.integers(12, 37, n)
    df["Qchat-10-Score"] = score
    df["Sex"] = np.where(rng.random(n) < np.where(latent, 0.74, 0.60), "m", "f")
    df["Ethnicity"] = rng.choice(ETHNICITIES, n)
    df["Jaundice"] = np.where(rng.random(n) < np.where(latent, 0.30, 0.22), "yes", "no")
    df["Family_mem_with_ASD"] = np.where(rng.random(n) < np.where(latent, 0.18, 0.12), "yes", "no")
    df["Who completed the test"] = rng.choice(RESPONDENTS, n, p=[0.93, 0.03, 0.01, 0.02, 0.01])
    df[TARGET_COL] = np.where(score > 3, "Yes", "No")
    return df


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer
    input_columns: list[str] = field(default_factory=list)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Map raw values to model-ready columns and return (features, labels).

    Case_No is an identifier and Qchat-10-Score is the sum of A1..A10 (it
    defines the label directly), so both are dropped to avoid leakage.
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["Sex"] = df["Sex"].astype(str).str.strip().str.lower().map({"m": 1, "male": 1, "f": 0, "female": 0})
    for col in ("Jaundice", "Family_mem_with_ASD"):
        df[col] = df[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    y = df[TARGET_COL].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any() or df[BINARY_COLS].isna().any().any():
        raise ValueError("Unexpected values in binary or target columns")
    X = df[ITEM_COLS + NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS]
    return X, y.to_numpy(dtype=int)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("passthrough", "passthrough", ITEM_COLS + BINARY_COLS),
            ("scale", MinMaxScaler(), NUMERIC_COLS),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ]
    )


def prepare(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Dataset:
    """Clean, split (stratified) and encode; the encoder is fit on train only."""
    X, y = clean(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_tr).astype("float32")
    Xte = pre.transform(X_te).astype("float32")
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    return Dataset(Xtr, Xte, y_tr, y_te, names, pre, list(X.columns))
