# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:20:26 2026

@author: edwin
"""

# ============================================================
# ksi_shared.py
# COMP 247 – KSI Group Project  |  Section 402  |  Group 5
#
# Shared module imported by BOTH deliverables:
#   • Issia_David_data_modelling_section.py  (Deliverable 2)
#   • KSI_Deliverable3_ModelBuilding.py      (Deliverable 3)
#
# Contains: data loading, feature engineering, preprocessing
# pipeline, and train/test split — defined ONCE so both
# deliverables use the exact same setup.
# ============================================================

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ============================================================
# CONSTANTS  (change here → changes everywhere)
# ============================================================
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 3

REQUIRED_COLUMNS = [
    "FATALITIES", "OCC_DOW", "OCC_MONTH", "OCC_YEAR", "OCC_HOUR",
    "DIVISION", "HOOD_158", "LONG_WGS84", "LAT_WGS84",
    "AUTOMOBILE", "MOTORCYCLE", "PASSENGER", "BICYCLE", "PEDESTRIAN",
]

DROP_COLS = [
    "IS_FATAL", "FATALITIES",
    "INJURY_COLLISIONS", "FTR_COLLISIONS", "PD_COLLISIONS",
    "OBJECTID", "EVENT_UNIQUE_ID", "OCC_DATE",
    "NEIGHBOURHOOD_158", "x", "y",
]

# These must match exactly what Deliverable 2 defined
CATEGORICAL_FEATURES = [
    "OCC_MONTH", "OCC_DOW", "DIVISION", "HOOD_158",
    "AUTOMOBILE", "MOTORCYCLE", "PASSENGER", "BICYCLE", "PEDESTRIAN",
]

NUMERICAL_FEATURES = [
    "OCC_YEAR", "OCC_HOUR", "LONG_WGS84", "LAT_WGS84", "IS_WEEKEND",
]


# ============================================================
# DATA LOADING
# ============================================================
def find_csv(start_dir: Path) -> Path:
    """Locate the KSI CSV in start_dir, trying known filenames first."""
    candidates = [
        start_dir / "Traffic_Collisions_Open_Data_2053198073974531286.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: first CSV found in folder
    csvs = list(start_dir.glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(
        f"No CSV found in {start_dir}. "
        "Place the KSI dataset CSV in the same folder as the scripts."
    )


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Load raw CSV and validate required columns exist."""
    print(f"[ksi_shared] Loading: {csv_path.name}")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"[ksi_shared] Raw shape: {df.shape}")
    return df


# ============================================================
# FEATURE ENGINEERING  (Deliverable 2 — section 4)
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps defined in Deliverable 2:
      - Convert FATALITIES to numeric, drop NaN rows
      - Create IS_FATAL binary target (1 = fatal, 0 = non-fatal)
      - Create IS_WEEKEND engineered feature (1 = Sat/Sun)
    """
    df = df.copy()
    df["FATALITIES"] = pd.to_numeric(df["FATALITIES"], errors="coerce")
    df = df.dropna(subset=["FATALITIES"])
    df["IS_FATAL"]   = (df["FATALITIES"] > 0).astype(int)
    df["IS_WEEKEND"] = df["OCC_DOW"].isin(["Saturday", "Sunday"]).astype(int)
    return df


# ============================================================
# X / y SPLIT  (Deliverable 2 — section 5 & 6)
# ============================================================
def get_X_y(df: pd.DataFrame):
    """
    Apply column drops from Deliverable 2 and return feature
    matrix X and binary target y.
    Only keeps columns listed in CATEGORICAL_FEATURES and
    NUMERICAL_FEATURES to guarantee consistency across scripts.
    """
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X_all = df.drop(columns=drop_existing)
    y     = df["IS_FATAL"]

    # Keep only the features defined in Deliverable 2
    keep_cols = [c for c in CATEGORICAL_FEATURES + NUMERICAL_FEATURES
                 if c in X_all.columns]
    X = X_all[keep_cols]
    return X, y


# ============================================================
# PREPROCESSING PIPELINE  (Deliverable 2 — section 7)
# ============================================================
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Reproduce the exact ColumnTransformer from Deliverable 2:
      Numerical  → median imputer → MinMaxScaler
      Categorical → most-frequent imputer → OneHotEncoder
    """
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    num_cols = [c for c in NUMERICAL_FEATURES   if c in X.columns]

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  MinMaxScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ])


# ============================================================
# FULL PIPELINE BUILDER  (Deliverable 2 — section 8)
# ============================================================
def build_pipeline(classifier, X: pd.DataFrame) -> Pipeline:
    """
    Wrap preprocessor + SelectPercentile(chi2, 80%) + classifier
    in a single Pipeline — same structure used in Deliverable 2.
    """
    return Pipeline([
        ("preprocess", build_preprocessor(X)),
        ("select",     SelectPercentile(score_func=chi2, percentile=80)),
        ("clf",        classifier),
    ])


# ============================================================
# TRAIN / TEST SPLIT  (Deliverable 2 — section 9)
# ============================================================
def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Stratified 80/20 split with random_state=42 — identical to
    what was used in Deliverable 2 so results are comparable.
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ============================================================
# ONE-CALL CONVENIENCE FUNCTION
# ============================================================
def prepare_all(script_path: Path):
    """
    Full pipeline from raw file → ready-to-train data.
    Returns: X_train, X_test, y_train, y_test, X, y, df
    """
    csv_path = find_csv(script_path.resolve().parent)
    df       = load_raw(csv_path)
    df       = engineer_features(df)
    X, y     = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    cat_used = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    num_used = [c for c in NUMERICAL_FEATURES   if c in X.columns]
    print(f"[ksi_shared] Features  : {len(cat_used)} categorical, "
          f"{len(num_used)} numerical")
    print(f"[ksi_shared] Train size: {X_train.shape[0]} | "
          f"Test size: {X_test.shape[0]}")
    print(f"[ksi_shared] Fatal rate: {y.mean():.4%}")

    return X_train, X_test, y_train, y_test, X, y, df