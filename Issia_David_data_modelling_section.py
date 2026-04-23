import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# =========================================================
# 1) LOAD DATA
# =========================================================
script_dir = Path(__file__).resolve().parent

candidate_paths = [
    script_dir / "Traffic_Collisions_Open_Data_2053198073974531286.csv",
    script_dir / "Traffic_Collisions_Open_Data_2053198073974531286",
]

DATA_PATH = None
for p in candidate_paths:
    if p.exists():
        DATA_PATH = p
        break

# Fallback: use the first CSV file found in the same folder
if DATA_PATH is None:
    csv_files = list(script_dir.glob("*.csv"))
    if csv_files:
        DATA_PATH = csv_files[0]

if DATA_PATH is None:
    raise FileNotFoundError(
        f"No CSV file found in: {script_dir}\n"
        "Put the CSV in the same folder as this script."
    )

print(f"Using CSV file: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# =========================================================
# 2) BASIC CHECKS
# =========================================================
required_columns = [
    "FATALITIES",
    "OCC_DOW",
    "OCC_MONTH",
    "OCC_YEAR",
    "OCC_HOUR",
    "DIVISION",
    "HOOD_158",
    "LONG_WGS84",
    "LAT_WGS84",
    "AUTOMOBILE",
    "MOTORCYCLE",
    "PASSENGER",
    "BICYCLE",
    "PEDESTRIAN",
]

missing_required = [col for col in required_columns if col not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns in CSV: {missing_required}")

# =========================================================
# 3) TARGET CREATION
#    Fatal collision target = 1 if fatalities > 0, else 0
# =========================================================
df["FATALITIES"] = pd.to_numeric(df["FATALITIES"], errors="coerce")
df = df.dropna(subset=["FATALITIES"]).copy()
df["IS_FATAL"] = (df["FATALITIES"] > 0).astype(int)

# =========================================================
# 4) SIMPLE FEATURE ENGINEERING
# =========================================================
df["IS_WEEKEND"] = df["OCC_DOW"].isin(["Saturday", "Sunday"]).astype(int)

# =========================================================
# 5) DROP COLUMNS
#    Remove leakage, IDs, duplicates, and unnecessary columns
# =========================================================
drop_cols = [
    "IS_FATAL",
    "FATALITIES",
    "INJURY_COLLISIONS",
    "FTR_COLLISIONS",
    "PD_COLLISIONS",
    "OBJECTID",
    "EVENT_UNIQUE_ID",
    "OCC_DATE",
    "NEIGHBOURHOOD_158",
    "x",
    "y",
]

drop_cols_existing = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=drop_cols_existing)
y = df["IS_FATAL"]

# =========================================================
# 6) FEATURE GROUPS
# =========================================================
categorical_features = [
    "OCC_MONTH",
    "OCC_DOW",
    "DIVISION",
    "HOOD_158",
    "AUTOMOBILE",
    "MOTORCYCLE",
    "PASSENGER",
    "BICYCLE",
    "PEDESTRIAN",
]

numeric_features = [
    "OCC_YEAR",
    "OCC_HOUR",
    "LONG_WGS84",
    "LAT_WGS84",
    "IS_WEEKEND",
]

# Keep only columns that really exist in X
categorical_features = [col for col in categorical_features if col in X.columns]
numeric_features = [col for col in numeric_features if col in X.columns]

# =========================================================
# 7) PREPROCESSING
# =========================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# =========================================================
# 8) MODEL PIPELINE
# =========================================================
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("select", SelectPercentile(score_func=chi2, percentile=80)),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=200,
        random_state=42
    ))
])

# =========================================================
# 9) TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================================================
# 10) TRAIN MODEL
# =========================================================
model.fit(X_train, y_train)

# =========================================================
# 11) PREDICTIONS
# =========================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================================================
# 12) RESULTS
# =========================================================
print("\n================ DATA SUMMARY ================")
print("Dataset shape:", df.shape)
print("Feature matrix shape:", X.shape)
print("Target positive rate:", round(y.mean(), 6))
print("Target counts:")
print(y.value_counts())

print("\n================ CLASSIFICATION REPORT ================")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

print("================ METRICS ================")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 4))
print("PR-AUC   :", round(average_precision_score(y_test, y_prob), 4))

print("\n================ CONFUSION MATRIX ================")
print(confusion_matrix(y_test, y_pred))