# ============================================================
# KSI_Deliverable3_ModelBuilding.py
# COMP 247 – Deliverable 3: Predictive Model Building
# Section 402  |  Group 5
#
# Builds directly on Deliverable 2 by importing ksi_shared.py.
# Same preprocessing pipeline, same features, same 80/20
# stratified split — results are directly comparable to the
# Logistic Regression baseline from Deliverable 2.
#
# Models: Logistic Regression, Decision Tree, SVM (LinearSVC),
#         Random Forest, Neural Network (MLPClassifier)
# Tuning: GridSearchCV (LR, DT, NN) + RandomizedSearchCV (SVM, RF)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# ── Import shared preprocessing from Deliverable 2 ──────────
# ksi_shared.py must be in the same folder as this script.
# It reproduces the exact pipeline from Issia_David_data_modelling_section.py
from ksi_shared import (
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    build_pipeline,
    prepare_all,
    get_tune_sample,
)

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. LOAD & PREPROCESS DATA  (via ksi_shared — same as D2)
# ============================================================
print("=" * 60)
print("  DELIVERABLE 3 — PREDICTIVE MODEL BUILDING")
print("  Extending Deliverable 2 (Issia_David_data_modelling_section.py)")
print("=" * 60)

X_train, X_test, y_train, y_test, X, y, df = prepare_all(Path(__file__))

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Sample used for all GridSearch / RandomizedSearch tuning.
# Best params are found on this sample, then refit on full X_train.
# This keeps tuning fast while still evaluating final models on full data.
X_tune, y_tune = get_tune_sample(X_train, y_train)

# ============================================================
# 2. HELPER — evaluate any trained pipeline
# ============================================================
def evaluate(name, pipeline, X_tr, X_te, y_tr, y_te):
    """Fit pipeline, print all metrics, return results dict."""
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")

    t0 = time.time()
    pipeline.fit(X_tr, y_tr)
    elapsed = round(time.time() - t0, 2)

    y_pred = pipeline.predict(X_te)
    y_prob = pipeline.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    roc  = roc_auc_score(y_te, y_prob)
    pr   = average_precision_score(y_te, y_prob)

    print(classification_report(y_te, y_pred, digits=4, zero_division=0))
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc:.4f}")
    print(f"PR-AUC    : {pr:.4f}")
    print(f"Train time: {elapsed}s")

    return dict(name=name, pipeline=pipeline,
                y_pred=y_pred, y_prob=y_prob,
                accuracy=acc, precision=prec,
                recall=rec, f1=f1, roc_auc=roc, pr_auc=pr)


# ============================================================
# 3. PHASE A — BASELINE MODELS
#    Logistic Regression here is the SAME model as Deliverable 2.
#    The other four are new for Deliverable 3.
# ============================================================
print("\n" + "#" * 60)
print("  PHASE A: BASELINE MODELS  (before tuning)")
print("#" * 60)

# -- Logistic Regression (Deliverable 2 baseline, reproduced here) --
lr_baseline = build_pipeline(
    LogisticRegression(class_weight="balanced", solver="liblinear",
                       max_iter=200, random_state=RANDOM_STATE),
    X,
)

# -- Decision Tree --
dt_baseline = build_pipeline(
    DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    X,
)

# -- SVM (LinearSVC) --
# Note: kernel SVM is impractical on ~780K rows.
# LinearSVC is equivalent to SVM with a linear kernel and scales to large data.
# CalibratedClassifierCV wraps it so predict_proba is available for ROC curves.
svm_baseline = build_pipeline(
    CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", max_iter=2000,
                  random_state=RANDOM_STATE)
    ),
    X,
)

# -- Random Forest --
rf_baseline = build_pipeline(
    RandomForestClassifier(n_estimators=100, class_weight="balanced",
                           n_jobs=-1, random_state=RANDOM_STATE),
    X,
)

# -- Neural Network --
nn_baseline = build_pipeline(
    MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100,
                  early_stopping=True, random_state=RANDOM_STATE),
    X,
)

baseline_configs = [
    ("Logistic Regression (Baseline - D2)", lr_baseline),
    ("Decision Tree (Baseline)",            dt_baseline),
    ("SVM / LinearSVC (Baseline)",          svm_baseline),
    ("Random Forest (Baseline)",            rf_baseline),
    ("Neural Network (Baseline)",           nn_baseline),
]

baseline_results = []
for name, pipe in baseline_configs:
    baseline_results.append(
        evaluate(name, pipe, X_train, X_test, y_train, y_test)
    )

# ============================================================
# 4. PHASE B — HYPERPARAMETER TUNING
# ============================================================
print("\n" + "#" * 60)
print("  PHASE B: HYPERPARAMETER TUNING")
print("#" * 60)

# ── 1. Logistic Regression — GridSearchCV ───────────────────
print("\n[1/5] Logistic Regression — GridSearchCV")
lr_grid = GridSearchCV(
    build_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=300,
                           random_state=RANDOM_STATE),
        X,
    ),
    param_grid={
        "clf__C":       [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l1", "l2"],
        "clf__solver":  ["liblinear"],
    },
    cv=cv, scoring="f1", n_jobs=-1, verbose=1,
)
lr_grid.fit(X_tune, y_tune)
print(f"  Best params : {lr_grid.best_params_}")
print(f"  Best CV F1  : {lr_grid.best_score_:.4f}")
# Refit best estimator on FULL training data before evaluation
lr_grid.best_estimator_.fit(X_train, y_train)
tuned_lr = evaluate("Logistic Regression (Tuned)",
                    lr_grid.best_estimator_, X_train, X_test, y_train, y_test)

# ── 2. Decision Tree — GridSearchCV ─────────────────────────
print("\n[2/5] Decision Tree — GridSearchCV")
dt_grid = GridSearchCV(
    build_pipeline(
        DecisionTreeClassifier(class_weight="balanced",
                               random_state=RANDOM_STATE),
        X,
    ),
    param_grid={
        "clf__max_depth":         [5, 10, 20, None],
        "clf__min_samples_split": [2, 10, 50],
        "clf__min_samples_leaf":  [1, 5, 20],
        "clf__criterion":         ["gini", "entropy"],
    },
    cv=cv, scoring="f1", n_jobs=-1, verbose=1,
)
dt_grid.fit(X_tune, y_tune)
print(f"  Best params : {dt_grid.best_params_}")
print(f"  Best CV F1  : {dt_grid.best_score_:.4f}")
dt_grid.best_estimator_.fit(X_train, y_train)
tuned_dt = evaluate("Decision Tree (Tuned)",
                    dt_grid.best_estimator_, X_train, X_test, y_train, y_test)

# ── 3. SVM — RandomizedSearchCV ─────────────────────────────
print("\n[3/5] SVM / LinearSVC — RandomizedSearchCV")
svm_rand = RandomizedSearchCV(
    build_pipeline(
        CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
        ),
        X,
    ),
    param_distributions={
        "clf__estimator__C":        [0.001, 0.01, 0.1, 1.0, 10.0],
        "clf__estimator__max_iter": [1000, 2000, 3000],
    },
    n_iter=10, cv=cv, scoring="f1",
    n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
)
svm_rand.fit(X_tune, y_tune)
print(f"  Best params : {svm_rand.best_params_}")
print(f"  Best CV F1  : {svm_rand.best_score_:.4f}")
svm_rand.best_estimator_.fit(X_train, y_train)
tuned_svm = evaluate("SVM (Tuned)",
                     svm_rand.best_estimator_, X_train, X_test, y_train, y_test)

# ── 4. Random Forest — RandomizedSearchCV ───────────────────
print("\n[4/5] Random Forest — RandomizedSearchCV")
rf_rand = RandomizedSearchCV(
    build_pipeline(
        RandomForestClassifier(class_weight="balanced",
                               n_jobs=-1, random_state=RANDOM_STATE),
        X,
    ),
    param_distributions={
        "clf__n_estimators":      [100, 200, 300],
        "clf__max_depth":         [10, 20, 30, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 5],
        "clf__max_features":      ["sqrt", "log2"],
    },
    n_iter=15, cv=cv, scoring="f1",
    n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
)
rf_rand.fit(X_tune, y_tune)
print(f"  Best params : {rf_rand.best_params_}")
print(f"  Best CV F1  : {rf_rand.best_score_:.4f}")
rf_rand.best_estimator_.fit(X_train, y_train)
tuned_rf = evaluate("Random Forest (Tuned)",
                    rf_rand.best_estimator_, X_train, X_test, y_train, y_test)

# ── 5. Neural Network — GridSearchCV ────────────────────────
print("\n[5/5] Neural Network — GridSearchCV")
nn_grid = GridSearchCV(
    build_pipeline(
        MLPClassifier(max_iter=150, early_stopping=True,
                      random_state=RANDOM_STATE),
        X,
    ),
    param_grid={
        "clf__hidden_layer_sizes": [(64, 32), (128, 64), (64, 64, 32)],
        "clf__activation":         ["relu", "tanh"],
        "clf__alpha":              [0.0001, 0.001, 0.01],
        "clf__learning_rate":      ["constant", "adaptive"],
    },
    cv=cv, scoring="f1", n_jobs=-1, verbose=1,
)
nn_grid.fit(X_tune, y_tune)
print(f"  Best params : {nn_grid.best_params_}")
print(f"  Best CV F1  : {nn_grid.best_score_:.4f}")
nn_grid.best_estimator_.fit(X_train, y_train)
tuned_nn = evaluate("Neural Network (Tuned)",
                    nn_grid.best_estimator_, X_train, X_test, y_train, y_test)

# ============================================================
# 5. PHASE C — COMPARISON TABLE
# ============================================================
tuned_results = [tuned_lr, tuned_dt, tuned_svm, tuned_rf, tuned_nn]

print("\n" + "#" * 60)
print("  PHASE C: TUNED MODEL COMPARISON")
print("#" * 60)

comparison_df = pd.DataFrame([{
    "Model":     r["name"],
    "Accuracy":  round(r["accuracy"],  4),
    "Precision": round(r["precision"], 4),
    "Recall":    round(r["recall"],    4),
    "F1-Score":  round(r["f1"],        4),
    "ROC-AUC":   round(r["roc_auc"],   4),
    "PR-AUC":    round(r["pr_auc"],    4),
} for r in tuned_results])

comparison_df.sort_values("F1-Score", ascending=False, inplace=True)
comparison_df.reset_index(drop=True, inplace=True)
print("\n" + comparison_df.to_string(index=False))

# ============================================================
# 6. CONFUSION MATRICES
# ============================================================
print("\nGenerating confusion matrices...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, r in enumerate(tuned_results):
    cm = confusion_matrix(y_test, r["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["Non-Fatal", "Fatal"]).plot(
        ax=axes[i], colorbar=False
    )
    axes[i].set_title(r["name"], fontsize=9)

axes[-1].axis("off")
plt.suptitle("Confusion Matrices — Tuned Models", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "D3_confusion_matrices.png", dpi=150)
plt.show()
print(f"  Saved -> {PLOTS_DIR}/D3_confusion_matrices.png")

# ============================================================
# 7. ROC CURVES
# ============================================================
print("Generating ROC curves...")
plt.figure(figsize=(10, 7))
colors = ["steelblue", "darkorange", "green", "red", "purple"]

for r, color in zip(tuned_results, colors):
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{r['name']}  (AUC = {r['roc_auc']:.4f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
plt.xlim([0, 1]); plt.ylim([0, 1.02])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves — All Tuned Models", fontsize=13, fontweight="bold")
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "D3_roc_curves.png", dpi=150)
plt.show()
print(f"  Saved -> {PLOTS_DIR}/D3_roc_curves.png")

# ============================================================
# 8. SAVE BEST MODEL  (for Deliverable 5 — Flask API)
# ============================================================
best_name     = comparison_df.iloc[0]["Model"]
best_result   = next(r for r in tuned_results if r["name"] == best_name)
best_pipeline = best_result["pipeline"]

pickle_path = Path("best_model.pkl")
with open(pickle_path, "wb") as f:
    pickle.dump(best_pipeline, f)

# Verify pickle round-trip
with open(pickle_path, "rb") as f:
    loaded = pickle.load(f)
sample = loaded.predict(X_test.head(3))

print(f"\n{'=' * 60}")
print(f"  BEST MODEL   : {best_name}")
print(f"  F1-Score     : {best_result['f1']:.4f}")
print(f"  ROC-AUC      : {best_result['roc_auc']:.4f}")
print(f"  Pickle path  : {pickle_path}")
print(f"  Pickle check : predictions on 3 samples -> {sample}")
print(f"{'=' * 60}")
print("\n Deliverable 3 Complete!")
print(f"   Plots  -> {PLOTS_DIR}/")
print(f"   Model  -> {pickle_path}  (used by Deliverable 5 Flask API)")