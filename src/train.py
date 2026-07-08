"""
train.py - Churn classification.
Runs 8 models, picks best, saves model locally + MLflow metrics/charts.
"""

import json
import os
import time
import warnings

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import data_loader, evaluate, mlflow_utils

warnings.filterwarnings("ignore")
with open("config.yaml") as f:
    config = yaml.safe_load(f)
RANDOM_SEED = config["globals"]["random_seed"]
TARGET_COL = config["globals"]["target_col"]
MODELS_DIR = "models"


def load_version_data(version):
    df = data_loader.load_version(version)
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])
    return X, y


def build_preprocessor_v2(X):
    return Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    )


def build_preprocessor_v3(X):
    passthrough = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Phone Service",
        "Contract",
        "Paperless Billing",
    ]
    passthrough = [c for c in passthrough if c in X.columns]
    remainder_cols = [c for c in X.columns if c not in passthrough]
    return ColumnTransformer(
        [
            (
                "keep",
                Pipeline([("imp", SimpleImputer(strategy="most_frequent"))]),
                passthrough,
            ),
            (
                "rest",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                remainder_cols,
            ),
        ]
    )


def find_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.15, 0.85, 0.005):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def get_all_models(spw):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=0.5, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=1500,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=1500,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=spw,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
        ),
        "CatBoost": cb.CatBoostClassifier(
            iterations=1500,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=5,
            auto_class_weights="Balanced",
            random_seed=RANDOM_SEED,
            verbose=0,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5, random_state=RANDOM_SEED
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=500, learning_rate=0.1, random_state=RANDOM_SEED
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=6,
            num_leaves=40,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.7,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
        ),
        "Voting Ensemble": VotingClassifier(
            estimators=[
                (
                    "lr",
                    LogisticRegression(
                        max_iter=2000,
                        C=0.5,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=1000,
                        max_depth=12,
                        min_samples_split=5,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
                (
                    "xgb",
                    xgb.XGBClassifier(
                        n_estimators=1000,
                        max_depth=5,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        scale_pos_weight=spw,
                        random_state=RANDOM_SEED,
                        eval_metric="logloss",
                    ),
                ),
                (
                    "lgbm",
                    lgb.LGBMClassifier(
                        n_estimators=1000,
                        learning_rate=0.03,
                        max_depth=6,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ],
            voting="soft",
            n_jobs=-1,
        ),
    }


def train_dataset_version(
    version: str, df: pd.DataFrame, target_model: str = None
) -> dict:
    X, y = load_version_data(version)
    print(f"  {version}: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")

    mlflow_utils.setup_mlflow()
    preprocessor = (
        build_preprocessor_v2(X) if version == "v2" else build_preprocessor_v3(X)
    )
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    all_models = get_all_models(spw)

    if target_model:
        clean = target_model.lower().replace(" ", "")
        matches = {k.lower().replace(" ", ""): k for k in all_models}
        if clean in matches:
            all_models = {matches[clean]: all_models[matches[clean]]}

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    results = []

    # ============ Step 1: CV on train ============
    print(f"\n{'=' * 80}")
    print(f"  Step 1: 10-Fold CV on train set - Model Selection")
    print(f"{'=' * 80}")

    for name, model in all_models.items():
        with mlflow_utils.start_run(
            run_name=f"{name.lower().replace(' ', '_')}_{version}_cv"
        ):
            t0 = time.time()
            pipe = Pipeline([("pre", preprocessor), ("clf", model)])
            cv_scores = cross_val_score(
                pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=1
            )
            cv_f1_mean = cv_scores.mean()
            cv_f1_std = cv_scores.std()
            oof_prob = cross_val_predict(
                pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
            )[:, 1]
            best_thr, best_thr_f1 = find_best_threshold(y_train, oof_prob)
            oof_pred = (oof_prob >= best_thr).astype(int)
            cv_acc = accuracy_score(y_train, oof_pred)
            elapsed = time.time() - t0
            mlflow_utils.log_params_clean(
                {
                    "dataset_version": version,
                    "model_name": name,
                    "best_threshold": best_thr,
                }
            )
            mlflow_utils.log_metrics(
                {
                    "cv_f1_mean": cv_f1_mean,
                    "cv_f1_std": cv_f1_std,
                    "cv_threshold_f1": best_thr_f1,
                    "cv_accuracy": cv_acc,
                }
            )
            print(
                f"  {name:<22} CV-F1={cv_f1_mean:.4f}(+/-{cv_f1_std:.4f}) Thr-F1={best_thr_f1:.4f} Acc={cv_acc:.4f} thr={best_thr:.2f} [{elapsed:.0f}s]"
            )
            results.append(
                {
                    "model": name,
                    "cv_f1": cv_f1_mean,
                    "thr_f1": best_thr_f1,
                    "threshold": best_thr,
                    "run_id": mlflow_utils.get_active_run_id(),
                }
            )

    # ============ Step 2: Pick best ============
    best = max(results, key=lambda r: r["thr_f1"])
    best_name, best_thr = best["model"], best["threshold"]
    print(f"\n  Best model: {best_name} (CV Thr-F1={best['thr_f1']:.4f})")

    # ============ Step 3: Fit on full train ============
    print(f"\n{'=' * 80}")
    print(f"  Step 2: Fitting {best_name} on full train set...")
    print(f"{'=' * 80}")
    final_pipe = Pipeline([("pre", preprocessor), ("clf", all_models[best_name])])
    final_pipe.fit(X_train, y_train)

    # ============ Step 4: Train results ============
    print(f"\n{'=' * 80}")
    print(f"  Step 3: Train Results - {best_name} ({version})")
    print(f"{'=' * 80}")
    train_prob = final_pipe.predict_proba(X_train)[:, 1]
    train_pred = (train_prob >= best_thr).astype(int)
    tr_cm = confusion_matrix(y_train, train_pred)
    tr_acc = accuracy_score(y_train, train_pred)
    tr_prec = precision_score(y_train, train_pred)
    tr_rec = recall_score(y_train, train_pred)
    tr_f1 = f1_score(y_train, train_pred)
    tr_auc = roc_auc_score(y_train, train_prob)
    print(f"  Accuracy:  {tr_acc:.4f}")
    print(f"  Precision: {tr_prec:.4f}")
    print(f"  Recall:    {tr_rec:.4f}")
    print(f"  F1 Score:  {tr_f1:.4f}")
    print(f"  ROC AUC:   {tr_auc:.4f}")
    print(f"  Threshold: {best_thr:.2f}")
    print(
        f"  Confusion Matrix: TN={tr_cm[0, 0]} FP={tr_cm[0, 1]} FN={tr_cm[1, 0]} TP={tr_cm[1, 1]}"
    )

    # ============ Step 5: Test results ============
    print(f"\n{'=' * 80}")
    print(f"  Step 4: Test Results - {best_name} ({version})")
    print(f"{'=' * 80}")
    test_prob = final_pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_thr).astype(int)
    cm = confusion_matrix(y_test, test_pred)
    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred)
    rec = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_prob)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")
    print(f"  Threshold: {best_thr:.2f}")
    print(
        f"  Confusion Matrix: TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}"
    )
    print(f"{'=' * 80}")

    # ============ Step 6: Save model locally + MLflow charts ============
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"churn_{version}.joblib")
    joblib.dump(final_pipe, model_path)
    print(f"\n  Model saved: {model_path}")

    info = {
        "model_name": best_name,
        "dataset_version": version,
        "best_threshold": best_thr,
        "features": list(X.columns),
        "train": {
            "accuracy": tr_acc,
            "precision": tr_prec,
            "recall": tr_rec,
            "f1": tr_f1,
            "roc_auc": tr_auc,
        },
        "test": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
        },
    }
    info_path = os.path.join(MODELS_DIR, f"churn_{version}_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Info saved: {info_path}")

    with mlflow_utils.start_run(
        run_name=f"{best_name.lower().replace(' ', '_')}_{version}_FINAL"
    ):
        mlflow_utils.log_params_clean(
            {
                "dataset_version": version,
                "model_name": best_name,
                "best_threshold": best_thr,
            }
        )
        mlflow_utils.log_metrics(
            {
                "train_accuracy": tr_acc,
                "train_f1": tr_f1,
                "test_accuracy": acc,
                "test_precision": prec,
                "test_recall": rec,
                "test_f1": f1,
                "test_roc_auc": auc,
            }
        )
        mlflow_utils.log_confusion_matrix(cm)
        mlflow_utils.log_figure(
            evaluate.plot_confusion_matrix(cm, f"{best_name} ({version}) - Test"),
            "confusion_matrix.png",
        )
        mlflow_utils.log_figure(
            evaluate.plot_roc_curve(y_test, test_prob, f"ROC ({best_name})"),
            "roc_curve.png",
        )
        mlflow_utils.log_figure(
            evaluate.plot_precision_recall_curve(
                y_test, test_prob, f"PR ({best_name})"
            ),
            "precision_recall_curve.png",
        )
        fi = evaluate.plot_feature_importance(
            final_pipe, list(X.columns), f"Features ({best_name})"
        )
        if fi:
            mlflow_utils.log_figure(fi, "feature_importance.png")
        evaluate.plot_confusion_matrix(
            cm,
            f"{best_name} ({version}) - Test",
            save_as=f"confusion_matrix_{version}.png",
        )
        final_id = mlflow_utils.get_active_run_id()

    print(f"\n  MLflow run: {final_id} — run 'mlflow ui' to see charts.\n")

    return {
        "dataset_version": version,
        "model": best_name,
        "run_id": final_id,
        "best_threshold": best_thr,
        "cv_f1": best["cv_f1"],
        "cv_thr_f1": best["thr_f1"],
        "train_accuracy": tr_acc,
        "train_f1": tr_f1,
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "test_roc_auc": auc,
        "selection_results": results,
    }
