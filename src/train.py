"""
train.py
--------
Owns everything related to *modeling*: the sklearn preprocessing
ColumnTransformer, the candidate model configurations, the
train/validation/test split, cross-validation, and the full
train -> validate -> (single) test evaluation cycle, with every run
logged to MLflow through mlflow_utils.

Design follows the professor's explicit requirement:
    - Several CV rounds on TRAIN to keep results stable.
    - VALIDATION is used to pick the best model / hyperparameters.
    - TEST is evaluated exactly ONCE, only for the final chosen model,
      and that is the number reported for this dataset version.
"""

import time
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

import xgboost as xgb
import catboost as cb
import lightgbm as lgb

from src import mlflow_utils
from src import evaluate
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

RANDOM_SEED = 42
TARGET_COL = "Churn Value"

# Columns that keep their original (already 0/1 or already-ordinal) encoding untouched;
# everything else gets median-imputed + standard-scaled.
PASSTHROUGH_FEATURES = [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Contract", "Paperless Billing",
]

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

RANDOM_SEED = config["globals"]["random_seed"]
TARGET_COL = config["globals"]["target_col"]
PASSTHROUGH_FEATURES = config["data"]["passthrough_features"]

def build_preprocessor(df: pd.DataFrame, target_col: str = "Churn Value"):
    df = df.copy()
    y = df[target_col]
    X = df.drop(columns=[target_col])

    passthrough = [c for c in PASSTHROUGH_FEATURES if c in X.columns]
    passthrough_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    remainder_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[("keep_as_is", passthrough_pipeline, passthrough)],
        remainder=remainder_pipeline,
    )
    return X, y, preprocessor


def split_train_val_test(X, y, test_size=0.15, val_size=0.15, is_regression=False):
    """
    Splits the data into Train, Validation, and Test sets.
    Disables stratification if it's a regression task (like CLTV).
    """
    stratify_target = None if is_regression else y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED, 
                                                        stratify=stratify_target)
    adjusted_val_size = val_size / (1.0 - test_size)
    stratify_target_val = None if is_regression else y_train

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  test_size=adjusted_val_size, 
                                                      random_state=RANDOM_SEED, stratify=stratify_target_val)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_model_configs(y_train) -> dict:
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    cfg_m = config["models"]
    
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=cfg_m["logistic_regression"]["max_iter"], random_state=RANDOM_SEED, class_weight=cfg_m["logistic_regression"]["class_weight"]),
            "type": "sklearn",
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=cfg_m["random_forest"]["n_estimators"], max_depth=cfg_m["random_forest"]["max_depth"],
                min_samples_split=cfg_m["random_forest"]["min_samples_split"], min_samples_leaf=cfg_m["random_forest"]["min_samples_leaf"],
                max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced_subsample",
            ),
            "type": "sklearn",
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                n_estimators=cfg_m["xgboost"]["n_estimators"], max_depth=cfg_m["xgboost"]["max_depth"], learning_rate=cfg_m["xgboost"]["learning_rate"],
                subsample=cfg_m["xgboost"]["subsample"], colsample_bytree=cfg_m["xgboost"]["colsample_bytree"], scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_SEED, eval_metric="logloss",
            ),
            "type": "xgboost",
        },
        "CatBoost": {
            "model": cb.CatBoostClassifier(
                iterations=cfg_m["catboost"]["iterations"], depth=cfg_m["catboost"]["depth"],
                learning_rate=cfg_m["catboost"]["learning_rate"],
                auto_class_weights="Balanced", random_seed=RANDOM_SEED, verbose=0,
                allow_writing_files=False
            ),
            "type": "catboost",
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                n_estimators=cfg_m["gradient_boosting"]["n_estimators"], learning_rate=cfg_m["gradient_boosting"]["learning_rate"],
                max_depth=cfg_m["gradient_boosting"]["max_depth"], random_state=RANDOM_SEED
            ),
            "type": "sklearn"
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(
                n_estimators=cfg_m["adaboost"]["n_estimators"], learning_rate=cfg_m["adaboost"]["learning_rate"], random_state=RANDOM_SEED
            ),
            "type": "sklearn"
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(
                n_estimators=cfg_m["lightgbm"]["n_estimators"], learning_rate=cfg_m["lightgbm"]["learning_rate"],
                max_depth=cfg_m["lightgbm"]["max_depth"], random_state=RANDOM_SEED, class_weight="balanced",
                n_jobs=-1, verbosity=-1
            ),
            "type": "lightgbm" 
        }
    }

def cross_validate(pipeline, X_train, y_train, n_splits: int = 5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    return scores.mean(), scores.std(), cv


def train_dataset_version(version: str, df: pd.DataFrame, target_model: str = None) -> dict:
    """
    Runs the full model-selection + final-test cycle for one dataset version.
    Returns the best run's info (model name, run_id, test metrics).
    """
    X, y, preprocessor = build_preprocessor(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

    mlflow_utils.setup_mlflow()
    all_models = get_model_configs(y_train)
    
    if target_model:
        available_models = {k.lower().replace(" ", ""): k for k in all_models.keys()}
        clean_target = target_model.lower().replace(" ", "").replace("_", "").replace("-", "")
        
        if clean_target in available_models:
            actual_key = available_models[clean_target]
            models = {actual_key: all_models[actual_key]}
            print(f"🎯 Filtering pipeline to run ONLY: {actual_key}")
        else:
            print(f"❌ Model '{target_model}' not found. Running all 7 models instead.")
            models = all_models
    else:
        models = all_models

    selection_results = []
    fitted_pipelines = {}

    print(f"\n{'='*100}\nModel selection on '{version}' (CV on train, scored on validation)\n{'='*100}")
    for model_name, cfg in models.items():
        run_name = f"{model_name.lower().replace(' ', '_')}_{version}_model_selection"
        with mlflow_utils.start_run(run_name=run_name):
            start = time.time()
            pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", cfg["model"])])

            cv_mean, cv_std, cv = cross_validate(pipeline, X_train, y_train)
            pipeline.fit(X_train, y_train)
            fitted_pipelines[model_name] = pipeline

            val_pred = pipeline.predict(X_val)
            val_prob = pipeline.predict_proba(X_val)[:, 1]
            val_metrics = evaluate.compute_metrics(y_val, val_pred, val_prob)
            elapsed = time.time() - start

            mlflow_utils.log_params_clean({
                "dataset_version": version, "model_name": model_name, "model_type": cfg["type"],
                "random_seed": RANDOM_SEED, "stage": "model_selection",
                "train_samples": len(X_train), "val_samples": len(X_val), "n_features": X_train.shape[1],
            })
            mlflow_utils.log_metrics({
                "cv_f1_mean": cv_mean, "cv_f1_std": cv_std,
                "val_accuracy": val_metrics["accuracy"], "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"], "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"], "training_time_sec": elapsed,
            })

            print(f"{version:<4} {model_name:<22} CV-F1={cv_mean:.4f} Val-F1={val_metrics['f1']:.4f}")
            selection_results.append({
                "dataset_version": version, "model": model_name, "cv_f1": cv_mean,
                "val_f1": val_metrics["f1"], "run_id": mlflow_utils.get_active_run_id(),
            })

    # Pick the best model using VALIDATION performance only (test is untouched so far)
    best = max(selection_results, key=lambda r: r["val_f1"])
    best_model_name = best["model"]
    best_pipeline = fitted_pipelines[best_model_name]
    best_cfg = models[best_model_name]

    print(f"\nBest on validation for '{version}': {best_model_name} (val_f1={best['val_f1']:.4f})")
    print("Running the SINGLE final evaluation on the untouched TEST set...")

    with mlflow_utils.start_run(run_name=f"{best_model_name.lower().replace(' ', '_')}_{version}_FINAL_TEST"):
        test_pred = best_pipeline.predict(X_test)
        test_prob = best_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate.compute_metrics(y_test, test_pred, test_prob)
        cm = evaluate.get_confusion_matrix(y_test, test_pred)

        mlflow_utils.log_params_clean({
            "dataset_version": version, "model_name": best_model_name, "model_type": best_cfg["type"],
            "random_seed": RANDOM_SEED, "stage": "final_test", "test_samples": len(X_test),
        })
        mlflow_utils.log_metrics({
            "test_accuracy": test_metrics["accuracy"], "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"], "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
        })
        mlflow_utils.log_confusion_matrix(cm)
        mlflow_utils.log_model(best_pipeline.named_steps["classifier"], best_cfg["type"],
                                artifact_name=f"{best_model_name.lower().replace(' ', '_')}_{version}")
        evaluate.plot_confusion_matrix(cm, f"{best_model_name} — {version} (test)",
                                        save_as=f"confusion_matrix_{version}.png")

        final_run_id = mlflow_utils.get_active_run_id()

    print(f"FINAL test metrics [{version}] {best_model_name}: {test_metrics}")

    return {
        "dataset_version": version,
        "model": best_model_name,
        "run_id": final_run_id,
        "val_f1": best["val_f1"],
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "selection_results": selection_results,
    }
