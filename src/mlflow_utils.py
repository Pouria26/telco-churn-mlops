"""
mlflow_utils.py
---------------
Thin wrapper around MLflow so that train.py / evaluate.py never call
mlflow.* directly. Keeping this in one place makes it trivial to change
the tracking backend later (e.g. sqlite -> a remote MLflow server) without
touching training code.
"""

import os
import mlflow
import mlflow.lightgbm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DIR = os.path.join(PROJECT_ROOT, "mlruns")
EXPERIMENT_NAME = "Telco_Churn_Prediction"


def setup_mlflow(experiment_name: str = EXPERIMENT_NAME):
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(MLFLOW_DIR)}/mlflow.db")
    mlflow.set_experiment(experiment_name)
    print(f"[mlflow_utils] tracking_uri={mlflow.get_tracking_uri()} experiment={experiment_name}")


def start_run(run_name: str):
    return mlflow.start_run(run_name=run_name)


def log_params_clean(params: dict):
    """MLflow only accepts primitive types as param values; stringify anything else."""
    clean = {}
    for k, v in params.items():
        clean[k] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
    mlflow.log_params(clean)


def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)


def log_confusion_matrix(cm):
    mlflow.log_metric("test_cm_tn", int(cm[0, 0]))
    mlflow.log_metric("test_cm_fp", int(cm[0, 1]))
    mlflow.log_metric("test_cm_fn", int(cm[1, 0]))
    mlflow.log_metric("test_cm_tp", int(cm[1, 1]))


def log_model(model, model_type: str, artifact_name: str):
    if model_type == "sklearn":
        mlflow.sklearn.log_model(sk_model=model, name=artifact_name)
    elif model_type == "xgboost":
        mlflow.xgboost.log_model(xgb_model=model, name=artifact_name)
    elif model_type == "catboost":
        mlflow.catboost.log_model(cb_model=model, name=artifact_name)
    elif model_type == "lightgbm": 
        mlflow.lightgbm.log_model(lgb_model=model, name=artifact_name)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")


def get_active_run_id() -> str:
    return mlflow.active_run().info.run_id


def get_best_run(experiment_name: str = EXPERIMENT_NAME, metric: str = "test_f1"):
    """Return the mlflow run with the highest value of `metric` in the given experiment."""
    setup_mlflow(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=[f"metrics.{metric} DESC"])
    if runs.empty:
        raise RuntimeError(f"No runs found for experiment '{experiment_name}'")
    return runs.iloc[0]
