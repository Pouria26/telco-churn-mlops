"""
mlflow_utils.py
---------------
MLflow wrapper. Saves models via joblib + log_artifact (works with MLflow 3.14 + SQLite).
"""

import os, tempfile, mlflow, joblib
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DIR = os.path.join(PROJECT_ROOT, "mlruns")
EXPERIMENT_NAME = "Telco_Churn_Prediction"


def setup_mlflow(experiment_name: str = EXPERIMENT_NAME):
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    db_path = os.path.abspath(os.path.join(MLFLOW_DIR, "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(experiment_name)


def start_run(run_name: str):
    return mlflow.start_run(run_name=run_name)


def log_params_clean(params: dict):
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


def log_figure(figure, artifact_name: str):
    mlflow.log_figure(figure, artifact_name)
    plt.close(figure)


def log_model(model, model_type: str, artifact_name: str):
    """Save model with joblib and log as MLflow artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.joblib")
        joblib.dump(model, path)
        mlflow.log_artifacts(tmp, artifact_path=artifact_name)
    print(f"  [mlflow_utils] Model saved as artifact: {artifact_name}")


def get_active_run_id() -> str:
    return mlflow.active_run().info.run_id


def get_best_run(experiment_name: str = EXPERIMENT_NAME, metric: str = "test_f1"):
    setup_mlflow(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=[f"metrics.{metric} DESC"])
    if runs.empty:
        raise RuntimeError(f"No runs found for experiment '{experiment_name}'")
    return runs.iloc[0]
