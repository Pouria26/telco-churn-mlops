"""
evaluate.py
-----------
Pure evaluation logic: turning predictions into metrics, confusion
matrices, and a leaderboard across MLflow runs. Nothing here trains
a model or touches MLflow's tracking API directly (that's train.py /
mlflow_utils.py) -- this module only *computes* and *visualizes*.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(cm: np.ndarray, title: str, save_as: str = None):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    if save_as:
        path = os.path.join(REPORTS_DIR, save_as)
        plt.savefig(path, dpi=150)
        print(f"[evaluate] saved confusion matrix -> {path}")
    plt.close()


def build_leaderboard(results: list, sort_by: str = "test_f1") -> "pandas.DataFrame":
    import pandas as pd
    df = pd.DataFrame(results)
    return df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)


def plot_leaderboard(leaderboard, metric: str = "test_f1", save_as: str = "leaderboard.png"):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=leaderboard, x="model", y=metric, hue="dataset_version", palette="muted")
    plt.title(f"Model comparison — {metric}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, save_as)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[evaluate] saved leaderboard chart -> {path}")
