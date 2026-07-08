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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score,
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
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    fig.tight_layout()

    if save_as:
        path = os.path.join(REPORTS_DIR, save_as)
        fig.savefig(path, dpi=150)
        print(f"[evaluate] saved confusion matrix -> {path}")
    return fig


def plot_roc_curve(y_true, y_prob, title: str = "ROC Curve", save_as: str = None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_as:
        path = os.path.join(REPORTS_DIR, save_as)
        fig.savefig(path, dpi=150)
        print(f"[evaluate] saved ROC curve -> {path}")
    return fig


def plot_precision_recall_curve(y_true, y_prob, title: str = "Precision-Recall Curve", save_as: str = None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="green", lw=2, label=f"PR curve (AP = {ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()

    if save_as:
        path = os.path.join(REPORTS_DIR, save_as)
        fig.savefig(path, dpi=150)
        print(f"[evaluate] saved PR curve -> {path}")
    return fig


def plot_feature_importance(pipeline, feature_names, title: str = "Feature Importance", save_as: str = None):
    """Plot feature importance for tree-based models. Returns None if model doesn't support it."""
    try:
        if hasattr(pipeline, "named_steps") and "classifier" in pipeline.named_steps:
            model = pipeline.named_steps["classifier"]
        else:
            model = pipeline

        if not hasattr(model, "feature_importances_"):
            return None

        importances = model.feature_importances_
        n_features = len(feature_names)
        n_importances = len(importances)

        if n_importances != n_features:
            feature_names = [f"feature_{i}" for i in range(n_importances)]

        indices = np.argsort(importances)[::-1][:20]

        os.makedirs(REPORTS_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(indices)), importances[indices][::-1], align="center")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices][::-1])
        ax.set_xlabel("Importance")
        ax.set_title(title)
        fig.tight_layout()

        if save_as:
            path = os.path.join(REPORTS_DIR, save_as)
            fig.savefig(path, dpi=150)
            print(f"[evaluate] saved feature importance -> {path}")
        return fig
    except Exception:
        return None


def build_leaderboard(results: list, sort_by: str = "cv_thr_f1") -> "pandas.DataFrame":
    import pandas as pd
    df = pd.DataFrame(results)
    if sort_by not in df.columns:
        sort_by = df.columns[-1]
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
