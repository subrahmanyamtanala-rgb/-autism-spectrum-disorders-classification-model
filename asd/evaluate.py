"""Metrics and plots shared by all models."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)


def metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall (sensitivity)": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan"),
    }


def plot_confusion_matrices(results: dict, y_true, path: str) -> None:
    n = len(results)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    for ax, (name, proba) in zip(np.ravel(axes), results.items()):
        cm = confusion_matrix(y_true, (proba >= 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["No ASD", "ASD"])
        ax.set_yticks([0, 1], ["No ASD", "ASD"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=10)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(results: dict, y_true, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, proba in results.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - test set")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: dict, path: str) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(history["loss"], label="train")
    a1.plot(history["val_loss"], label="validation")
    a1.set_title("CNN loss")
    a1.set_xlabel("Epoch")
    a1.legend()
    a2.plot(history["accuracy"], label="train")
    a2.plot(history["val_accuracy"], label="validation")
    a2.set_title("CNN accuracy")
    a2.set_xlabel("Epoch")
    a2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
