"""Evaluation helpers shared by both models."""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob):
    """Return accuracy, precision, recall, F1, ROC AUC and confusion-matrix counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def format_metrics(name, m):
    return (
        f"{name:<14} acc={m['accuracy']*100:6.2f}%  prec={m['precision']*100:6.2f}%  "
        f"rec={m['recall']*100:6.2f}%  f1={m['f1']*100:6.2f}%  auc={m['roc_auc']:.4f}"
    )
