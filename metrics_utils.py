"""Shared metric computation used by both the training script and the app."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Return the six assignment metrics for a binary classifier.

    y_prob is the predicted probability of the positive class (1).
    """
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:  # only one class present in y_true
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
