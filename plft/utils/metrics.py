# metrics.py

from typing import Dict, Callable
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score
from plft.utils.config import TaskType

def seq_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute sequence‐level classification metrics.

    Args:
        logits (np.ndarray): Model raw outputs, shape (batch_size, num_labels).
        labels (np.ndarray): True labels, shape (batch_size,).

    Returns:
        Dict[str, float]:
            - accuracy: fraction of correct predictions
            - precision: weighted precision across classes
            - recall: weighted recall across classes
            - f1: weighted F1 score across classes
    """
    preds = np.argmax(logits, axis=-1)
    accuracy  = float((preds == labels).mean())
    precision = float(precision_score(labels, preds, average="weighted", zero_division=0))
    recall    = float(recall_score(labels, preds, average="weighted", zero_division=0))
    f1        = float(f1_score(labels, preds, average="weighted", zero_division=0))

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


def token_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute token‐level classification metrics, ignoring padding tokens (-100).

    Args:
        logits (np.ndarray): Raw outputs, shape (batch_size, seq_len, num_labels).
        labels (np.ndarray): True labels with padding mask -100, shape (batch_size, seq_len).

    Returns:
        Dict[str, float]:
            - accuracy: fraction of correct non‐padded tokens
            - precision: weighted precision on non‐padded tokens
            - recall: weighted recall on non‐padded tokens
            - f1: weighted F1 score on non‐padded tokens
    """
    preds = np.argmax(logits, axis=-1)
    mask  = labels != -100

    valid_preds  = preds[mask]
    valid_labels = labels[mask]

    if valid_labels.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    accuracy  = float((valid_preds == valid_labels).mean())
    precision = float(precision_score(valid_labels, valid_preds, average="weighted", zero_division=0))
    recall    = float(recall_score(valid_labels, valid_preds, average="weighted", zero_division=0))
    f1        = float(f1_score(valid_labels, valid_preds, average="weighted", zero_division=0))

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }

def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return R² or NaN if not computable (e.g., <2 valid samples)."""
    if y_true.size < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))

def seq_regression_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Sequence-level regression metrics.
    logits: (batch,) or (batch,1)
    labels: (batch,)
    """
    preds  = np.asarray(logits).squeeze(-1)
    labels = np.asarray(labels).squeeze()

    mse  = float(np.mean((preds - labels) ** 2))
    mae  = float(np.mean(np.abs(preds - labels)))
    rmse = float(np.sqrt(mse))
    r2   = _safe_r2(labels, preds)
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def token_regression_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Token-level regression metrics.
    logits: (batch, seq_len) or (batch, seq_len, 1)
    labels: (batch, seq_len) with pads either as -100 (int) or NaN (float).
    """
    preds  = np.asarray(logits).squeeze(-1)   # (B, L)
    labels = np.asarray(labels)

    # Build mask of valid tokens:
    if np.issubdtype(labels.dtype, np.integer):
        mask = labels != -100
        labels = labels.astype(np.float32)
    else:
        # floating labels: treat NaN as padding
        mask = ~np.isnan(labels)

    vpreds  = preds[mask]
    vlabels = labels[mask]

    if vlabels.size == 0:
        return {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0}

    mse  = float(np.mean((vpreds - vlabels) ** 2))
    mae  = float(np.mean(np.abs(vpreds - vlabels)))
    rmse = float(np.sqrt(mse))
    r2   = _safe_r2(vlabels, vpreds)
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def get_compute_metrics_fn(task_type: TaskType) -> Callable[[EvalPrediction], Dict[str, float]]:
    """
    Return the appropriate metrics function for the given task type.
    """
    if task_type is TaskType.SEQ_CLASSIFICATION:
        return lambda pred: seq_classification_metrics(pred.predictions, pred.label_ids)
    elif task_type is TaskType.TOKEN_CLASSIFICATION:
        return lambda pred: token_classification_metrics(pred.predictions, pred.label_ids)
    elif task_type is TaskType.SEQ_REGRESSION:
        return lambda pred: seq_regression_metrics(pred.predictions, pred.label_ids)
    elif task_type is TaskType.TOKEN_REGRESSION:
        return lambda pred: token_regression_metrics(pred.predictions, pred.label_ids)
    else:
        # fallback: no metrics
        return lambda pred: {}