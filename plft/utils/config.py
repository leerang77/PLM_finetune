"""
Configuration module for PLFT (Pretrained Language Fine-Tuning) project.
"""

from enum import Enum

class TaskType(Enum):
    """
    Enum representing different task types for the prediction head.
    """
    SEQ_CLASSIFICATION   = "seq_classification"
    SEQ_REGRESSION       = "seq_regression"
    TOKEN_CLASSIFICATION = "token_classification"
    TOKEN_REGRESSION     = "token_regression"

def to_task_type(s: str) -> TaskType:
    """Converts a string to a TaskType enum.
    """
    # accept either enum-name or enum-value
    task_type = getattr(TaskType, s, None)
    if task_type is None:
        valid = [k for k in vars(TaskType).keys() if k.isupper()]
        raise ValueError(f"Unknown TaskType '{s}'. Valid: {valid}")