"""
Configuration module for PLFT (Pretrained Language Fine-Tuning) project.
"""

from enum import Enum
from dataclasses import dataclass
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification
import torch

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
    return task_type

@dataclass
class DataCollatorForTokenRegression:
    """
    Data collator that will dynamically pad the inputs for token-level regression tasks.
    Args:
        tokenizer: The tokenizer used for encoding the data.
        pad_value (float): The value to use for padding labels (default: -100).
    """
    tokenizer: any
    pad_value: float = -100
    def __call__(self, features):
        # pull labels off first so tokenizer.pad() doesn't see them
        labels = [torch.tensor(f.pop("label"), dtype=torch.float32) for f in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")

        max_len = batch["input_ids"].size(1)
        padded = [
            torch.nn.functional.pad(y, (0, max_len - y.numel()), value=self.pad_value)
            for y in labels
        ]
        batch["labels"] = torch.stack(padded)                       # [B, L] float32
        batch["label_mask"] = batch["labels"] != self.pad_value         # [B, L] bool
        return batch

def choose_data_collator(task_type: TaskType, tokenizer: any):
    """
    Choose the appropriate data collator based on the task type.
    Args:
        task_type (TaskType): The type of task (e.g., SEQ_CLASSIFICATION, TOKEN_REGRESSION).
        tokenizer: The tokenizer used for encoding the data.
    Returns:
        A data collator instance suitable for the specified task type.
    """
    if task_type == TaskType.TOKEN_CLASSIFICATION:
        return DataCollatorForTokenClassification(
            tokenizer,
            label_pad_token_id=-100,
        )
    elif task_type == TaskType.TOKEN_REGRESSION:
        return DataCollatorForTokenRegression(tokenizer, pad_value=-100)
    else:
        return DataCollatorWithPadding(tokenizer)