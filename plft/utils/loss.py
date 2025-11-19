from plft.utils.config import TaskType
from typing import Optional
import torch
import torch.nn as nn

def get_loss_for_task(
    task_type: TaskType,
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mak: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the appropriate loss for the given task type.
    Args:
        task_type (TaskType): The type of task.
        logits (torch.Tensor): The model outputs.
        labels (torch.Tensor): The true labels.
        attention_mask (Optional[torch.Tensor]): Attention mask for token-level tasks.
    Returns:
        torch.Tensor: The computed loss.
    """
    if task_type is TaskType.TOKEN_REGRESSION:
        # logits: (batch, seq_len, 1) → squeeze
        preds = logits.squeeze(-1)                    # (batch, seq_len)
        # some residues may be real but is missing a label;
        # the labels will be padded but they will be included in the attention.
        # Thus, additionally ignore pads in loss computation:
        # labels: (batch, seq_len)
        mask = (labels != -100).to(int)  # combine with label mask if any

        # compute squared error only on real tokens
        se    = (preds - labels.float()) ** 2         # (batch, seq_len)
        loss  = (se * mask).sum() / mask.sum()        # mean over real positions


    elif task_type is TaskType.TOKEN_CLASSIFICATION:
            # logits: (batch, seq_len, num_labels)
        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

    elif task_type is TaskType.SEQ_REGRESSION:
            # logits: (batch, 1)
        loss = nn.MSELoss()(logits.squeeze(-1), labels.float())

    else:  # SEQ_CLASSIFICATION
            # logits: (batch, num_labels)
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
    return loss