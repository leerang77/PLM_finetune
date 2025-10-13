"""
Main model that combines a pretrained language model backbone with a task-specific head.
"""
from typing import Optional, Any
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    SequenceClassifierOutput,
)
from plft.utils.config import TaskType

class PLMTaskModel(PreTrainedModel):
    """General model for sequence/token classification and regression."""
    def __init__(
        self,
        task_type: TaskType,
        backbone_name: str,
        head: nn.Module,
    ):
        """
        Initializes the PLMTaskModel with a backbone model, and head for task-specific processing
        
        Args:
            task_type (TaskType): Type of the task (e.g., SEQ_CLASSIFICATION, TOKEN_CLASSIFICATION).
            backbone_name (str): Name of the pretrained model backbone.
            head (nn.Module): Task-specific head to be used on top of the backbone.
        """
        # Load the config and backbone
        config = AutoConfig.from_pretrained(backbone_name)
        backbone = AutoModel.from_pretrained(backbone_name, config=config)
        
        # Call the PretrainedModel constructor
        super().__init__(config)
        
        # Attach the modules
        self.backbone = backbone # Assigns the pretrained weights
        self.head = head # Assigns the prediction head
        self.task_type = task_type
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **head_args: Any,
    ) -> SequenceClassifierOutput:
        """
        Forward pass for the PLMTaskModel.
        
        Args:
            input_ids (torch.LongTensor): Input token IDs.
            attention_mask (Optional[torch.FloatTensor]): Attention mask.
            labels (Optional[torch.LongTensor]): Labels for the data.
            **head_args (Any): Additional arguments for the head.
        Returns:
            SequenceClassifierOutput: Output of the model including logits and loss if labels are provided.
        """
        # Compute embedding
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.head(hidden_states, attention_mask=attention_mask, **head_args)
        
        # Compute loss
        loss = None
        if labels is not None:
            if self.task_type is TaskType.TOKEN_REGRESSION:
                # logits: (batch, seq_len, 1) → squeeze
                preds = logits.squeeze(-1)                    # (batch, seq_len)
                # build a mask of the real (non-pad) tokens
                mask  = attention_mask.to(preds.dtype)        # 1.0 for real tokens, 0.0 for pads
                # some residues may be real but is missing a label;
                # the labels will be padded but they will be included in the attention.
                # Thus, additionally ignore pads in loss computation:
                mask = mask*(labels != -100).to(mask.dtype)  # combine with label mask if any
                # compute squared error only on real tokens
                se    = (preds - labels.float()) ** 2         # (batch, seq_len)
                loss  = (se * mask).sum() / mask.sum()        # mean over real positions
    
            elif self.task_type is TaskType.TOKEN_CLASSIFICATION:
                    # logits: (batch, seq_len, num_labels)
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
    
            elif self.task_type is TaskType.SEQ_REGRESSION:
                    # logits: (batch, 1)
                loss = nn.MSELoss()(logits.squeeze(-1), labels.float())
    
            else:  # SEQ_CLASSIFICATION
                    # logits: (batch, num_labels)
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
     
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states   if output_hidden_states else None,
            attentions=outputs.attentions         if output_attentions    else None,
        )