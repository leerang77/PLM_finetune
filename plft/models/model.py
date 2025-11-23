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
)
from transformers.modeling_outputs import SequenceClassifierOutput
from plft.utils.config import TaskType
from plft.utils.loss import get_loss_for_task

class PLMTaskModel(PreTrainedModel):
    """General model for sequence/token classification and regression."""
    def __init__(
        self,
        task_type: TaskType,
        backbone_name: str,
        head: nn.Module,
        freeze_backbone: bool = False,
    ):
        """
        Initializes the PLMTaskModel with a backbone model, and head for task-specific processing
        
        Args:
            task_type (TaskType): Type of the task (e.g., SEQ_CLASSIFICATION, TOKEN_CLASSIFICATION).
            backbone_name (str): Name of the pretrained model backbone.
            head (nn.Module): Task-specific head to be used on top of the backbone.
            freeze_backbone (bool): Whether to freeze the backbone during training.
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
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **head_args: Any,
    ) -> SequenceClassifierOutput: # For convenience we duck type this to use for all tasks for now, 
                                   # though it can be a bit confusing
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
        backbone_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
            "output_hidden_states": output_hidden_states,
            "output_attentions": output_attentions,
        }
        # Compute embedding
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**backbone_kwargs)
        else:
            outputs = self.backbone(**backbone_kwargs)

        hidden_states = outputs.last_hidden_state
        
        # The Hugging Face Trainer may inject `num_items_in_batch` (and other
        # internal kwargs) into the model call. Many head implementations
        # don't accept these extra kwargs, which causes a TypeError when
        # forwarded directly. Remove known Trainer-only keys before calling
        # the head.
        for arg in ["num_items_in_batch", "return_dict", "inputs_embeds", "label_mask"]:
            head_args.pop(arg, None)

        logits = self.head(hidden_states, attention_mask=attention_mask, **head_args)
        # Compute loss
        loss = None
        if labels is not None:
            loss = get_loss_for_task(self.task_type, logits, labels, attention_mask)
     
        return SequenceClassifierOutput(
            loss=loss, # (1,)
            logits=logits, # Can be different shapes depending on task
            hidden_states=outputs.hidden_states   if output_hidden_states else None,
            attentions=outputs.attentions         if output_attentions    else None,
        )