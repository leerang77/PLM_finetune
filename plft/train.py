"""
Trainer Module
"""
from typing import Optional, Dict
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
)
# from metrics import get_compute_metrics_fn
from plft.utils.metrics import get_compute_metrics_fn
from plft.models.model import PLMTaskModel
from plft.utils.config import TaskType

class ProteinTaskTrainer:
    """
    Trainer for protein sequence tasks, handling training and evaluation
    using Hugging Face's Trainer API.
    """
    def __init__(
        self,
        model: PLMTaskModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        test_dataset: Optional[Dataset],
        tokenizer: AutoTokenizer,
        output_dir: str,
        num_train_epochs: int = 100,
        per_device_train_batch_size: int = 8,
        learning_rate: float = 1e-4,
        eval_strategy: str = "epoch",
        save_strategy: str = "epoch",
        logging_steps: int = 50,
    ):
        """
        Initializes the ProteinTaskTrainer with model, datasets, tokenizer, and training parameters.
        """
        self.model         = model
        self.train_dataset = train_dataset
        self.eval_dataset  = eval_dataset
        self.test_dataset  = test_dataset

        # pick the right collator:
        # for token-classification, pad labels to -100 so CrossEntropyLoss(ignore_index=-100) skips them
        # for everything else (seq-classification, seq-/token-regression), plain padding is sufficient
        if self.model.task_type == TaskType.TOKEN_CLASSIFICATION:
            self.data_collator = DataCollatorForTokenClassification(
                tokenizer,
                label_pad_token_id=-100,
            )
        else:
            self.data_collator = DataCollatorWithPadding(tokenizer)

        compute_metrics = get_compute_metrics_fn(self.model.task_type) # Get the appropriate metrics function based on task type

        # Initialize the Trainer
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            learning_rate=learning_rate,
            logging_strategy="steps",
            logging_steps=logging_steps,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            tokenizer=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )

    def train(self):
        """
        Train the model using the huggingface Trainer module.
        """
        return self.trainer.train()

    def evaluate(self, split: str = "validation") -> Dict[str, float]:
        """
        Evaluate the model on the specified dataset split (train, validation, or test).
        Args:
            split (str): The dataset split to evaluate on. Can be "train", "validation", or "test".
        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        if split == "train":
            ds = self.train_dataset
        elif split == "validation":
            ds = self.eval_dataset
        elif split == "test":
            if self.test_dataset is None:
                raise ValueError("No test dataset provided.")
            ds = self.test_dataset
        else:
            raise ValueError(f"Unknown split: {split}")
        return self.trainer.evaluate(eval_dataset=ds)
