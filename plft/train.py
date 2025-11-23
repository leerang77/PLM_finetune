"""
Trainer Module
"""
from typing import Optional, Dict
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
# from metrics import get_compute_metrics_fn
from plft.utils.metrics import get_compute_metrics_fn
from plft.utils.config import choose_data_collator
from plft.models.model import PLMTaskModel

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
        max_steps: int = -1,
        per_device_train_batch_size: int = 8,
        learning_rate: float = 1e-4,
        eval_strategy: str = "epoch",
        save_strategy: str = "epoch",
        save_total_limit: int=1,
        load_best_model_at_end: bool=True,
        metric_for_best_model: str="eval_loss",
        greater_is_better: bool=False,
        logging_steps: int = 50,
        early_stopping: bool = False,
        early_stopping_patience: int = 3,
    ):
        """
        Initializes the ProteinTaskTrainer with model, datasets, tokenizer, and training parameters.
        """
        self.model         = model
        self.train_dataset = train_dataset
        self.eval_dataset  = eval_dataset
        self.test_dataset  = test_dataset

        # pick the right collator:
        # for token classification or regression, pad labels to -100 to ignore in loss computation
        # for seq classification and regression, plain padding is sufficient
        self.data_collator = choose_data_collator(self.model.task_type, tokenizer)

        compute_metrics = get_compute_metrics_fn(self.model.task_type) # Get the appropriate metrics function based on task type
        
        # Early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=0.0,
        ) if early_stopping else None

        # Initialize the Trainer
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            learning_rate=learning_rate,
            logging_strategy="steps",
            logging_steps=logging_steps,
            logging_dir=f"{output_dir}/runs",     # where tensorboard logs go
            report_to=["tensorboard"],            # enable tensorboard backend
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            tokenizer=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping_callback] if early_stopping else None,
        )

    def train(self):
        """
        Train the model using the huggingface Trainer module.
        """
        return self.trainer.train()
    
    def predict(self, dataset: Dataset) -> Dict[str, any]:
        """
        Make predictions on the given dataset.
        Args:
            dataset (Dataset): The dataset to make predictions on.
        Returns:
            Dict[str, any]: A dictionary containing predictions and related information.
        """
        return self.trainer.predict(test_dataset=dataset)

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
    
    def save_metrics(self, split: str, metrics: Dict[str, float]):
        """
        Save evaluation metrics to a file.
        Args:
            split (str): The dataset split the metrics correspond to.
            metrics (Dict[str, float]): The evaluation metrics to save.
        """
        self.trainer.save_metrics(split, metrics)
