from typing import Optional, Callable, List

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict

import ast

class ProteinDataModule:
    """
    Data module for protein sequence datasets, handling loading, preprocessing,
    and tokenization using Hugging Face's datasets library.
    This module supports training, validation, and optional test datasets.
    """
    def __init__(
        self,
        train_file: str,
        val_file: str,
        tokenizer: AutoTokenizer,
        preprocess_fn: Optional[Callable[[str], str]] = None,
        max_length: int = 1024,
        test_file: Optional[str] = None,
        sequence_column: str = "sequence",
        label_column: str = "label",
        optional_features: Optional[List[str]] = None,
    ):
        """
        Initializes the ProteinDataModule with training and validation files,
        a tokenizer, and optional preprocessing function and optional test file.
        Input data files should be csv with the column "sequence", and optionally "label"
        Args:
            train_file (str): Path to the training dataset file.
            val_file (str): Path to the validation dataset file.
            tokenizer (AutoTokenizer): Tokenizer for processing sequences.
            preprocess_fn (Optional[Callable[[str], str]]): Function to preprocess sequences.
            max_length (int): Maximum length for tokenized sequences.
            test_file (Optional[str]): Path to the test dataset file, if available.
            sequence_column (str): Name of the column containing sequences.
            label_column (str): Name of the column containing labels.
            optional_features (Optional[List[str]]): Additional features to include in the dataset.
        """
        self.tokenizer = tokenizer
        self.preprocess_fn = preprocess_fn
        self.max_length = max_length
        self.optional_features = optional_features if optional_features else []
        files = {"train": train_file, "validation": val_file}
        if test_file:
            files["test"] = test_file
        raw = load_dataset("csv", data_files=files)

        def preprocess(dataset):
            """
            Preprocesses the input dataset by applying the preprocessing function
            and tokenizing the sequences.
            """
            seqs = dataset[sequence_column]
            if self.preprocess_fn: # Optional preprocessing step (e.g. Add space for ProtBert)
                seqs = [self.preprocess_fn(s) for s in seqs]
                print("Applied preprocess_fn to sequences.")
                print("Sample preprocessed sequence:", seqs[0] if len(seqs) > 0 else "n/a")
            tokenized = self.tokenizer( # Tokenization
                seqs,
                truncation=True,
                max_length=self.max_length,
            )
            if label_column in dataset: # Attach labels if available
                if isinstance(dataset[label_column][0], str):
                    tokenized["label"] = [ast.literal_eval(x) for x in dataset[label_column]]
                else:
                    tokenized["label"] = dataset[label_column]
            for key in self.optional_features: # Optional keys for additional features to keep
                if key in dataset:
                    tokenized[key] = dataset[key]
            return tokenized

        self.datasets = raw.map(preprocess, batched=True)

    def get_datasets(self) -> DatasetDict[str, Dataset]:
        """
        Returns the processed datasets for training, validation, and optional test.
        Returns:
            DatasetDict[str, Dataset]: Dictionary containing the processed datasets.
        """
        return self.datasets