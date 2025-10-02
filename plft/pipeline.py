"""
PipeLine for PLFT (Protein Language Fine-Tuning).
This module orchestrates the training and evaluation of pLM fine tuning tasks.
Uses a config-driven approach to set up the model, data, and training parameters.
"""
import os
import random
import torch
from hydra import main
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoConfig
from plft.utils.config import TaskType, to_task_type
from plft.utils.lora_utils import inject_lora
from plft.models.model import PLMTaskModel
from plft.datamodule import ProteinDataModule
from plft.train import ProteinTaskTrainer
from plft.configs.registries import HEAD_REGISTRY, PREPROC_REGISTRY

@main(config_path="configs", config_name="protbert_seqcls.yaml")
def run(cfg: DictConfig):
    """
    Defines the main pipeline for PLFT.
    Loads configuration, initializes tokenizer, data module, model, and trainer.
    Then runs the training and evaluation process.
    Args:
        cfg_path (str): Path to the configuration YAML file.
    """
    # Seed
    seed = cfg.get("seed", 42)
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 1. define the data module. We need to get tokenizer and preprocess function to do that.
    # Tokenizer
    tok_cfg = cfg["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tok_cfg["name"],
        do_lower_case=tok_cfg.get("do_lower_case", False),
    )
    # Get the preprocess function using the registry
    preproc_name = cfg["data"].get("preprocess", "none")
    preprocess_fn = PREPROC_REGISTRY.get(preproc_name, lambda s: s)
    # DataModule
    data_cfg = cfg["data"]
    dm = ProteinDataModule(
        train_file=data_cfg["train_file"],
        val_file=data_cfg["val_file"],
        test_file=data_cfg.get("test_file", None),
        tokenizer=tokenizer,
        preprocess_fn=preprocess_fn,
        max_length=data_cfg["max_length"],
        label_column=data_cfg["label_column"],
        sequence_column=data_cfg["sequence_column"],
        optional_features=data_cfg.get("optional_features", []),
    )
    datasets = dm.get_datasets()

    # 2. Define the task head
    # Infer num_labels from dataset if possible
    num_labels = cfg["head"]["params"].get("output_dim")
    feat = datasets["train"].features
    if num_labels is None and "label" in feat and hasattr(feat["label"], "num_classes"):
        num_labels = feat["label"].num_classes
        cfg["head"]["params"]["output_dim"] = num_labels

    # Backbone hidden size → head.input_dim
    backbone_name = cfg["model"]["backbone_name"]
    hf_config = AutoConfig.from_pretrained(backbone_name)
    hidden_size = getattr(hf_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"Backbone {backbone_name} has no hidden_size in config.")
    cfg["head"]["params"]["input_dim"] = hidden_size

    # Define the head using the registry
    head_type = cfg["head"]["type"]
    head_model = HEAD_REGISTRY[head_type]
    head = head_model(**cfg["head"]["params"])

    # 3. Define the full model
    # Model
    model = PLMTaskModel(
        task_type=to_task_type(cfg["model"]["task_type"]),
        backbone_name=backbone_name,
        head=head,
    )
    # Optional PEFT/LoRA
    if bool(cfg.peft.enabeled):
        p = cfg["peft"]["params"]
        model = inject_lora(model, p)
    # Trainer
    tr_cfg = cfg["trainer"]
    trainer = ProteinTaskTrainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        test_dataset=datasets.get("test"),
        tokenizer=tokenizer,
        output_dir=tr_cfg["output_dir"],
        num_train_epochs=tr_cfg["epochs"],
        per_device_train_batch_size=tr_cfg["batch_size"],
        learning_rate=tr_cfg["learning_rate"],
        eval_strategy=tr_cfg["eval_strategy"],
        save_strategy=tr_cfg["save_strategy"],
        logging_steps=tr_cfg["logging_steps"],
    )

    # 4. Train + eval
    trainer.train()
    split = tr_cfg.get("eval_split", "validation")
    val_metrics = trainer.evaluate(split=split)
    print(f"Eval metrics on {split}:", val_metrics)

    if datasets.get("test") is not None:
        test_metrics = trainer.evaluate(split="test")
        print("Test:", test_metrics)

if __name__ == "__main__":
    run()
