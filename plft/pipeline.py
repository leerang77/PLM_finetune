"""
PipeLine for PLFT (Protein Language Fine-Tuning).
This module orchestrates the training and evaluation of pLM fine tuning tasks.
Uses a config-driven approach to set up the model, data, and training parameters.
"""
import os
import random
import json
from pathlib import Path
import torch
from hydra import main
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoConfig
from transformers.trainer_utils import get_last_checkpoint
from plft.utils.config import to_task_type
from plft.utils.lora_utils import inject_lora
from plft.models.model import PLMTaskModel
from plft.datamodule import ProteinDataModule
from plft.train import ProteinTaskTrainer
from plft.configs.registries import HEAD_REGISTRY, PREPROC_REGISTRY
from plft.analysis.analyze_run import analyze_run

def seed_all(seed):
    """
    Seed all random number generators for reproducibility.
    """
    # Seed
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_datamodule(cfg: DictConfig) -> tuple[ProteinDataModule, AutoTokenizer]:
    """
    Initializes the ProteinDataModule based on the provided configuration.
    Args:
        cfg (DictConfig): Configuration dictionary containing data module parameters.
    Returns:
        ProteinDataModule: Initialized data module.
        AutoTokenizer: Initialized tokenizer.
    """
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
    return dm, tokenizer

def get_task_head(cfg: DictConfig, datasets: dict) -> torch.nn.Module:
    """
    Initializes the task-specific head based on the provided configuration.
    Args:
        cfg (DictConfig): Configuration dictionary containing head parameters.
        datasets (dict): Dictionary of datasets to infer properties like num_labels.
    Returns:
        torch.nn.Module: Initialized task head.
    """
    # Infer num_labels from dataset if possible
    num_labels = cfg["head"]["params"].get("output_dim")
    if num_labels is None:
        if cfg["model"]["task_type"] in ["SEQ_REGRESSION", "TOKEN_REGRESSION"]:
            num_labels = 1
        elif cfg["model"]["task_type"] in ["SEQ_CLASSIFICATION"]:
            num_labels = len(set([x['label'] for x in datasets["train"] if x != -100]))
        elif cfg["model"]["task_type"] in ["TOKEN_CLASSIFICATION"]:
            all_labels = []
            for x in datasets["train"]:
                all_labels.extend([lbl for lbl in x['label'] if lbl != -100])
            num_labels = len(set(all_labels))
            
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
    return head

def get_full_model(head: torch.nn.Module, cfg: DictConfig) -> PLMTaskModel:
    """
    Initializes the full PLMTaskModel with backbone and head.
    Args:
        head (torch.nn.Module): Task-specific head.
        cfg (DictConfig): Configuration dictionary containing model parameters.
    Returns:
        PLMTaskModel: Initialized full model.
    """
    model = PLMTaskModel(
        task_type=to_task_type(cfg["model"]["task_type"]),
        backbone_name=cfg["model"]["backbone_name"],
        head=head,
        freeze_backbone=cfg["model"].get("freeze_backbone", False)
    )

    # Optional PEFT/LoRA
    if bool(cfg.peft.enabled):
        p = cfg["peft"]["params"]
        if p.get("task_type") is None:
            # Auto infer PEFT task type
            if model.task_type in [to_task_type("SEQ_CLASSIFICATION"), to_task_type("SEQ_REGRESSION")]:
                p["task_type"] = "SEQ_CLS"
            elif model.task_type in [to_task_type("TOKEN_CLASSIFICATION"), to_task_type("TOKEN_REGRESSION")]:
                p["task_type"] = "TOKEN_CLS"
        model = inject_lora(model, p)
    return model

def get_trainer(
    model: PLMTaskModel,
    tokenizer: AutoTokenizer,
    datasets: dict,
    cfg: DictConfig,
) -> ProteinTaskTrainer:
    """
    Initializes the ProteinTaskTrainer based on the provided configuration.
    Args:
        model (PLMTaskModel): The model to be trained.
        tokenizer (AutoTokenizer): The tokenizer used for data processing.
        datasets (dict): Dictionary of datasets for training, evaluation, and testing.
        cfg (DictConfig): Configuration dictionary containing trainer parameters.
    Returns:
        ProteinTaskTrainer: Initialized trainer.
    """
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
        early_stopping=tr_cfg["early_stopping"],
        early_stopping_patience=tr_cfg["early_stopping_patience"]
    )
    return trainer

@main(config_path="configs", config_name="protbert_seqcls.yaml")
def run(cfg: DictConfig):
    """
    Defines the main pipeline for PLFT.
    Loads configuration, initializes tokenizer, data module, model, and trainer.
    Then runs the training and evaluation process.
    Args:
        cfg_path (str): Path to the configuration YAML file.
    """
    # Seed everything
    seed_all(cfg.get("seed", 42))

    # 1. define the data module. We need to get tokenizer and preprocess function to do that.
    dm, tokenizer = get_datamodule(cfg)
    datasets = dm.get_datasets()

    # 2. Define the task head
    head = get_task_head(cfg, datasets)

    # 3. Define the full model
    model = get_full_model(head, cfg)

    # 4. Define the trainer
    trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        cfg=cfg,
    )

    # 5. Train
    output_dir = Path(cfg["trainer"]["output_dir"])
    resume_from_checkpoint = cfg["trainer"]["resume_from_checkpoint"]
    last_checkpoint = None
    if resume_from_checkpoint == "auto":
        if os.path.isdir(output_dir):
            last_checkpoint = get_last_checkpoint(output_dir)
            print(f"Resuming from last checkpoint: {last_checkpoint}")
    elif isinstance(resume_from_checkpoint, str):
        last_checkpoint = resume_from_checkpoint
        if not os.path.exists(last_checkpoint):
            raise ValueError(f"Checkpoint path {last_checkpoint} does not exist.")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # 6. Evaluate
    split = cfg["trainer"].get("eval_split", "validation")
    val_metrics = trainer.evaluate(split=split)
    trainer.save_metrics(split, val_metrics)
    print(f"Eval metrics on {split}:", val_metrics)

    if datasets.get("test") is not None:
        test_metrics = trainer.evaluate(split="test")
        trainer.save_metrics("test", test_metrics)
        print("Test:", test_metrics)

    # 7. Save trainer state and analyze run
    trainer.trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
    result = analyze_run(
        run_dir=output_dir,
    )

if __name__ == "__main__":
    run()
