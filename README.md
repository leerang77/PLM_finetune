# Package for protein language model fine-tuning using huggingface models.
The package aims to offer flexibility in combining various protein language model and task head for fine-tuning easily.

1. The task head may perform classification or regression, and can be residue-level or protein-level.
2. For protein-level prediction, both pooled representation and representation from [CLS] token are supported for BERT-like models.
3. Various protein language models can be used just by passing the checkpoint (e.g. Prot-T5, Prot-BERT, ESM2, variants of ESM2, etc)
4. Can pass preprocessor to handle model-specific data input requirement (e.g. Prot T5 requires amino acids in sequence to be separated by a space)
5. Uses hydra for configuration management, making it easy to run experiments with different settings.

For more details, refer to my blog post series on finetuning protein language models:
- [Part 1: Motivation for fine-tuning and intro to Huggingface](https://leerang77.github.io/blog/2025/foundation-model/)
- [Part 2: Fine-tuning protein language models with Huggingface](https://leerang77.github.io/blog/2025/foundation-model2/)
- [Part 3: Parameter-efficient fine-tuning of protein language models](https://leerang77.github.io/blog/2025/foundation-model3/)

# Getting Started
### Installation
Clone the repository and install the package using pip:
```bash
pip install -e .
```
Make sure to also install the required dependencies listed in `requirements.txt` using your package manager of choice. Also make sure that a compatible CUDA version is installed if you plan to use GPU acceleration.

### Data Preparation
The jupyter notebook `data/prepare_data.ipynb` downloaded and preprocesses the datasets for each of the four residue/protein + classiffication/regression task combinations.

### Config files
Next, define the config file for the experiment. Example configs are available in the `plft/configs/` folder as .yaml files. You can modify them as needed. Some of the key fields are:
* `model.train_file`, `model.val_file`, `model.test_file`: paths to the training, validation, and test dataset files.
* `model.backbone_name`: checkpoint name of the protein language model from Huggingface.
* `model.preprocess`: optional preprocessing function to handle model-specific quirks for input sequence requirement. When specified, it should be defined in `plft/configs/registries.py`.
* `model.task_type`: Type of task head to use. Should be one of the `TaskType` enum defined in `plft/enums.py`.
* `model.freeze_backbone`: whether to freeze the backbone pLM during training.
* `peft.enabled`: whether to use parameter-efficient fine-tuning (PEFT) or full model fine-tuning.

### Running the training
Finally, run the training using the defined config file:
```bash
python -m plft.pipeline.py --config-name protbert_chezod_token_regression.yaml
```
Hydra allows you to easily override any config field from the command line as well. For example, to change the learning rate and batch size, you can run:
```bash
python -m plft.pipeline.py --config-name protbert_chezod_token_regression.yaml trainer.learning_rate=1e-5 trainer.batch_size=16
```

For running the training on Google Colab, a sample notebook is provided that demonstrates how to set up and run the training in Google Colab: `test_notebooks/protbert_scl_seq_cls_frozen_backbone.ipynb`. This notebook uses ProtBERT as the backbone pLM, with a classification head for protein-level classification task of subcellular location, and freezes the backbone during training.
