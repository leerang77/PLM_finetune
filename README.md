# Package for protein language model fine-tuning using huggingface models.
The package aims to offer flexibility in combining various protein language model and task head for fine-tuning easily.
1. The task head may perform classification or regression, and can be residue-level or protein-level.
2. For protein-level prediction, both pooled representation and representation from [CLS] token are supported for BERT-like models.
3. Various protein language models can be used just by passing the checkpoint (e.g. Prot-T5, Prot-BERT, ESM2, variants of ESM2, etc)
4. Can pass preprocessor to handle model-specific data input requirement (e.g. Prot T5 requires amino acids in sequence to be separated by a space)
5. Uses hydra for configuration management, making it easy to run experiments with different settings.

## Installation

```bash
pip install plft
```
## Usage


