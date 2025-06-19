Package for facilitating protein language model fine-tuning using huggingface models.
The package aims to offer flexibility in combining any protein language model and any task head for fine-tuning easily.
(1) The task head may perform classification or regression, and can be residue-level or protein-level.
(2) Various protein language models can be used just by passing the checkpoint (e.g. Prot-T5, Prot-BERT, ESM2, variants of ESM2, etc)
(3) Can pass processor to handle model-specific data input requirement (e.g. Prot T5 requires amino acids in sequence to be separated by a space)
