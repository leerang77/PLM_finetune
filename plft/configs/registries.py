"""
This module defines registries for model heads and preprocessing functions.
Update as new heads or preprocessing functions are added to the PLFT framework.
"""
from typing import Dict, Callable
from plft.models.taskhead import MLPHead, MPNNClassifierHead
from plft.data.preprocess_functions import ProtBert_preprocess

HEAD_REGISTRY: Dict[str, Callable] = {
    "mlp": MLPHead,
    "mpnn": MPNNClassifierHead,
}

PREPROC_REGISTRY: Dict[str, Callable[[str], str]] = {
    "protbert": ProtBert_preprocess,
    "none": lambda s: s,
}
