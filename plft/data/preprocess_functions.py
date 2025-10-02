"""
Module for preprocessing functions for protein sequences.
"""

def ProtBert_preprocess(seq: str) -> str:
    """
    Turn a contiguous amino-acid string into uppercase
    letters separated by spaces.
    E.g. "mkta" → "M K T A"
    """
    seq = seq.strip().upper()
    return " ".join(list(seq))