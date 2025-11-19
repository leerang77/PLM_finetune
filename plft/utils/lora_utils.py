# lora_utils.py
import re
from typing import Iterable, List, Union
import torch.nn as nn
from peft import LoraConfig, get_peft_model

def _linear_module_names(model) -> Iterable[str]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name

def auto_qkv_target_modules(model) -> List[str]:
    """
    Infer the correct Q/K/V linear layer names for LoRA across common architectures.
    Priority order:
      1) *proj family:  q_proj/k_proj/v_proj  (LLaMA, ESM2 HF, GPT-NeoX, etc.)
      2) BERT family:   query/key/value       (BERT, RoBERTa, ProtBert)
      3) T5 family:     q/k/v                 (T5-style SelfAttention wrappers)
      4) Fairseq-style: W_q/W_k/W_v
    Returns a list suitable for PEFT's `target_modules` (substring match).
    """
    names = list(_linear_module_names(model))

    if any(".q_proj" in n for n in names):
        return ["q_proj", "k_proj", "v_proj"]
    if any(n.endswith(".query") or ".attention.self.query" in n for n in names):
        return ["query", "key", "value"]
    if any(re.search(r"\.(q|k|v)$", n) for n in names):
        return ["q", "k", "v"]
    if any("W_q" in n for n in names):
        return ["W_q", "W_k", "W_v"]

    # Conservative fallback: works for most HF transformer families.
    return ["q_proj", "k_proj", "v_proj"]

def resolve_lora_targets(model, spec: Union[str, List[str]]) -> List[str]:
    """
    - spec == "auto_qkv" → infer via model structure
    - spec == list[str]  → return as-is
    """
    if isinstance(spec, str):
        if spec.lower() == "auto_qkv":
            return auto_qkv_target_modules(model)
        raise ValueError(f"Unknown target_modules spec: {spec}")
    return list(spec)

def inject_lora(base_model, lora_cfg_input):
    """
    Injects LoRA layers into the model based on the provided configuration.
    Args:
        base_model (nn.Module): The base model to inject LoRA into.
        lora_cfg_input (Config): Configuration object containing LoRA parameters.
    Returns:
        nn.Module: The model with LoRA layers injected.
    """
    targets = resolve_lora_targets(base_model, lora_cfg_input.target_modules)  # "auto_qkv" or list[str]
    print("Injecting LoRA into target modules:", targets)
    lora_cfg = LoraConfig(
        r=lora_cfg_input.r,
        lora_alpha=lora_cfg_input.lora_alpha,
        target_modules=targets,
        lora_dropout=lora_cfg_input.dropout,
        bias=lora_cfg_input.bias,
        task_type=lora_cfg_input.task_type,      # e.g., "SEQ_CLS", "TOKEN_CLS", etc.
    )
    model = get_peft_model(base_model, lora_cfg)
    return model