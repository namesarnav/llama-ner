"""LoRA configuration helpers."""
from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model

_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_lora_model(
    model,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
):
    """Attach LoRA adapters to the supplied base model."""

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=_TARGET_MODULES,
        bias="none",
        task_type=TaskType.TOKEN_CLS,
    )

    return get_peft_model(model, peft_config)
