from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

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
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=_TARGET_MODULES,
        bias="none",
        task_type=TaskType.TOKEN_CLS,
    )

    return get_peft_model(model, peft_config)

