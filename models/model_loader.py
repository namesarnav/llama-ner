import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from typing import Dict, Optional

def load_tokenizer(model_id, token):
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=token)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"

    return tokenizer


def load_model(
    model_id: str,
    num_labels: int,
    dtype: torch.dtype = torch.float16,
    device_map: str | dict | None = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    token: Optional[str] = None,
):
    
    quant_config = None
    if load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_8bit_compute_dtype=dtype,
        )

    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )   

    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        quantization_config=quant_config,
        device_map=device_map,
        token=token,
        torch_dtype=dtype,
    )

    return model
