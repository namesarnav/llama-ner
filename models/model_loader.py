"""Model and tokenizer loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_tokenizer(model_id: str, *, hf_token: Optional[str] = None) -> PreTrainedTokenizerBase:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    model_id: str,
    *,
    hf_token: Optional[str],
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    dtype: torch.dtype = torch.float16,
    device_map: str | dict | None = "auto",
    load_in_8bit: bool = True,
) -> PreTrainedModel:
    quant_config = None
    if load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_8bit_compute_dtype=dtype,
        )

    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        token=hf_token,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        quantization_config=quant_config,
        device_map=device_map,
    )

    if quant_config is None and dtype:
        model.to(dtype=dtype)

    model.config.use_cache = False
    return model
