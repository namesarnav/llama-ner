"""Runtime configuration utilities."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

_THIS_DIR = Path(__file__).parent
_CONFIG_PATH = _THIS_DIR / "config.yml"


@dataclass(frozen=True)
class AppConfig:
    model_id: str
    tokenizer_id: str
    train_file: Path
    eval_file: Path
    output_dir: Path
    labels: List[str]
    max_length: int | None
    hf_token: str | None
    label_all_tokens: bool


def _load_yaml_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def load_config() -> AppConfig:
    raw = _load_yaml_config()

    model_id = os.getenv("MODEL_ID", raw.get("model_id", "meta-llama/Llama-3.2-1B"))
    tokenizer_id = os.getenv("TOKENIZER_ID", raw.get("tokenizer_id", model_id))

    train_file = Path(os.getenv("TRAIN_FILE", raw.get("train_file", "data/train.conll")))
    eval_file = Path(os.getenv("EVAL_FILE", raw.get("eval_file", "data/test.conll")))
    output_dir = Path(os.getenv("FINAL_MODEL_DIR", raw.get("final_model_dir", "artifacts/final-model")))

    labels = raw.get("labels") or ["B-T", "I-T", "O"]
    labels_env = os.getenv("LABELS")
    if labels_env:
        labels = [label.strip() for label in labels_env.split(",") if label.strip()]

    max_length_env = os.getenv("MAX_LENGTH")
    max_length = int(max_length_env) if max_length_env else raw.get("max_length")

    hf_token = os.getenv("HF_TOKEN", raw.get("hf_token"))

    label_all_tokens_env = os.getenv("LABEL_ALL_TOKENS")
    if label_all_tokens_env is not None:
        label_all_tokens = label_all_tokens_env.lower() in {"1", "true", "yes", "y"}
    else:
        label_all_tokens = bool(raw.get("label_all_tokens", False))

    return AppConfig(
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        train_file=train_file,
        eval_file=eval_file,
        output_dir=output_dir,
        labels=labels,
        max_length=max_length,
        hf_token=hf_token,
        label_all_tokens=label_all_tokens,
    )


DEFAULT_CONFIG = load_config()
