"""Inference helpers for the fine-tuned token classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from peft import AutoPeftModelForTokenClassification
from transformers import AutoTokenizer, BitsAndBytesConfig

from config import DEFAULT_CONFIG
from utils.device import describe_device, resolve_device

TokenPrediction = Tuple[str, str]


class TokenClassifier:
    """Lightweight wrapper around the fine-tuned PEFT model for inference."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        hf_token: str | None = None,
        device: torch.device | None = None,
        load_in_8bit: bool = True,
    ) -> None:
        cfg = DEFAULT_CONFIG
        model_path = Path(model_path or cfg.output_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model directory not found: {model_path}")

        self.device = device or resolve_device()
        supports_8bit = load_in_8bit and self.device.type == "cuda"

        quantization_config = None
        device_map = None
        if supports_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_8bit_compute_dtype=torch.float16,
            )
            device_map = "auto"

        print(f"Loading inference model on {describe_device(self.device)} ...")

        self.model = AutoPeftModelForTokenClassification.from_pretrained(
            model_path,
            token=hf_token or cfg.hf_token,
            quantization_config=quantization_config,
            device_map=device_map,
        )

        self.model.eval()
        if not supports_8bit:
            self.model.to(self.device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token=hf_token or cfg.hf_token,
                use_fast=True,
            )
        except ValueError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token=hf_token or cfg.hf_token,
                use_fast=False,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    @property
    def id2label(self) -> dict[int, str]:
        return self.model.config.id2label

    def predict_sentence(self, tokens: Sequence[str]) -> List[TokenPrediction]:
        if not tokens:
            return []

        encoding = self.tokenizer(
            list(tokens),
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
        try:
            word_ids = encoding.word_ids(batch_index=0)
        except AttributeError:
            word_ids = None
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        with torch.no_grad():
            output = self.model(**encoding)

        predictions = output.logits.argmax(dim=-1).cpu().tolist()[0]

        if word_ids is None:
            word_ids = list(range(len(predictions)))

        results: List[TokenPrediction] = []
        used_word_ids: set[int] = set()

        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id in used_word_ids:
                continue
            if not (0 <= word_id < len(tokens)):
                continue
            token = tokens[word_id]
            pred_id = predictions[position]
            results.append((token, self.id2label[int(pred_id)]))
            used_word_ids.add(word_id)

        return results

    def predict(self, sentences: Iterable[Sequence[str]]) -> List[List[TokenPrediction]]:
        return [self.predict_sentence(sentence) for sentence in sentences]


def run_demo():
    classifier = TokenClassifier()
    sample_sentences = [
        ["New", "York", "City"],
        ["Meta", "released", "Llama", "3", "models"],
    ]
    predictions = classifier.predict(sample_sentences)
    for sentence, preds in zip(sample_sentences, predictions):
        print("Sentence:", " ".join(sentence))
        print("Predictions:", preds)


if __name__ == "__main__":
    run_demo()
