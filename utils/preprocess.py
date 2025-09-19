"""Dataset preparation utilities for token classification."""
from __future__ import annotations

from functools import partial
from typing import Dict, Iterable, List, Mapping, Tuple

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from utils.data_loader import load_conll


def build_label_mappings(labels: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create consistent label⇄id mappings using the provided label order."""

    label_list = list(labels)
    if not label_list:
        raise ValueError("Label list cannot be empty")

    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def _align_labels(
    tokenized_inputs: Mapping[str, List[List[int]]],
    batch_labels: List[List[str]],
    label2id: Mapping[str, int],
    label_all_tokens: bool,
) -> List[List[int]]:
    """Align word-level labels to tokenized inputs producing label ids."""

    aligned_labels: List[List[int]] = []

    for batch_index, word_labels in enumerate(batch_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids: List[int] = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
                continue

            current_label = word_labels[word_id]

            if word_id != previous_word_id:
                label_ids.append(label2id[current_label])
            else:
                if label_all_tokens:
                    inside_label = current_label
                    if inside_label.startswith("B-"):
                        inside_label = inside_label.replace("B-", "I-", 1)
                    label_ids.append(label2id[inside_label])
                else:
                    label_ids.append(-100)

            previous_word_id = word_id

        aligned_labels.append(label_ids)

    return aligned_labels


def tokenize_and_align(
    examples: Mapping[str, List[List[str]]],
    tokenizer: PreTrainedTokenizerBase,
    label2id: Mapping[str, int],
    *,
    max_length: int | None = None,
    label_all_tokens: bool = False,
) -> Dict[str, List[List[int]]]:
    """Tokenize raw token sequences and align labels for token classification."""

    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )

    aligned_labels = _align_labels(tokenized_inputs, examples["ner_tags"], label2id, label_all_tokens)
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def prepare_datasets(
    tokenizer: PreTrainedTokenizerBase,
    train_path: str,
    eval_path: str,
    labels: Iterable[str],
    *,
    max_length: int | None = None,
    label_all_tokens: bool = False,
) -> Tuple[Dataset, Dataset, Dict[str, int], Dict[int, str]]:
    """Load raw CoNLL data and return tokenized Hugging Face datasets."""

    train_records = load_conll(train_path)
    eval_records = load_conll(eval_path)

    label2id, id2label = build_label_mappings(labels)

    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(eval_records)

    preprocess_fn = partial(
        tokenize_and_align,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
        label_all_tokens=label_all_tokens,
    )

    tokenized_train = train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_eval = eval_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    return tokenized_train, tokenized_eval, label2id, id2label
