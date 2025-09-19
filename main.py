from __future__ import annotations

import torch
from transformers import (
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import DEFAULT_CONFIG
from models import get_lora_model, load_model, load_tokenizer
from utils.device import describe_device, resolve_device
from utils.metrics import build_compute_metrics
from utils.preprocess import prepare_datasets


def main() -> None:
    cfg = DEFAULT_CONFIG

    device = resolve_device()
    print(f"Using device: {describe_device(device)}")

    print("Loading tokenizer ...")
    tokenizer = load_tokenizer(cfg.tokenizer_id, hf_token=cfg.hf_token)

    print("Preparing datasets ...")
    train_dataset, eval_dataset, label2id, id2label = prepare_datasets(
        tokenizer,
        str(cfg.train_file),
        str(cfg.eval_file),
        cfg.labels,
        max_length=cfg.max_length,
        label_all_tokens=cfg.label_all_tokens,
    )

    num_labels = len(label2id)

    supports_8bit = device.type == "cuda"
    dtype = torch.float16 if supports_8bit else torch.float32
    device_map = "auto" if supports_8bit else None
    bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    print("Loading base model ...")
    model = load_model(
        cfg.model_id,
        hf_token=cfg.hf_token,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        dtype=dtype,
        device_map=device_map,
        load_in_8bit=supports_8bit,
    )

    if getattr(model.config, "vocab_size", None) and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    model.config.pad_token_id = pad_token_id

    print("Applying LoRA adapters ...")
    model = get_lora_model(model)
    model.print_trainable_parameters()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if not supports_8bit:
        model.to(device)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest")

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        eval_steps=200,
        logging_steps=50,
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        fp16=supports_8bit,
        bf16=bf16_available,
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )

    compute_metrics = build_compute_metrics(id2label)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("Starting training ...")
    trainer.train()

    print("Saving model to", cfg.output_dir)
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    print("Evaluating best checkpoint ...")
    metrics = trainer.evaluate(eval_dataset)
    print(metrics)


if __name__ == "__main__":
    main()
