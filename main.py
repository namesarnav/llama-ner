from config import (
    MODEL_ID,
    HF_TOKEN,
    TRAIN_FILE,
    TEST_FILE,
    FINAL_MODEL_DIR
)

import torch 
from models.model_loader import load_model, load_tokenizer
from models.peft_setup import get_lora_model
from data.dataset_loader import preprocess_data
from utils.metrics import compute_metrics
from transformers import (
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
    EarlyStoppingCallback
)


def set_device(): 

    device = "cpu" # defaults to cpu device
    
    if torch.backends.mps.is_available():
        print("Model running on Apple Silicon, MPS is available\nSelecting MPS as default device")
        device = torch.device("mps")
        return device

    elif torch.cuda.is_available(): 
        print("CUDA Device found, Defaulting to CUDA")
        device = torch.device("cuda")
        return device

    else: 
        print("No accelarator device found, Defaulting to CPU")
        return device


def main():
    #-----
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(MODEL_ID, HF_TOKEN)

    #-----
    print("Preprocessing dataset...")
    tk_data_train, tk_data_test, label_encoder = preprocess_data(tokenizer, TRAIN_FILE, TEST_FILE)
    num_labels = len(label_encoder.classes_)

    #-----
    print("Loading model...")
    model = load_model(MODEL_ID, HF_TOKEN, num_labels=num_labels)

    if getattr(model.config, "vocab_size", None) and len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id

    #-----
    print("Preparing LoRA model...")
    model = get_lora_model(model)
    model.print_trainable_parameters()


    model.to(gpu_device)
    model.train()

    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir="./logs",
        logging_steps=10,
        save_steps=512,
        save_total_limit=3,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.001,
        eval_strategy="steps",
        eval_steps=256,
        load_best_model_at_end=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        seed=42,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tk_data_train,
        eval_dataset=tk_data_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print("Evaluating...")
    results = trainer.evaluate(tk_data_test)
    print(f"Final results: {results}")


if __name__ == "__main__":
    main()