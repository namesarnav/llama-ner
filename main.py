from transformers import Trainer, TrainingArguments
from models.model_loader import load_model, load_tokenizer
from utils import * 
from models import get_lora_model
from transformers import DataCollatorForTokenClassification
from datasets import Dataset
from .config import *

from dotenv import load_dotenv
load_dotenv()
import os

train_data = load_data('/data/train.txt')
test_data = load_data('/data/test.txt')
HF_TOKEN = os.getenv('HF_TOKEN')

tokenizer = load_tokenizer()

model = load_model(
    model_id=MODEL,
    num_labels=len(set(tag for tags in train_data['tags'] for tag in tags)),
    id2label={i: tag for i, tag in enumerate(set(tag for tags in train_data['tags'] for tag in tags))},
    label2id={tag: i for i, tag in enumerate(set(tag for tags in train_data['tags'] for tag in tags))},
    dtype='float16',
    device_map="auto",
    load_in_4bit=True,
    token=HF_TOKEN,
)

model = get_lora_model(model)

device = set_device()
model.to(device)

data_collator = DataCollatorForTokenClassification(tokenizer)

compute_metrics = build_compute_metrics(tokenizer)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=train_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


