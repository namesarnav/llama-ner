import yaml
from transformers import (
    LlamaForTokenClassification,
    LlamaTokenizer
)

config = yaml.safe_load('../config/config.yml')

MODEL = config['model_id']
TOKENIZER = config['tokenizer_id']


def load_base_model(MODEL, num_labels):
    return LlamaForTokenClassification.from_pretrained(MODEL, num_labels=num_labels)

def load_tokenizer(TOKENIZER):
    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER)