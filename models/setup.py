import yaml
from config import *
from transformers import (
    LlamaForTokenClassification,
    LlamaTokenizer
)

config = yaml.safe_load('config/config.yml')

MODEL = config['model_id']