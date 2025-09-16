from transformers import ( 
    LlamaTokenizer,
    BitsAndBytesConfig,
    LlamaForTokenClassification
)

bnb_conf = BitsAndBytesConfig(
    bnb_4bit_compute_dtype=
)