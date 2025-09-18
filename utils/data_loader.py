import os
import pandas

from datasets import Dataset

def read_conll(filepath):
    sentences, labels, sent, labs = [], [], [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sentences.append(sent)
                    labels.append(labs)
                    sent, labs = [], []
            else:
                token, tag = line.split()
                sent.append(token)
                labs.append(tag)
    return sentences, labels

def load_dataset(train_file, test_file):
    train_sentences, train_labels = read_conll(train_file)
    test_sentences, test_labels = read_conll(test_file)

    train_data = Dataset.from_dict({"tokens": train_sentences, "ner_tags": train_labels})
    test_data = Dataset.from_dict({"tokens": test_sentences, "ner_tags": test_labels})

    return train_data, test_data
    





