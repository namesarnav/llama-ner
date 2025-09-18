# import numpy as np
# import pandas as pd
# import pyarrow as pa


def load_data(filepath):
    '''
    build sentences
    '''
    words, tags = [], []
    sentence, labels = [], []
    
    with open(filepath, "r") as dataset:

        for line in dataset:
            
            if line:
                sample = line.strip().split()
                inp, label = sample[0], sample[1]
                words.append(inp)
                tags.append(label)

            # if condition not met, means there's no line, so append the word list to sentence    
            sentence.append(words)
            labels.append(tags)
            words, tags = [], []
            