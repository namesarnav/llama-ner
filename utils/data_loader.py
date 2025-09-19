
def load_data(filepath):
   
    sentences = []
    labels = []
    words = []
    tags = []

    with open(filepath, "r") as dataset:
        for line in dataset:
            if line != "\n":
                sample = line.strip().split()
                words.append(sample[0])
                tags.append(sample[1])
            else:
                sentences.append(words)
                labels.append(tags)
                words, tags = [], []

    return sentences, labels