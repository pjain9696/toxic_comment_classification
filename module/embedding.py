import numpy as np
from tqdm import tqdm

def load_pretrained_embeddings(file_path):
    embeddings_index = {}
    f = open(file_path)
    for line in tqdm(f):
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print("Found %s word vectors" % len(embeddings_index))
    return embeddings_index
