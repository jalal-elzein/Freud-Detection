# THIS CODE HAS NOT BEEN TESTED IN ISOLATION

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from gensim.models import Word2Vec


DATASET_FILENAME = "nn_dataset.csv"


df = pd.read_csv(DATASET_FILENAME)
labels = np.array(list(df["class"]))
sentences = df["sentence"]


df['tokenized_sentences'] = df['sentence'].apply(lambda x: x.split() if not isinstance(x, float) else [] )
data = df['tokenized_sentences'].tolist()


model = Word2Vec(data, window=5, min_count=1, workers=4)


def sentence_embedding(sentence):
    # Check if the sentence is a float or int, and if so, return a zero vector
    if isinstance(sentence, (float, int)):
        return np.zeros(384)
    words = sentence.split()
    word_embeddings = [model.wv[word] for word in words if word in model.wv]
    if len(word_embeddings) == 0:
        return np.zeros(384)
    # Averaging the word vectors to create a sentence vector
    sentence_embedding = np.mean(word_embeddings, axis=0)
    return sentence_embedding


embeddings = []

for s in sentences:
    embeddings.append(sentence_embedding(s))

embeddings = np.ndarray(embeddings)

np.save("w2v_embeddings.npy", embeddings)
