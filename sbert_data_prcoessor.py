import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


DATASET_FILENAME = "nn_dataset.csv"


df = pd.read_csv(DATASET_FILENAME)
labels = np.array(list(df["class"]))
sentences = df["sentence"]

sbert_encoder = SentenceTransformer('all-MiniLM-L6-v2')
sbert_embeddings = sbert_encoder.encode(sentences)

assert len(sbert_embeddings) == len(labels)
print(sbert_embeddings.shape)

np.save("sbert_embeddings.npy", sbert_embeddings)
