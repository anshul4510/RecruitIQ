import faiss
import numpy as np

class ResumeSimilarityEngine:
    def __init__(self,embeddings:np.ndarray):
        self.embeddings=embeddings
        self.dimension=embeddings.shape[1]

        self.index=faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        scores,indices=self.index.search(query_embedding,top_k)
        return scores[0],indices[0]
