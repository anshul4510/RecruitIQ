from sentence_transformers import SentenceTransformer
import numpy as np

class ResumeEmbedder:
    def __init__(self,model_name: str="all-MiniLM-L6-v2"):
        self.model=SentenceTransformer(model_name)
    
    def embed_texts(self,texts:list[str])->np.ndarray:
        embeddings=self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings