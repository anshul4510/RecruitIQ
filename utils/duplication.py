from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def detect_duplicates(resumes_text_list, threshold=0.95):
    """
    Detects duplicate resumes passing a specific similarity threshold.
    Returns a list of tuples (index1, index2, similarity_score).
    """
    if not resumes_text_list or len(resumes_text_list) < 2:
        return []
        
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sbert.encode(resumes_text_list)
    
    sim_matrix = cosine_similarity(embeddings)
    
    duplicates = []
    n = len(resumes_text_list)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= threshold:
                duplicates.append((i, j, float(sim_matrix[i][j])))
                
    return duplicates
