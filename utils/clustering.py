from sklearn.cluster import KMeans
import pandas as pd
from sentence_transformers import SentenceTransformer

def cluster_resumes(resumes_text_list, num_clusters=3):
    """
    Clusters a list of resume texts into groups using KMeans and SentenceTransformers.
    """
    if not resumes_text_list:
        return []
    if len(resumes_text_list) < num_clusters:
        return [0] * len(resumes_text_list)
        
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sbert.encode(resumes_text_list)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters.tolist()
