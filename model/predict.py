import os
import sys
import joblib
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
import sys
import os

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.engineer import engineer_features_for_single
import numpy as np

class ResumeScorerModel:
    def __init__(self, model_path=None):
        if not model_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "model", "model.pkl")
        
        try:
            print(f"DEBUG: Loading model from {model_path}")
            self.model = joblib.load(model_path)
            if hasattr(self.model, 'feature_names_in_'):
                print(f"DEBUG: Model feature names: {self.model.feature_names_in_}")
        except Exception as e:
            print(f"Could not load model from {model_path}. Did you run model/train.py? Error: {e}")
            self.model = None
            
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        
    def early_rejection_check(self, resume_text: str, jd_text: str) -> bool:
        """
        Fast cosine similarity check. If the candidate's first 1000 bytes 
        are completely irrelevant to the JD, reject instantly.
        """
        if not resume_text or not jd_text:
            return False
            
        # Fast extraction sample
        res_sample = resume_text[:1000]
        jd_sample = jd_text[:1000]
        
        res_emb = self.sbert.encode(res_sample)
        jd_emb = self.sbert.encode(jd_sample)
        
        sim = np.dot(res_emb, jd_emb) / (np.linalg.norm(res_emb) * np.linalg.norm(jd_emb))
        return bool(sim < 0.10)
        
    def predict_score(self, resume_json, jd_json, jd_embeddings=None):
        if not self.model:
            return 0.0, {}, 0.0, 0.0
            
        t0 = time.time()
        df_features = engineer_features_for_single(resume_json, jd_json, self.sbert, jd_embeddings=jd_embeddings)
        t1 = time.time()
        
        score = self.model.predict(df_features)[0]
        t2 = time.time()
        
        # Cap score between 0 and 1
        final_score = max(min(float(score), 1.0), 0.0)
        
        embedding_time = t1 - t0
        xgboost_time = t2 - t1
        
        return final_score, df_features.iloc[0].to_dict(), embedding_time, xgboost_time
