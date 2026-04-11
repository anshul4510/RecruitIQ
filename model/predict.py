import os
import sys
import joblib
import pandas as pd
import time
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.engineer import engineer_features_for_single
from ranking.scorer import hard_filter

class ResumeScorerModel:
    def __init__(self, model_path=None):
        if not model_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "model", "model.pkl")
        
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            print(f"Model not loaded. Using fallback feature aggregations. Error: {e}")
            self.model = None
            
        try:
            from sentence_transformers import SentenceTransformer
            self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.sbert = None
        
    def early_rejection_check(self, resume_text: str, jd_text: str) -> bool:
        if not resume_text or not jd_text or not self.sbert: return False
        try:
            res_emb = self.sbert.encode(resume_text[:1000])
            jd_emb = self.sbert.encode(jd_text[:1000])
            sim = np.dot(res_emb, jd_emb) / (np.linalg.norm(res_emb) * np.linalg.norm(jd_emb))
            return bool(sim < 0.10)
        except: return False
        
    def predict_score(self, resume_json, jd_json, jd_embeddings=None):
        t0 = time.time()
        
        # 1. Hard Filter Stage
        status, score_mod, reasons = hard_filter(resume_json, jd_json)
        if status == "REJECT":
            return 0.0, {"rejection_reasons": reasons}, 0.0, 0.0
            
        # 2. Feature Extraction
        df_features = engineer_features_for_single(resume_json, jd_json, self.sbert, jd_embeddings=jd_embeddings)
        t1 = time.time()
        emb_time = t1 - t0
        
        # 3. Model Inference
        ml_score = 0.5
        if self.model and hasattr(self.model, 'predict'):
            try:
                # Need to drop target if present or ensure exact columns
                cols = self.model.feature_name_ if hasattr(self.model, 'feature_name_') else df_features.columns
                raw_pred = self.model.predict(df_features[cols])[0]
                # lightgbm ranker margin, wrap with sigmoid for 0-1
                ml_score = 1 / (1 + math.exp(-raw_pred))
            except: pass
        t2 = time.time()
        xgb_time = t2 - t1
        
        feat_dict = df_features.iloc[0].to_dict()
        
        # 4. Final Scoring Weights composition:
        existing_13 = ['skill_overlap_score', 'education_match_score', 'responsibility_similarity_score', 'language_match_score', 'certification_match_score', 'experience_years_score', 'projects_count_score', 'major_match_score', 'seniority_match_score', 'skill_breadth_score', 'job_stability_score', 'online_presence_score', 'responsibility_depth_score']
        semantic_3 = ['resume_jd_embedding_score', 'skill_embedding_match_score', 'experience_embedding_score']
        
        f_existing = sum(feat_dict.get(k, 0) for k in existing_13) / len(existing_13)
        f_semantic = sum(feat_dict.get(k, 0) for k in semantic_3) / len(semantic_3)
        
        new_feats = [k for k in feat_dict.keys() if k not in existing_13 and k not in semantic_3 and k != 'target']
        f_new = sum(feat_dict.get(k, 0) for k in new_feats) / max(1, len(new_feats))
        
        # W1: 0.20, W2: 0.25, W3: 0.20, W4: 0.20 (Remaining 0.15 for LLM later)
        base_composite = (0.20 * f_existing) + (0.25 * f_new) + (0.20 * f_semantic) + (0.20 * ml_score)
        
        # We scale base_composite up slightly since LLM (0.15) hasn't been added yet (we want output ~0-1)
        base_composite = base_composite * (1.0 / 0.85)
        
        final_score = max(min(float(base_composite * score_mod), 1.0), 0.0)
        
        return final_score, feat_dict, emb_time, xgb_time
