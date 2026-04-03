import os
import sys
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.engineer import engineer_features_for_single

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
        
    def predict_score(self, resume_json, jd_json):
        if not self.model:
            return 0.0, {}
            
        df_features = engineer_features_for_single(resume_json, jd_json, self.sbert)
        score = self.model.predict(df_features)[0]
        # Cap score between 0 and 1
        final_score = max(min(float(score), 1.0), 0.0)
        return final_score, df_features.iloc[0].to_dict()
