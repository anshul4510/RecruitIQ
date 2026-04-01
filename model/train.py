import os
import sys
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer

# Add parent directory to path to import features
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.engineer import generate_features_for_dataframe

def train_model(data_path, model_output_path):
    print("Loading Dataset...")
    try:
        df = pd.read_csv(data_path)
        print(f"Original dataset size: {len(df)}")
        # Use a small sample to avoid long running embeddings locally
        df = df.sample(min(200, len(df)), random_state=42)
        print(f"Sampled dataset size: {len(df)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Loading SentenceTransformer model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating features. This may take a while...")
    df_features = generate_features_for_dataframe(df, sbert_model)

    features = [
        'skill_overlap_score', 
        'education_match_score', 
        'responsibility_similarity_score', 
        'language_match_score', 
        'certification_match_score',
        'experience_match_score'
    ]

    X = df_features[features]
    y = df_features['target']

    print("Training XGBoost Regressor...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X, y)

    print(f"Saving model to {model_output_path}...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    
    print("Training completed successfully.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv = os.path.join(base_dir, "data", "raw", "resume.csv")
    out_model = os.path.join(base_dir, "model", "model.pkl")
    train_model(data_csv, out_model)
