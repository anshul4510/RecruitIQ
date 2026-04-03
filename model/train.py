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
        'experience_years_score',
        'projects_count_score',
        'major_match_score',
        'seniority_match_score',
        'skill_breadth_score',
        'job_stability_score',
        'online_presence_score',
        'responsibility_depth_score'
    ]

    X = df_features[features]
    y = df_features['target']

    print("Finding the best parameters using GridSearchCV...")
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    base_model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='r2', # Optimizing for Variance Explained
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best Parameters found for the dataset: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score (R^2): {grid_search.best_score_:.4f}")
    
    model = grid_search.best_estimator_

    print(f"Saving model to {model_output_path}...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    
    print("Training completed successfully.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv = os.path.join(base_dir, "data", "raw", "resume.csv")
    out_model = os.path.join(base_dir, "model", "model.pkl")
    train_model(data_csv, out_model)
