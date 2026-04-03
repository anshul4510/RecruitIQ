import joblib
import pandas as pd
import os

base_dir = r"c:\Users\DELL\Documents\UdemyDataEng\udemyGENAI\Projects\ResumeScreening"
model_path = os.path.join(base_dir, "model", "model.pkl")

print(f"Loading model from {model_path}...")
model = joblib.load(model_path)

if hasattr(model, 'feature_names_in_'):
    print("Feature names in model:", model.feature_names_in_)
else:
    print("Model does not have feature_names_in_ attribute.")
    # For XGBoost
    try:
        booster = model.get_booster()
        print("Feature names from booster:", booster.feature_names)
    except:
        print("Could not get booster feature names.")
