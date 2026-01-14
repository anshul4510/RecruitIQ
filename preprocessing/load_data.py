import pandas as pd
from pathlib import Path

def load_resume_data(file_path: str="data/raw/resumes.csv")->pd.DataFrame:
    path=Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df=pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataset is empty")
    return df

if __name__=="__main__":
    df=load_resume_data()
    print("Dataset loaded successfully")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())