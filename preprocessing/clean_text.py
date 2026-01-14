import re
import pandas as pd

def normalize_text(text:str)->str:
    if pd.isna(text):
        return ""
    
    text=text.lower()
    text=re.sub(r"[^a-z0-9,\s]"," ",text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()

def clean_skills(skills:str)->list:
    skills=normalize_text(skills)
    skill_list=[skill.strip() for skill in skills.split(",") if skill.strip()]
    return skill_list

def clean_column(df: pd.DataFrame,column_name:str)->pd.Series:
    return df[column_name].apply(normalize_text)

