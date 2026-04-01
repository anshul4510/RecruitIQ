import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def parse_list_string(val):
    if pd.isna(val) or val is None:
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except:
        # Fallback for plain text, e.g. newline separated
        if isinstance(val, str):
            return [line.strip() for line in val.split('\n') if line.strip()]
        return []

def calculate_skill_overlap(resume_skills, jd_skills):
    if not resume_skills or not jd_skills:
        return 0.0
    
    r_skills_lower = set(str(s).lower().strip() for s in resume_skills)
    j_skills_lower = set(str(s).lower().strip() for s in jd_skills)
    
    if not j_skills_lower:
        return 0.0
    
    overlap = r_skills_lower.intersection(j_skills_lower)
    return len(overlap) / len(j_skills_lower)

def calculate_education_match(resume_degrees, jd_edu):
    if not resume_degrees:
        return 0.0
    if pd.isna(jd_edu) or not str(jd_edu).strip():
        # If no strict requirement but they have a degree
        return 1.0 
    
    jd_edu_str = str(jd_edu).lower()
    match_score = 0.0
    
    for degree in resume_degrees:
        if not degree:
            continue
        degree_str = str(degree).lower()
        if degree_str in jd_edu_str or jd_edu_str in degree_str:
            return 1.0
        
        # Check general keywords
        if 'bachelor' in jd_edu_str and ('b.sc' in degree_str or 'b.tech' in degree_str or 'b.a' in degree_str or 'bba' in degree_str or 'bca' in degree_str):
            match_score = max(match_score, 0.8)
        if 'master' in jd_edu_str and ('m.sc' in degree_str or 'm.tech' in degree_str or 'm.a' in degree_str or 'mba' in degree_str or 'mca' in degree_str):
            match_score = max(match_score, 0.8)
            
    return match_score

def calculate_similarity(text1, text2, model):
    if not text1 or not text2:
        return 0.0
    if pd.isna(text1) or pd.isna(text2):
        return 0.0
        
    try:
        embeddings1 = model.encode([str(text1)])
        embeddings2 = model.encode([str(text2)])
        sim = cosine_similarity(embeddings1, embeddings2)
        return float(sim[0][0])
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def generate_features_for_dataframe(df, model):
    """
    Given a raw DataFrame (like our resume.csv), process and generate numeric features.
    """
    df_feat = df.copy()
    
    df_feat['parsed_skills_resume'] = df_feat['skills'].apply(parse_list_string)
    df_feat['parsed_skills_jd'] = df_feat['skills_required'].apply(parse_list_string)
    
    df_feat['skill_overlap_score'] = df_feat.apply(
        lambda row: calculate_skill_overlap(row['parsed_skills_resume'], row['parsed_skills_jd']), axis=1)
    
    df_feat['parsed_degrees'] = df_feat['degree_names'].apply(parse_list_string)
    df_feat['education_match_score'] = df_feat.apply(
        lambda row: calculate_education_match(row['parsed_degrees'], row['educationaL_requirements']), axis=1)
    
    df_feat['responsibility_similarity_score'] = df_feat.apply(
        lambda row: calculate_similarity(row.get('responsibilities', ''), row.get('responsibilities.1', ''), model), axis=1)
    
    df_feat['parsed_languages'] = df_feat['languages'].apply(parse_list_string)
    df_feat['language_match_score'] = df_feat['parsed_languages'].apply(lambda x: 1.0 if x else 0.0)
    
    df_feat['parsed_certs'] = df_feat['certification_providers'].apply(parse_list_string)
    df_feat['certification_match_score'] = df_feat['parsed_certs'].apply(lambda x: 1.0 if x else 0.0)
    
    # We lack exact structured years of experience extraction from raw CSV, so we use a dummy logic
    # In a full pipeline, we'd extract numeric years from experience_requirement and professional dates.
    # We will simulate experience_match_score using a baseline of 0.8 if start_dates exists.
    df_feat['experience_match_score'] = df_feat['start_dates'].apply(lambda x: 0.8 if pd.notna(x) else 0.0)
    
    # Target label: matched_score
    if 'matched_score' in df_feat.columns:
        df_feat['target'] = df_feat['matched_score'].fillna(0)
    else:
        df_feat['target'] = 0.0
        
    features_columns = [
        'skill_overlap_score', 
        'education_match_score', 
        'responsibility_similarity_score', 
        'language_match_score', 
        'certification_match_score',
        'experience_match_score'
    ]
    
    for c in features_columns:
        df_feat[c] = df_feat[c].fillna(0.0)
        
    return df_feat[features_columns + ['target']]

def engineer_features_for_single(resume_json, jd_json, model):
    """
    Computes features for a single resume parsed JSON vs JD parsed JSON.
    Used during live inference.
    """
    r_skills = resume_json.get('skills', [])
    j_skills = jd_json.get('skills_required', [])
    skill_score = calculate_skill_overlap(r_skills, j_skills)
    
    r_edu = resume_json.get('degree_names', [])
    j_edu = jd_json.get('educational_requirements', '')
    edu_score = calculate_education_match(r_edu, j_edu)
    
    r_resp = " ".join(resume_json.get('responsibilities', []))
    j_resp = " ".join(jd_json.get('responsibilities', []))
    resp_score = calculate_similarity(r_resp, j_resp, model)
    
    lang_score = 1.0 if resume_json.get('languages') else 0.0
    cert_score = 1.0 if resume_json.get('certification_providers') else 0.0
    
    # Live inference mock exp
    exp_score = 0.8 if resume_json.get('positions') else 0.0
    
    return pd.DataFrame([{
        'skill_overlap_score': skill_score,
        'education_match_score': edu_score,
        'responsibility_similarity_score': resp_score,
        'language_match_score': lang_score,
        'certification_match_score': cert_score,
        'experience_match_score': exp_score
    }])
