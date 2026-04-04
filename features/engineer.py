import pandas as pd
import numpy as np
import ast
import re
from datetime import datetime
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
        
    synonym_map = {
        'mysql': 'sql',
        'postgresql': 'sql',
        'postgres': 'sql',
        'mssql': 'sql',
        'ms sql': 'sql',
        'sql server': 'sql',
        'reactjs': 'react',
        'react.js': 'react',
        'nodejs': 'node',
        'node.js': 'node',
        'ts': 'typescript',
        'js': 'javascript',
        'aws': 'amazon web services',
        'gcp': 'google cloud',
        'google cloud platform': 'google cloud',
        'ml': 'machine learning',
        'dl': 'deep learning',
        'ai': 'artificial intelligence',
        'nlp': 'natural language processing',
        'cv': 'computer vision'
    }
    
    def normalize_skill(skill):
        s = str(skill).lower().strip()
        return synonym_map.get(s, s)
    
    r_skills_lower = set(normalize_skill(s) for s in resume_skills)
    j_skills_lower = set(normalize_skill(s) for s in jd_skills)
    
    if not j_skills_lower:
        return 0.0
    
    overlap = r_skills_lower.intersection(j_skills_lower)
    return len(overlap) / len(j_skills_lower)

from utils.education_processor import EducationProcessor

def calculate_education_match(resume_degrees, jd_edu):
    """
    Expert education matching logic using the redesigned EducationProcessor.
    Handles hierarchical matching and field similarity.
    """
    return EducationProcessor.calculate_match_score(resume_degrees, jd_edu)

def extract_years_from_text(text):
    if not text or pd.isna(text):
        return 0.0
    nums = re.findall(r'\d+', str(text))
    if nums:
        return float(nums[0])
    return 0.0

def calculate_exp_years_score(resume_years, jd_requirement_text):
    try:
        jd_years = extract_years_from_text(jd_requirement_text)
        if jd_years == 0:
            return 1.0
        
        if resume_years >= jd_years:
            return 1.0
        else:
            # Linear decay for lack of experience
            score = resume_years / jd_years
            return max(0.0, score)
    except:
        return 0.0

def detect_seniority(title):
    if not title: return 0
    t = str(title).lower()
    if any(k in t for k in ['sr', 'senior', 'lead', 'principal', 'head']): return 2
    if any(k in t for k in ['jr', 'junior', 'trainee', 'intern', 'fresher']): return 1
    return 0

def calculate_seniority_match(resume_positions, jd_title):
    jd_seniority = detect_seniority(jd_title)
    if jd_seniority == 0: return 1.0 # Role is level agnostic
    
    recent_position = resume_positions[0] if isinstance(resume_positions, list) and resume_positions else ""
    resume_seniority = detect_seniority(recent_position)
    
    if resume_seniority >= jd_seniority: return 1.0
    if resume_seniority == 0 and jd_seniority > 0: return 0.5 # Unknown seniority
    return 0.2 # Mismatch

def calculate_job_stability(start_dates, end_dates):
    # Calculate average months at each company
    # Simplified logic for now: count entries or use dates if available
    if not start_dates or not end_dates: return 0.5
    
    # We'll just proxy stability by number of roles vs total extracted years if possible
    # but more robustly we could compare dates. 
    # For now, let's use a conservative baseline
    return 0.8

def calculate_skill_recency(resume_end_dates, skills):
    # If skills were used in the most recent job (e.g. empty end_date or modern year)
    if not resume_end_dates: return 0.5
    return 1.0 # Mock recency for now

_embedding_cache = {}

def calculate_similarity(text1, text2, model):
    if not text1 or not text2:
        return 0.0
    if pd.isna(text1) or pd.isna(text2):
        return 0.0
        
    text1_str = str(text1)
    text2_str = str(text2)
        
    try:
        from sentence_transformers import util
        if text1_str not in _embedding_cache:
            _embedding_cache[text1_str] = model.encode(text1_str, convert_to_tensor=True)
        if text2_str not in _embedding_cache:
            _embedding_cache[text2_str] = model.encode(text2_str, convert_to_tensor=True)
            
        emb1 = _embedding_cache[text1_str]
        emb2 = _embedding_cache[text2_str]
        
        sim = util.cos_sim(emb1, emb2)
        return float(sim[0][0])
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def calculate_total_years(start_dates, end_dates):
    if not start_dates or not end_dates:
        return 0.0
    try:
        s_list = parse_list_string(start_dates)
        e_list = parse_list_string(end_dates)
        
        total_months = 0
        for s, e in zip(s_list, e_list):
            if not s or not e: continue
            
            # Simple handle for "Till Date"
            if 'till date' in str(e).lower() or 'present' in str(e).lower():
                e_obj = datetime.now()
            else:
                try:
                    e_obj = pd.to_datetime(e)
                except: continue
                
            try:
                s_obj = pd.to_datetime(s)
                diff = e_obj - s_obj
                total_months += diff.days / 30.44
            except: continue
            
        return round(total_months / 12.0, 1)
    except:
        return 0.0

def generate_features_for_dataframe(df, model):
    df_feat = df.copy()
    
    # Handle missing columns in raw dataset
    if 'projects_count' not in df_feat.columns:
        df_feat['projects_count'] = 0
    if 'experience_years' not in df_feat.columns:
        df_feat['experience_years'] = df_feat.apply(lambda row: calculate_total_years(row.get('start_dates'), row.get('end_dates')), axis=1)
    if 'online_links' not in df_feat.columns:
        df_feat['online_links'] = ""

    # Core Overlap Features
    df_feat['parsed_skills_resume'] = df_feat['skills'].apply(parse_list_string)
    df_feat['parsed_skills_jd'] = df_feat['skills_required'].apply(parse_list_string)
    df_feat['skill_overlap_score'] = df_feat.apply(
        lambda row: calculate_skill_overlap(row['parsed_skills_resume'], row['parsed_skills_jd']), axis=1)
    
    df_feat['parsed_degrees'] = df_feat['degree_names'].apply(parse_list_string)
    df_feat['education_match_score'] = df_feat.apply(
        lambda row: calculate_education_match(row['parsed_degrees'], row.get('educational_requirements', row.get('educationaL_requirements', ''))), axis=1)
    
    # Text Similarity
    df_feat['responsibility_similarity_score'] = df_feat.apply(
        lambda row: calculate_similarity(row.get('responsibilities', ''), row.get('responsibilities.1', ''), model), axis=1)
    
    # Experience Logic
    df_feat['experience_years_score'] = df_feat.apply(
        lambda row: calculate_exp_years_score(row.get('experience_years', 0), row.get('experiencere_requirement', '')), axis=1)
    
    # Supplemental Features
    df_feat['parsed_languages'] = df_feat['languages'].apply(parse_list_string)
    df_feat['language_match_score'] = df_feat['parsed_languages'].apply(lambda x: 1.0 if x else 0.5)
    
    df_feat['parsed_certs_provider'] = df_feat['certification_providers'].apply(parse_list_string)
    df_feat['certification_match_score'] = df_feat.apply(lambda row: 0.8 if row['parsed_certs_provider'] else 0.2, axis=1)
    
    # New Advanced Features
    df_feat['projects_count_score'] = df_feat['projects_count'].apply(lambda x: min(float(x or 0)/10.0, 1.0))
    
    df_feat['major_match_score'] = df_feat.apply(
        lambda row: calculate_similarity(row.get('major_field_of_studies', ''), row.get('job_position_name', ''), model), axis=1)
    
    df_feat['seniority_match_score'] = df_feat.apply(
        lambda row: calculate_seniority_match(parse_list_string(row.get('positions', [])), row.get('job_position_name', '')), axis=1)
        
    df_feat['skill_breadth_score'] = df_feat['parsed_skills_resume'].apply(lambda x: min(len(x)/20.0, 1.0))
    
    df_feat['job_stability_score'] = df_feat.apply(
        lambda row: calculate_job_stability(row.get('start_dates'), row.get('end_dates')), axis=1)
        
    df_feat['online_presence_score'] = df_feat['online_links'].apply(lambda x: 1.0 if x and str(x) != '[]' and str(x) != 'nan' else 0.0)
    
    def get_resp_depth(x):
        try:
            return min(len(str(x))/1000.0, 1.0)
        except: return 0.2
    df_feat['responsibility_depth_score'] = df_feat['responsibilities'].apply(get_resp_depth)

    # Existing dummy for backward compatibility
    df_feat['experience_match_score'] = df_feat['experience_years_score'] 
    
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
        'experience_years_score',
        'projects_count_score',
        'major_match_score',
        'seniority_match_score',
        'skill_breadth_score',
        'job_stability_score',
        'online_presence_score',
        'responsibility_depth_score'
    ]
    
    for c in features_columns:
        df_feat[c] = df_feat[c].fillna(0.0)
        
    return df_feat[features_columns + ['target']]

def engineer_features_for_single(resume_json, jd_json, model, jd_embeddings=None):
    r_skills = resume_json.get('skills', [])
    j_skills = jd_json.get('skills_required', [])
    skill_score = calculate_skill_overlap(r_skills, j_skills)
    
    r_edu = resume_json.get('degree_names', [])
    j_edu = jd_json.get('educational_requirements', '')
    edu_score = calculate_education_match(r_edu, j_edu)
    
    r_resp = " ".join(resume_json.get('responsibilities', []))
    j_resp = " ".join(jd_json.get('responsibilities', []))
    
    # Use pre-calculated JD embeddings if available
    if jd_embeddings and 'responsibilities' in jd_embeddings:
        r_resp_emb = model.encode(r_resp, convert_to_tensor=True)
        j_resp_emb = jd_embeddings['responsibilities']
        from sentence_transformers import util
        resp_score = float(util.cos_sim(r_resp_emb, j_resp_emb)[0][0])
    else:
        resp_score = calculate_similarity(r_resp, j_resp, model)
    
    exp_years_score = calculate_exp_years_score(resume_json.get('experience_years', 0), jd_json.get('experience_requirement', ''))
    
    j_lang = jd_json.get('languages', [])
    lang_score = 1.0 if not j_lang or resume_json.get('languages') else 0.5
    
    cert_score = 1.0 if resume_json.get('certification_providers') else 0.2
    
    projects_score = min(float(resume_json.get('projects_count', 0))/10.0, 1.0)
    
    r_major = " ".join(resume_json.get('major_field_of_studies', []))
    j_pos = jd_json.get('job_position_name', '')
    
    # Use pre-calculated JD embeddings if available
    if jd_embeddings and 'job_position_name' in jd_embeddings:
        r_major_emb = model.encode(r_major, convert_to_tensor=True)
        j_pos_emb = jd_embeddings['job_position_name']
        from sentence_transformers import util
        major_score = float(util.cos_sim(r_major_emb, j_pos_emb)[0][0])
    else:
        major_score = calculate_similarity(r_major, j_pos, model)
    
    seniority_score = calculate_seniority_match(resume_json.get('positions', []), jd_json.get('job_position_name', ''))
    
    breadth_score = min(len(r_skills)/20.0, 1.0)
    
    stability_score = calculate_job_stability(resume_json.get('start_dates'), resume_json.get('end_dates'))
    
    online_score = 1.0 if resume_json.get('online_links') else 0.0
    
    depth_score = min(len(str(resume_json.get('responsibilities', [])))/1000.0, 1.0)
    
    return pd.DataFrame([{
        'skill_overlap_score': skill_score,
        'education_match_score': edu_score,
        'responsibility_similarity_score': resp_score,
        'language_match_score': lang_score,
        'certification_match_score': cert_score,
        'experience_years_score': exp_years_score,
        'projects_count_score': projects_score,
        'major_match_score': major_score,
        'seniority_match_score': seniority_score,
        'skill_breadth_score': breadth_score,
        'job_stability_score': stability_score,
        'online_presence_score': online_score,
        'responsibility_depth_score': depth_score
    }])
