import pandas as pd
import numpy as np
import ast
import re
from datetime import datetime
try:
    from sentence_transformers import util
except ImportError:
    pass
from sklearn.metrics.pairwise import cosine_similarity
from utils.education_processor import EducationProcessor

# --- UTILITIES ---
def parse_list_string(val):
    if pd.isna(val) or val is None: return []
    if isinstance(val, list): return val
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list): return parsed
        return [str(parsed)]
    except:
        if isinstance(val, str):
            return [line.strip() for line in val.split('\n') if line.strip()]
        return []

def calculate_similarity(text1, text2, model):
    if not text1 or not text2 or pd.isna(text1) or pd.isna(text2): return 0.0
    text1_str = str(text1)
    text2_str = str(text2)
    try:
        emb1 = model.encode(text1_str, convert_to_tensor=True)
        emb2 = model.encode(text2_str, convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2)
        return float(sim[0][0])
    except: return 0.0

def extract_years_from_text(text):
    if not text or pd.isna(text): return 0.0
    nums = re.findall(r'\d+', str(text))
    if nums: return float(nums[0])
    return 0.0

# --- A. EXISTING FEATURES (UNCHANGED 13) ---
def calculate_skill_overlap(resume_skills, jd_skills):
    if not resume_skills or not jd_skills: return 0.0
    synonym_map = {'mysql': 'sql', 'postgresql': 'sql', 'aws': 'amazon web services', 'gcp': 'google cloud', 'ml': 'machine learning'}
    r_skills_lower = set(synonym_map.get(str(s).lower().strip(), str(s).lower().strip()) for s in resume_skills)
    j_skills_lower = set(synonym_map.get(str(s).lower().strip(), str(s).lower().strip()) for s in jd_skills)
    if not j_skills_lower: return 0.0
    return len(r_skills_lower.intersection(j_skills_lower)) / len(j_skills_lower)

def calculate_education_match(resume_degrees, jd_edu):
    return EducationProcessor.calculate_match_score(resume_degrees, jd_edu)

def calculate_exp_years_score(resume_years, jd_req):
    jd_years = extract_years_from_text(jd_req)
    if jd_years == 0: return 1.0
    if resume_years >= jd_years: return 1.0
    return max(0.0, resume_years / jd_years)

def detect_seniority(title):
    if not title: return 0
    t = str(title).lower()
    if any(k in t for k in ['sr', 'senior', 'lead', 'principal', 'head']): return 2
    if any(k in t for k in ['jr', 'junior', 'trainee', 'intern']): return 1
    return 0

def calculate_seniority_match(resume_positions, jd_title):
    jd_snr = detect_seniority(jd_title)
    if jd_snr == 0: return 1.0
    res_snr = detect_seniority(resume_positions[0] if resume_positions else "")
    if res_snr >= jd_snr: return 1.0
    return 0.5 if res_snr == 0 else 0.2

# --- B. SKILL INTELLIGENCE ---
def calc_core_skill_coverage(res_skills, req_skills):
    if not req_skills: return 1.0
    rs = set(s.lower().strip() for s in res_skills)
    rq = set(s.lower().strip() for s in req_skills)
    return len(rs.intersection(rq)) / len(rq) if rq else 0.0

def calc_skill_relevance_score(res_skills, req_skills):
    return calc_core_skill_coverage(res_skills, req_skills)  # Proxy for weights

def calc_skill_context_score(req_skills, responsibilities):
    if not req_skills: return 1.0
    resp_text = " ".join(responsibilities).lower()
    matched = sum(1 for s in req_skills if str(s).lower() in resp_text)
    return matched / len(req_skills)

def calc_skill_frequency_score(res_skills, responsibilities):
    return min(1.0, len(res_skills) / max(1, len(responsibilities)))

def calc_rare_skill_bonus(res_skills):
    rare = {'triton', 'jax', 'ebpf', 'cuda', 'rust', 'golang', 'solidity'}
    rs = set(s.lower().strip() for s in res_skills)
    return min(1.0, len(rs.intersection(rare)) * 0.15)

def calc_skill_recency_score(): return 0.8
def calc_skill_group_match_score(): return 0.7
def calc_skill_depth_score(): return 0.65

# --- C. EXPERIENCE INTELLIGENCE ---
def calc_experience_relevance_score(): return 0.8
def calc_role_progression_score(positions): return 1.0 if len(positions) > 1 else 0.5
def calc_role_similarity_score(): return 0.75
def calc_company_relevance_score(): return 0.5
def calc_experience_gap_penalty(): return 0.0
def calc_leadership_experience_score(responsibilities):
    resp_text = " ".join(responsibilities).lower()
    leaders = ['led', 'managed', 'mentor', 'own', 'cross-functional']
    return min(1.0, sum(1 for l in leaders if l in resp_text) * 0.2)
def calc_role_duration_consistency(): return 0.9

# --- D. RESPONSIBILITY INTELLIGENCE ---
def calc_responsibility_alignment_score(): return 0.8
def calc_responsibility_complexity_score(responsibilities):
    return min(1.0, sum(len(r.split()) for r in responsibilities) / 500.0)
def calc_impact_score(responsibilities):
    resp_text = " ".join(responsibilities).lower()
    impacts = len(re.findall(r'\d+%|\$\d+|reduced|increased|grew|scaled', resp_text))
    return min(1.0, impacts * 0.1)
def calc_action_verb_density(): return 0.6
def calc_responsibility_diversity_score(): return 0.7

# --- E. EDUCATION INTELLIGENCE ---
def calc_degree_level_score(degrees):
    d_text = " ".join(degrees).lower()
    if 'phd' in d_text: return 1.0
    if 'master' or 'ms' in d_text: return 0.85
    if 'bachelor' or 'bs' in d_text: return 0.7
    return 0.5

def calc_education_relevance_score(): return 0.8
def calc_academic_performance_score(): return 0.65
def calc_institution_tier_score(): return 0.4

# --- F. PROJECT INTELLIGENCE ---
def calc_project_relevance_score(): return 0.7
def calc_project_complexity_score(): return 0.6
def calc_project_impact_score(): return 0.5
def calc_project_recency_score(): return 0.8

# --- G. CERTIFICATION INTELLIGENCE ---
def calc_certification_relevance_score(): return 0.5
def calc_certification_authority_score(): return 0.6
def calc_certification_recency_score(): return 0.8

# --- H. SOFT SIGNALS ---
def calc_communication_score(): return 0.8
def calc_initiative_score(): return 0.5
def calc_leadership_signal_score(): return 0.5

# --- I. SEMANTIC FEATURES ---
def calc_resume_jd_embedding_score(r_text, j_text, model):
    return calculate_similarity(r_text, j_text, model)
def calc_skill_embedding_match_score(): return 0.7
def calc_experience_embedding_score(r_exp, j_req, model):
    return calculate_similarity(r_exp, j_req, model)


def engineer_features_for_single(resume_json, jd_json, model, jd_embeddings=None):
    # Base extracted
    r_skills = resume_json.get('skills', [])
    j_skills = jd_json.get('skills_required', [])
    r_edu = resume_json.get('degree_names', [])
    j_edu = jd_json.get('educational_requirements', '')
    r_resp_list = resume_json.get('responsibilities', [])
    r_resp = " ".join(r_resp_list)
    j_resp_list = jd_json.get('responsibilities', [])
    j_resp = " ".join(j_resp_list)
    r_major = " ".join(resume_json.get('major_field_of_studies', []))
    j_pos = jd_json.get('job_position_name', '')
    r_pos = resume_json.get('positions', [])

    df_dict = {}

    # --- 13 ORIGINAL FEATURES ---
    df_dict['skill_overlap_score'] = calculate_skill_overlap(r_skills, j_skills)
    df_dict['education_match_score'] = calculate_education_match(r_edu, j_edu)
    if jd_embeddings and 'responsibilities' in jd_embeddings:
        r_resp_emb = model.encode(r_resp, convert_to_tensor=True)
        df_dict['responsibility_similarity_score'] = float(util.cos_sim(r_resp_emb, jd_embeddings['responsibilities'])[0][0])
    else:
        df_dict['responsibility_similarity_score'] = calculate_similarity(r_resp, j_resp, model)
    df_dict['language_match_score'] = 1.0 if not jd_json.get('languages', []) or resume_json.get('languages') else 0.5
    df_dict['certification_match_score'] = 1.0 if resume_json.get('certification_providers') else 0.2
    df_dict['experience_years_score'] = calculate_exp_years_score(resume_json.get('experience_years', 0), jd_json.get('experience_requirement', ''))
    df_dict['projects_count_score'] = min(float(resume_json.get('projects_count', 0))/10.0, 1.0)
    df_dict['major_match_score'] = calculate_similarity(r_major, j_pos, model)
    df_dict['seniority_match_score'] = calculate_seniority_match(r_pos, j_pos)
    df_dict['skill_breadth_score'] = min(len(r_skills)/20.0, 1.0)
    df_dict['job_stability_score'] = 0.8
    df_dict['online_presence_score'] = 1.0 if resume_json.get('online_links') else 0.0
    df_dict['responsibility_depth_score'] = min(len(r_resp)/1000.0, 1.0)

    # --- B. SKILL INTELLIGENCE (8) ---
    df_dict['core_skill_coverage'] = calc_core_skill_coverage(r_skills, j_skills)
    df_dict['skill_relevance_score'] = calc_skill_relevance_score(r_skills, j_skills)
    df_dict['skill_context_score'] = calc_skill_context_score(j_skills, r_resp_list)
    df_dict['skill_frequency_score'] = calc_skill_frequency_score(r_skills, r_resp_list)
    df_dict['rare_skill_bonus'] = calc_rare_skill_bonus(r_skills)
    df_dict['skill_recency_score'] = calc_skill_recency_score()
    df_dict['skill_group_match_score'] = calc_skill_group_match_score()
    df_dict['skill_depth_score'] = calc_skill_depth_score()

    # --- C. EXPERIENCE INTELLIGENCE (7) ---
    df_dict['experience_relevance_score'] = calc_experience_relevance_score()
    df_dict['role_progression_score'] = calc_role_progression_score(r_pos)
    df_dict['role_similarity_score'] = calc_role_similarity_score()
    df_dict['company_relevance_score'] = calc_company_relevance_score()
    df_dict['experience_gap_penalty'] = calc_experience_gap_penalty()
    df_dict['leadership_experience_score'] = calc_leadership_experience_score(r_resp_list)
    df_dict['role_duration_consistency'] = calc_role_duration_consistency()

    # --- D. RESPONSIBILITY INTELLIGENCE (5) ---
    df_dict['responsibility_alignment_score'] = calc_responsibility_alignment_score()
    df_dict['responsibility_complexity_score'] = calc_responsibility_complexity_score(r_resp_list)
    df_dict['impact_score'] = calc_impact_score(r_resp_list)
    df_dict['action_verb_density'] = calc_action_verb_density()
    df_dict['responsibility_diversity_score'] = calc_responsibility_diversity_score()

    # --- E. EDUCATION INTELLIGENCE (4) ---
    df_dict['degree_level_score'] = calc_degree_level_score(r_edu)
    df_dict['education_relevance_score'] = calc_education_relevance_score()
    df_dict['academic_performance_score'] = calc_academic_performance_score()
    df_dict['institution_tier_score'] = calc_institution_tier_score()

    # --- F. PROJECT INTELLIGENCE (4) ---
    df_dict['project_relevance_score'] = calc_project_relevance_score()
    df_dict['project_complexity_score'] = calc_project_complexity_score()
    df_dict['project_impact_score'] = calc_project_impact_score()
    df_dict['project_recency_score'] = calc_project_recency_score()

    # --- G. CERTIFICATION INTELLIGENCE (3) ---
    df_dict['certification_relevance_score'] = calc_certification_relevance_score()
    df_dict['certification_authority_score'] = calc_certification_authority_score()
    df_dict['certification_recency_score'] = calc_certification_recency_score()

    # --- H. SOFT SIGNALS (3) ---
    df_dict['communication_score'] = calc_communication_score()
    df_dict['initiative_score'] = calc_initiative_score()
    df_dict['leadership_signal_score'] = calc_leadership_signal_score()

    # --- I. SEMANTIC FEATURES (3) ---
    df_dict['resume_jd_embedding_score'] = calc_resume_jd_embedding_score(r_resp, j_resp, model)
    df_dict['skill_embedding_match_score'] = calc_skill_embedding_match_score()
    df_dict['experience_embedding_score'] = calc_experience_embedding_score(r_resp, str(jd_json.get('requirements', '')), model)

    return pd.DataFrame([df_dict])

def generate_features_for_dataframe(df, model):
    # In a full deployment, applying element-wise map.
    # For now, we return empty structure for mock training compat if needed
    features_columns = [
        'skill_overlap_score', 'education_match_score', 'responsibility_similarity_score', 
        'language_match_score', 'certification_match_score', 'experience_years_score',
        'projects_count_score', 'major_match_score', 'seniority_match_score',
        'skill_breadth_score', 'job_stability_score', 'online_presence_score', 'responsibility_depth_score',
        'core_skill_coverage', 'skill_relevance_score', 'skill_context_score', 'skill_frequency_score',
        'rare_skill_bonus', 'skill_recency_score', 'skill_group_match_score', 'skill_depth_score',
        'experience_relevance_score', 'role_progression_score', 'role_similarity_score', 'company_relevance_score',
        'experience_gap_penalty', 'leadership_experience_score', 'role_duration_consistency',
        'responsibility_alignment_score', 'responsibility_complexity_score', 'impact_score',
        'action_verb_density', 'responsibility_diversity_score',
        'degree_level_score', 'education_relevance_score', 'academic_performance_score', 'institution_tier_score',
        'project_relevance_score', 'project_complexity_score', 'project_impact_score', 'project_recency_score',
        'certification_relevance_score', 'certification_authority_score', 'certification_recency_score',
        'communication_score', 'initiative_score', 'leadership_signal_score',
        'resume_jd_embedding_score', 'skill_embedding_match_score', 'experience_embedding_score'
    ]
    
    df_feat = pd.DataFrame(0.0, index=np.arange(len(df)), columns=features_columns)
    if 'matched_score' in df.columns:
        df_feat['target'] = df['matched_score'].fillna(0)
    else:
        df_feat['target'] = 0.0
    return df_feat
