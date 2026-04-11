import logging
logger = logging.getLogger(__name__)

def hard_filter(resume_json, jd_json):
    """
    Applies strict ATS hard-filtering rules before ranking.
    Returns (status, score_modifier, rejection_reasons)
    """
    score_modifier = 1.0
    rejection_reasons = []
    
    r_skills = set(s.lower() for s in resume_json.get('skills', []))
    # Skill filtering removed as per user request. 
    # Critical skills now influence the 50-feature score but do not cause immediate rejection.

    res_exp = resume_json.get('experience_years', 0)
    try:
        from features.engineer import extract_years_from_text
        req_exp = extract_years_from_text(jd_json.get('experience_requirement', '0'))
    except dict:
        req_exp = 0.0
        
    if res_exp < req_exp - 1:
        score_modifier *= 0.70
        rejection_reasons.append("Experience below threshold")
    elif res_exp < req_exp:
        score_modifier *= 0.90

    from features.engineer import calc_degree_level_score
    r_deg_score = calc_degree_level_score(resume_json.get('degree_names', []))
    j_deg_score = calc_degree_level_score([jd_json.get('educational_requirements', 'Bachelors')])
    
    if r_deg_score < j_deg_score:
        if jd_json.get('degree_strict', False):
            return "REJECT", 0.0, ["Education requirement not met"]
        else:
            score_modifier *= 0.85

    return "PASS", score_modifier, rejection_reasons

def rank_candidates(candidates):
    sorted_list = sorted(candidates, key=lambda x: (
        x.get('score', 0)
    ), reverse=True)
    logger.info(f"Sorted {len(candidates)} candidates.")
    return sorted_list

def filter_candidates(candidates, min_score=None, required_skills=None):
    filtered = candidates
    if min_score is not None:
        filtered = [c for c in filtered if c.get('score', 0) >= min_score]
        
    if required_skills and len(required_skills) > 0:
        req_skills_lower = set(s.lower().strip() for s in required_skills)
        def has_skills(c):
            c_skills = set(str(s).lower().strip() for s in c.get('resume_json', {}).get('skills', []))
            return len(req_skills_lower.intersection(c_skills)) > 0
        filtered = [c for c in filtered if has_skills(c)]
        
    logger.info(f"Filtered {len(candidates)} candidates down to {len(filtered)}")
    return filtered
