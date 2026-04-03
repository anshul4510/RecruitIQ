def rank_candidates(candidates):
    """
    candidates is a list of dictionaries, where each dict has at least 'score'.
    Sorts descending by score.
    """
    return sorted(candidates, key=lambda x: (
        x.get('score', 0),
        x.get('quality', {}).get('quality_score', 0) if isinstance(x.get('quality', {}), dict) else 0,
        x.get('match_breakdown', {}).get('skill_overlap_score', 0) if isinstance(x.get('match_breakdown', {}), dict) else 0
    ), reverse=True)

def filter_candidates(candidates, min_score=None, required_skills=None):
    """
    Filters candidates based on score or required skills.
    `candidates` contains parsed JSON info and calculated scores.
    """
    filtered = candidates
    if min_score is not None:
        filtered = [c for c in filtered if c.get('score', 0) >= min_score]
        
    if required_skills and len(required_skills) > 0:
        req_skills_lower = set(s.lower().strip() for s in required_skills)
        def has_skills(c):
            c_skills = set(str(s).lower().strip() for s in c.get('resume_json', {}).get('skills', []))
            # Just require at least one overlap for relaxed filtering or strict subset for strict filtering
            # Here we do relaxed filtering: must have at least one of the required skills
            return len(req_skills_lower.intersection(c_skills)) > 0
        filtered = [c for c in filtered if has_skills(c)]
        
    return filtered
