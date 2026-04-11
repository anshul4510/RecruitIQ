import logging

logger = logging.getLogger(__name__)

def apply_hard_filters(resume_json: dict, jd_json: dict, config: dict = None) -> dict:
    """
    Applies strict gating criteria to immediately filter out clearly unqualified candidates 
    before expensive NLP processing and model scoring is done.
    """
    if config is None:
        config = {
            "min_core_skill_coverage": 0.30, # Allow some leeway, e.g., have at least 30% of core skills
            "allow_project_offset": True
        }

    reasons = []

    # 1. Experience Check
    exp_required = jd_json.get("experiencere_requirement", "")
    req_years = _extract_years(exp_required)
    
    resume_years = resume_json.get("experience_years", 0)
    if resume_years < req_years:
        if config.get("allow_project_offset") and resume_json.get("projects_count", 0) >= 3:
            pass # Waive experience if they have sufficient project volume
        else:
            reasons.append(f"Insufficient experience: {resume_years} yrs (Required: {req_years} yrs)")

    # 2. Skill Minimum Coverage
    jd_skills = set(str(s).lower().strip() for s in jd_json.get("skills_required", []))
    res_skills = set(str(s).lower().strip() for s in resume_json.get("skills", []))
    
    if len(jd_skills) > 0:
        coverage = len(jd_skills & res_skills) / len(jd_skills)
        if coverage < config.get("min_core_skill_coverage", 0.0):
            reasons.append(f"Below minimum skill coverage ({coverage*100:.0f}% vs req {config.get('min_core_skill_coverage')*100:.0f}%)")

    # 3. Education Strict Constraint
    # Assuming education_level mapping: None=0, HS=1, BS=2, MS=3, PhD=4
    # Optional logic: skip for now as ATS mostly filters on exp/skills strictly.

    if reasons:
        logger.info(f"Resume Rejected by Hard Filters: {reasons}")
        return {"status": "REJECTED", "reasons": reasons}
    
    return {"status": "PASSED"}

def _extract_years(text):
    if not text:
        return 0.0
    import re
    nums = re.findall(r'\d+', str(text))
    if nums:
        return float(nums[0])
    return 0.0
