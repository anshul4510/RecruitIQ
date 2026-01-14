def process_jd_skills(jd_skills: str):
    return [s.strip().lower() for s in jd_skills.split(",") if s.strip()]


def build_jd_payload(jd_text: str, jd_skills: str):
    return {
        "text": jd_text.strip(),
        "skills": process_jd_skills(jd_skills)
    }