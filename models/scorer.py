from config.config import MODEL_CONFIG

class ResumeScorer:
    def __init__(self, required_experience: int, jd_skills: list[str]):
        self.required_experience = max(required_experience, 1)
        self.jd_skills = set(jd_skills)

        self.weights = MODEL_CONFIG["weights"]
        self.max_edu = MODEL_CONFIG["education_max_level"]

    def skill_match_score(self, resume_skills: list[str]) -> float:
        resume_skills = set(resume_skills)
        if not self.jd_skills:
            return 0.0
        return len(self.jd_skills & resume_skills) / len(self.jd_skills)

    def experience_score(self, resume_exp: float) -> float:
        return min(resume_exp / self.required_experience, 1.0)

    def education_score(self, education_level: int) -> float:
        return education_level / self.max_edu

    def final_score(
        self,
        semantic_similarity: float,
        resume_skills: list[str],
        resume_exp: float,
        education_level: int
    ) -> dict:

        skill = self.skill_match_score(resume_skills)
        exp = self.experience_score(resume_exp)
        edu = self.education_score(education_level)

        final = (
            self.weights["semantic_similarity"] * semantic_similarity +
            self.weights["skill_match"] * skill +
            self.weights["experience_match"] * exp +
            self.weights["education_match"] * edu
        )

        return {
            "final_score": round(final * 100, 2),
            "semantic_score": round(semantic_similarity * 100, 2),
            "skill_score": round(skill * 100, 2),
            "experience_score": round(exp * 100, 2),
            "education_score": round(edu * 100, 2)
        }
