import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

logger = logging.getLogger(__name__)

def evaluate_candidate_fit(resume_json: dict, jd_json: dict, features: dict) -> dict:
    """
    Expert system evaluator providing a qualitative review of the candidate.
    Takes extracted JSON data plus the calculated feature metrics to save tokens and manual counting.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found. Skipping LLM Evaluation.")
        return _fallback_evaluation()

    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert Senior Technical Recruiter and ATS Evaluator.
            Your goal is to evaluate candidate fit, prioritizing contextual skill usage, measurable impact, and career progression over simple keyword matching.
            
            Evaluate the candidate based on these strict guidelines:
            1. Impact over Length: Reward the candidate if they used numbers/metrics to quantify their results in 'responsibilities'.
            2. Contextual Skills: Note if they used a required skill actively in a recent project/job, versus just listing it in a generic "Skills" array.
            3. Career Growth: Note if they show consistent progression.
            
            Return ONLY a JSON response in the following strict schema. DO NOT include markdown formatting outside the JSON:
            {{
              "final_ats_reasoning_score": 85,
              "hiring_recommendation": "<Strong Hire | Consider | Reject>",
              "strengths": ["string", "string"],
              "weaknesses": ["string"],
              "recruiter_summary": "A strict 2-3 sentence summary justifying the recommendation focusing on context and impact."
            }}
            """
        ),
        (
            "human",
            """
            Job Description Required Context: {jd_json}
            
            Candidate Extraction Data: {resume_json}
            
            Pre-computed ATS Features for context: {features}
            """
        )
    ])

    try:
        # We strip long lists to save context window and avoid overwhelm
        clean_resume = {
            "name": resume_json.get("name"),
            "skills": resume_json.get("skills", [])[:20], # top 20 skills
            "experience_years": resume_json.get("experience_years"),
            "positions": resume_json.get("positions", [])[:3],
            "responsibilities": resume_json.get("responsibilities", [])[:10], # Top bullets
        }
        
        clean_jd = {
            "job_position_name": jd_json.get("job_position_name"),
            "skills_required": jd_json.get("skills_required", []),
            "experiencere_requirement": jd_json.get("experiencere_requirement")
        }

        logger.info("Sending expert Evaluation request to LLM...")
        response = llm.invoke(
            prompt.format_messages(
                jd_json=json.dumps(clean_jd),
                resume_json=json.dumps(clean_resume),
                features=json.dumps(features)
            )
        )
        
        content = response.content.strip()
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        logger.error(f"LLM Evaluation failed: {e}")
        return _fallback_evaluation()


def _fallback_evaluation():
    return {
        "final_ats_reasoning_score": 50,
        "hiring_recommendation": "Consider",
        "strengths": ["System could not reach LLM for dynamic evaluation."],
        "weaknesses": ["Incomplete evaluation payload."],
        "recruiter_summary": "Automated evaluation skipped due to API limitations or timeout. Rely on basic numerical scoring."
    }
