import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

def get_llm(temperature=0.2, json_mode=True):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini", 
            temperature=temperature,
            model_kwargs=kwargs
        )
    except Exception as e:
        print(f"Warning: Failed to initialize ChatOpenAI explainer: {e}")
        return None

def analyze_and_explain(resume_json, jd_json):
    """
    Returns an expert explanation of a top candidate's match quality.
    """
    active_llm = get_llm(temperature=0.2, json_mode=True)
    if not active_llm:
        return {
            "ats_score": 0.5,
            "strengths": ["LLM unconfigured"],
            "weaknesses": ["LLM unconfigured"],
            "recommendation": "Consider",
            "reasoning": "LLM not configured. Please set OPENAI_API_KEY."
        }

    prompt_template = """
    You are an expert technical recruiter evaluating a candidate for a specific role.
    Your evaluation must prioritize:
    - Relevance of experience over total years
    - Demonstrated impact over listed responsibilities
    - Career progression and growth trajectory
    - Contextual skill use over keyword presence

    Be concise, specific, and evidence-based. Do not reward verbosity.
    Penalize vague claims without measurable outcomes.
    
    Provide ONLY a JSON object with the exact following schema:
    {{
      "ats_score": 0.85, 
      "strengths": ["string", "string"], 
      "weaknesses": ["string"], 
      "recommendation": "Strong Hire | Consider | Reject", 
      "reasoning": "2-3 sentence evidence-based justification."
    }}
    
    Candidate Resume:
    {resume}
    
    Job Description:
    {jd}
    """
    prompt = PromptTemplate(input_variables=["resume", "jd"], template=prompt_template)
    chain = prompt | active_llm
    
    try:
        response = chain.invoke({"resume": json.dumps(resume_json), "jd": json.dumps(jd_json)})
        content = response.content.strip()
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return {
            "ats_score": 0.5,
            "strengths": ["Error in generation"],
            "weaknesses": [],
            "recommendation": "Consider",
            "reasoning": "Failed to analyze."
        }

def generate_outreach_email(resume_json, jd_json, explanation=None):
    """
    Generates a personalized outreach email for the candidate based on their resume and the JD.
    """
    email_llm = get_llm(temperature=0.7, json_mode=False)
    if not email_llm:
        return "LLM not configured. Please set OPENAI_API_KEY."

    try:
        prompt_template = """
        You are a technical recruiter reaching out to a candidate who looks like a great fit for your open role.
        
        Write a professional, personalized outreach email to this candidate.
        
        Guidelines:
        1. Keep it under 150 words. Be concise and respectful of their time.
        2. Specifically mention 1 or 2 impressive things from THEIR resume (a specific company they worked at, an impressive project, or a rare skill).
        3. Explain briefly why that specific background makes them a great fit for YOUR Job Description.
        4. End with a low-friction Call to Action (CTA) asking for a brief chat.
        5. Do not invent details. Use only the provided information.
        6. Return ONLY the email text. Do not include placeholders like "[Your Name]". Use generic sign-offs instead.
        
        Candidate Name: {name}
        
        Candidate Resume Highlight:
        {resume}
        
        Job Description Highlight:
        {jd}
        
        Explanation context (optional context on why they matched):
        {explanation_text}
        """
        
        prompt = PromptTemplate(input_variables=["name", "resume", "jd", "explanation_text"], template=prompt_template)
        chain = prompt | email_llm
        
        # We only need the summary of resume/jd to save tokens and keep it focused
        res_summary = {
            "skills": resume_json.get("skills", [])[:10],
            "experience": resume_json.get("positions", [])[:2],
            "companies": resume_json.get("professional_company_names", [])[:2]
        }
        
        jd_summary = {
            "title": jd_json.get("job_position_name", "Open Role"),
            "core_skills": jd_json.get("skills_required", [])[:5]
        }
        
        exp_text = explanation.get("reasoning", "") if explanation else ""
        c_name = resume_json.get("name", "Candidate")
        if c_name.lower() == "unknown":
            c_name = "Candidate"
        
        response = chain.invoke({
            "name": c_name,
            "resume": json.dumps(res_summary), 
            "jd": json.dumps(jd_summary),
            "explanation_text": exp_text
        })
        
        return response.content.strip()
    except Exception as e:
        print(f"Error generating email: {e}")
        return "Failed to generate outreach email. Please try again later."

