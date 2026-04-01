import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

try:
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-120b", 
        temperature=0.2
    )
except Exception as e:
    print(f"Warning: Failed to initialize ChatGroq explainer: {e}")
    llm = None

def generate_explanation(resume_json, jd_json):
    """
    Returns an explanation of why the candidate is a good match, 
    missing skills, and strength summary.
    """
    if not llm:
        return {
            "match_reason": "LLM not configured. Please add GROQ_API_KEY to .env.",
            "missing_skills": [],
            "strength_summary": "LLM not configured."
        }

    prompt_template = """
    You are an expert HR recruiter. Compare the candidate's resume with the job description.
    Provide a JSON response with the following keys:
    - match_reason (str): A brief explanation of why this candidate is a good fit.
    - missing_skills (list of str): Skills required by the JD that the candidate lacks.
    - strength_summary (str): A 1-2 sentence summary of the candidate's core strengths.
    
    Return ONLY valid JSON with no extra formatting.
    
    Candidate Resume:
    {resume}
    
    Job Description:
    {jd}
    """
    prompt = PromptTemplate(input_variables=["resume", "jd"], template=prompt_template)
    chain = prompt | llm
    
    try:
        response = chain.invoke({"resume": json.dumps(resume_json), "jd": json.dumps(jd_json)})
        content = response.content.strip()
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(content)
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return {
            "match_reason": "Failed to generate explanation.",
            "missing_skills": [],
            "strength_summary": "Failed to generate explanation."
        }

def analyze_resume_quality(resume_json):
    """
    Scores the overall quality of the resume format and content out of 10.
    """
    if not llm:
        return {"quality_score": 0, "feedback": "LLM not configured."}
        
    prompt_template = """
    You are an expert recruiter. Analyze the overall quality of this parsed resume based on completeness, impact, and structure.
    Return a JSON object with:
    - quality_score (int): 1 to 10
    - feedback (str): 1-2 sentences on how to improve it
    
    Return ONLY valid JSON.
    
    Candidate Resume:
    {resume}
    """
    prompt = PromptTemplate(input_variables=["resume"], template=prompt_template)
    chain = prompt | llm
    
    try:
        response = chain.invoke({"resume": json.dumps(resume_json)})
        content = response.content.strip()
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(content)
    except:
        return {"quality_score": 0, "feedback": "Failed to analyze."}
