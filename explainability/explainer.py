import os
import json
from dotenv import load_dotenv
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

try:
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini", 
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
except Exception as e:
    print(f"Warning: Failed to initialize ChatOpenAI explainer: {e}")
    llm = None

def analyze_and_explain(resume_json, jd_json):
    """
    Returns an explanation of why the candidate is a good match, 
    missing skills, strength summary, quality score, and feedback in a single LLM call.
    """
    if not llm:
        return {
            "match_reason": "LLM not configured. Please add OPENAI_API_KEY to .env.",
            "missing_skills": [],
            "strength_summary": "LLM not configured.",
            "quality_score": 0,
            "feedback": "LLM not configured."
        }

    prompt_template = """
    You are an expert HR recruiter. Compare the candidate's resume with the job description.
    Also analyze the overall quality of this parsed resume based on completeness, impact, and structure.
    
    Provide ONLY a JSON object with the exact following keys:
    {{
        "match_reason": "A brief explanation of why this candidate is a good fit.",
        "missing_skills": ["List", "of", "missing", "skills required by JD"],
        "strength_summary": "A 1-2 sentence summary of the candidate's core strengths.",
        "quality_score": 8, # Integer from 1 to 10
        "feedback": "1-2 sentences on how to improve the resume format/content."
    }}
    
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
        return json.loads(content)
    except Exception as e:
        print(f"Error generating explanation and quality: {e}")
        return {
            "match_reason": "Failed to generate explanation.",
            "missing_skills": [],
            "strength_summary": "Failed to generate explanation.",
            "quality_score": 0,
            "feedback": "Failed to analyze."
        }
