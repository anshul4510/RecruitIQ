import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

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
    Returns an expert explanation of a top candidate's match quality.
    """
    if not llm:
        return {
            "ats_score": 0.5,
            "strengths": ["LLM unconfigured"],
            "weaknesses": ["LLM unconfigured"],
            "recommendation": "Consider",
            "reasoning": "LLM not configured. Please add OPENAI_API_KEY to .env."
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
    chain = prompt | llm
    
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
