import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize the Groq LLM
try:
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-120b", 
        temperature=0
    )
except Exception as e:
    print(f"Warning: Failed to initialize ChatGroq: {e}. Please ensure GROQ_API_KEY is in your .env file.")
    llm = None

def parse_resume(text):
    """
    Passes the raw resume text to the LLM and demands a JSON response with the requested fields.
    """
    if not llm:
        return {"error": "LLM was not initialized. Check GROQ_API_KEY in .env"}

    prompt_template = """
    You are an expert HR parsed. Extract the following information from the provided resume text and return it as a valid JSON object ONLY, with no preamble or explanation.
    Do not add formatting like ```json ... ```, just return the raw JSON string.

    Required JSON keys:
    - career_objective (str)
    - skills (list of str)
    - educational_institution_name (list of str)
    - degree_names (list of str)
    - passing_years (list of str)
    - major_field_of_studies (list of str)
    - educational_results (list of str)
    - professional_company_names (list of str)
    - positions (list of str)
    - responsibilities (list of str)
    - certification_providers (list of str)
    - certification_skills (list of str)
    - issue_dates (list of str)
    - expiry_dates (list of str)
    - languages (list of str)
    - proficiency_levels (list of str)
    - extra_curricular_activity_types (list of str)
    - extra_curricular_organization_names (list of str)
    - role_positions (list of str)

    Resume Text:
    {text}
    """
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    chain = prompt | llm
    
    try:
        # Strictly limit the input text to 4000 chars to never exceed Groq TPM limits
        text = text[:4000]
        response = chain.invoke({"text": text})
        content = response.content.strip()
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing resume with LLM: {e}")
        return {"error": str(e)}

def parse_job_description(text):
    """
    Passes the raw JD text to the LLM and demands a JSON response with the requested fields.
    """
    if not llm:
        return {"error": "LLM was not initialized. Check GROQ_API_KEY in .env"}

    prompt_template = """
    You are an expert HR parser. Extract the following information from the provided job description text and return it as a valid JSON object ONLY, with no preamble or explanation.
    Do not add formatting like ```json ... ```, just return the raw JSON string.

    CRITICAL INSTRUCTION: The text may be poorly formatted due to OCR scanning. You MUST read and analyze the ENTIRE document thoroughly.
    Pay special attention to BOTH "Required Qualifications" AND "Preferred Qualifications" / "Additional skills". You MUST extract and combine skills/requirements from ALL sections.

    Required JSON keys:
    - job_position_name (str)
    - educational_requirements (str): Extract from anywhere in the text (combine both required and preferred).
    - experience_requirement (str): Extract from anywhere in the text (combine both required and preferred).
    - age_requirement (str)
    - skills_required (list of str): Extract ALL skills, tools, technologies, and languages mentioned ANYWHERE in the entire text. *CRITICALLY IMPORTANT*: You MUST include everything listed under both "Required Qualifications" and "Preferred Qualifications". Output clean keywords only (e.g. "Python", "Machine Learning", "AWS", "Agile").
    - responsibilities (list of str): Extract all job duties from the entire text.

    Job Description Text:
    {text}
    """
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    chain = prompt | llm
    
    try:
        text = text[:4000]
        response = chain.invoke({"text": text})
        content = response.content.strip()
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing job description with LLM: {e}")
        return {"error": str(e)}
