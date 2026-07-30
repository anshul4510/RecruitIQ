import os
import json
from dotenv import load_dotenv
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini", 
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChatOpenAI: {e}")
        return None

def parse_resume(text):
    """
    Passes the raw resume text to the LLM and demands a JSON response with the requested fields.
    """
    active_llm = get_llm()
    if not active_llm:
        return {"error": "LLM was not initialized. Check OPENAI_API_KEY."}

    prompt_template = """
    You are an expert HR parser. Extract the following information from the provided resume text.
    Return ONLY a valid JSON object. Do not add formatting like ```json ... ```.

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
    chain = prompt | active_llm
    
    try:
        # Strictly limit the input text to 4000 chars to never exceed Groq TPM limits
        text = text[:4000]
        response = chain.invoke({"text": text})
        logger.info("LLM resume parsing request sent.")
        content = response.content.strip()
        parsed = json.loads(content)
        logger.info("Successfully parsed LLM resume JSON natively.")
        return parsed
    except Exception as e:
        logger.error(f"Error parsing resume with LLM: {e}")
        return {"error": str(e)}

def parse_job_description(text):
    """
    Passes the raw JD text to the LLM and demands a JSON response with the requested fields.
    """
    active_llm = get_llm()
    if not active_llm:
        return {"error": "LLM was not initialized. Check OPENAI_API_KEY."}

    prompt_template = """
    You are an expert Senior Technical Recruiter and ATS Specialist parsing a Job Description.
    Your goal is to extract and infer a comprehensive, candidate-centric list of concrete technical skills, languages, tools, frameworks, and databases that candidates, engineers, and students actually list on their resumes.

    CRITICAL RULES FOR CONCRETE SKILL KEYWORDS:
    1. BREAK DOWN HIGH-LEVEL TITLES & ABSTRACT CONCEPTS INTO SPECIFIC RESUME SKILLS:
       - NEVER return generic abstract domain labels or project titles like "Data Visualization", "Data Science", "UI/UX", "Functional Programming", "API Integrations", "Intelligent Traffic Routing", "Self-Healing Systems", "Automatic Anomaly Detection", "AI-powered Payment Operations", or "Multi-DC Architecture".
       - You MUST decompose every high-level term into the exact concrete libraries, packages, programming languages, databases, and developer tools required to implement it.
       - Mandatory Decomposition Examples:
         * If "Data Visualization" -> output "Python", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly".
         * If "Data Science" / "Machine Learning" -> output "Python", "Pandas", "NumPy", "Scikit-Learn", "SciPy", "SQL", "Statistics".
         * If "React" / "UI/UX" / "Frontend" -> output "JavaScript", "TypeScript", "HTML5", "CSS3", "React", "Redux", "REST APIs", "Git".
         * If "API Integrations" -> output "REST APIs", "JSON", "Axios", "Postman", "FastAPI", "Node.js".
         * If "Distributed Computing" / "Traffic Routing" / "Systems" -> output "Python", "C++", "Go", "Docker", "Kubernetes", "Kafka", "Redis", "Linux", "System Design".

    2. ALWAYS INCLUDE BASIC FOUNDATIONAL SKILLS:
       - Extract and include basic prerequisite skills written by students and junior-to-senior candidates (programming languages: Python, Java, JavaScript, C++, SQL; web basics: HTML5, CSS3, REST APIs; tools: Git, Linux; core concepts: Data Structures, Object-Oriented Programming).

    3. EXPAND AND INFER IMPLICIT TECH STACK REQUIREMENTS:
       - Generate a rich, broad list of 15 to 30 concrete, market-aligned technical skills (libraries, languages, frameworks, tools) expected for this role.

    Required JSON keys:
    - job_position_name (str): Concise clean position title (e.g. "Full Stack Software Engineer").
    - educational_requirements (str): Concise summary of degree requirements (e.g. "Bachelor's in CS / IT", "Master's Degree"). Keep it brief (under 6 words).
    - experience_requirement (str): Concise summary of years/level (e.g. "0-1 Years (Entry Level)", "2-4 Years"). Keep it brief (under 5 words).
    - salary_range (str): Extract compensation, pay range, CTC, or salary mentioned in the text (e.g. "$100,000 - $130,000 / year", "10 - 15 LPA", "Competitive / Not Specified").
    - age_requirement (str): Concise age requirement if mentioned, otherwise "Any".
    - explicit_skills (list of str): Concrete technical keywords, libraries, tools, and languages mentioned directly or implied by JD requirements.
    - inferred_skills (list of str): A broad list of 10-25+ concrete technical skills, languages, tools, frameworks, and libraries implied by modern industry tech stack standards.
    - skills_required (list of str): The COMBINED list of all explicit AND market-inferred concrete technical skills (deduplicated clean keywords).
    - responsibilities (list of str): Extract all job duties from the entire text.

    Job Description Text:
    {text}
    """
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    chain = prompt | active_llm
    
# Top-level Skill Decomposition Map & Abstract Concepts Filter
DECOMPOSITION_MAP = {
    "data visualization": ["Python", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly"],
    "data science": ["Python", "Pandas", "NumPy", "Scikit-Learn", "SciPy", "SQL"],
    "ui/ux": ["HTML5", "CSS3", "JavaScript", "TypeScript", "React", "Figma"],
    "react": ["JavaScript", "TypeScript", "HTML5", "CSS3", "React", "Redux", "REST APIs", "Git"],
    "functional programming": ["JavaScript", "TypeScript", "Python"],
    "api integrations": ["REST APIs", "JSON", "Axios", "Postman", "FastAPI"],
    "intelligent traffic routing": ["Nginx", "Docker", "Kubernetes", "Linux", "Networking"],
    "self-healing systems": ["Kubernetes", "Docker", "Prometheus", "Grafana", "Linux"],
    "automatic anomaly detection": ["Python", "Scikit-Learn", "PyTorch", "Pandas", "Statistics"],
    "ai-powered payment operations": ["Python", "REST APIs", "SQL", "PostgreSQL"],
    "multi-dc architecture": ["Docker", "Kubernetes", "AWS", "Redis", "Kafka", "Linux", "System Design"],
    "distributed computing": ["Python", "C++", "Go", "Docker", "Kubernetes", "Kafka", "Spark", "Redis"],
    "machine learning": ["Python", "Scikit-Learn", "TensorFlow", "PyTorch", "Pandas", "NumPy", "SQL"]
}

ABSTRACT_CONCEPTS_TO_SKIP = {
    "ui/ux", "functional programming", "api integrations", "data science", 
    "data visualization", "intelligent traffic routing", "self-healing systems",
    "automatic anomaly detection", "ai-powered payment operations", 
    "multi-dc architecture", "distributed computing", "machine learning",
    "deep learning", "cloud computing", "backend development", "frontend development"
}

def process_and_decompose_skills(skill_list):
    if not skill_list: return []
    cleaned = []
    for s in skill_list:
        if not isinstance(s, str): continue
        st = s.strip()
        if not st: continue
        
        s_lower = st.lower()
        
        # 1. Check if term matches decomposition map
        decomposed = False
        for key, concrete_list in DECOMPOSITION_MAP.items():
            if key in s_lower:
                for concrete in concrete_list:
                    if concrete not in cleaned:
                        cleaned.append(concrete)
                decomposed = True
                break
        
        # 2. If not decomposed and NOT an abstract concept title, keep if concise keyword (<=3 words)
        if not decomposed:
            if any(abs_concept in s_lower for abs_concept in ABSTRACT_CONCEPTS_TO_SKIP):
                continue
            if len(st.split()) <= 3 and st not in cleaned:
                cleaned.append(st)
    return cleaned

def parse_job_description(text):
    """
    Passes the raw JD text to the LLM and demands a JSON response with the requested fields.
    """
    active_llm = get_llm()
    if not active_llm:
        return {"error": "LLM was not initialized. Check OPENAI_API_KEY."}

    prompt_template = """
    You are an expert Senior Technical Recruiter and ATS Specialist parsing a Job Description.
    Your goal is to extract and infer a comprehensive, candidate-centric list of concrete technical skills, languages, tools, frameworks, and databases that candidates, engineers, and students actually list on their resumes.

    CRITICAL RULES FOR CONCRETE SKILLS:
    1. BREAK DOWN HIGH-LEVEL TITLES & ABSTRACT CONCEPTS INTO SPECIFIC RESUME SKILLS:
       - NEVER return generic abstract domain labels or project titles like "Data Visualization", "Data Science", "UI/UX", "Functional Programming", "API Integrations", "Intelligent Traffic Routing", "Self-Healing Systems", "Automatic Anomaly Detection", "AI-powered Payment Operations", or "Multi-DC Architecture".
       - You MUST decompose every high-level term into the exact concrete libraries, packages, programming languages, databases, and developer tools required to implement it.
       - Mandatory Decomposition Examples:
         * If "Data Visualization" -> output "Python", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly".
         * If "Data Science" / "Machine Learning" -> output "Python", "Pandas", "NumPy", "Scikit-Learn", "SciPy", "SQL", "Statistics".
         * If "React" / "UI/UX" / "Frontend" -> output "JavaScript", "TypeScript", "HTML5", "CSS3", "React", "Redux", "REST APIs", "Git".
         * If "API Integrations" -> output "REST APIs", "JSON", "Axios", "Postman", "FastAPI", "Node.js".
         * If "Distributed Computing" / "Traffic Routing" -> output "Python", "C++", "Go", "Docker", "Kubernetes", "Kafka", "Redis", "Linux", "System Design".

    2. ALWAYS INCLUDE BASIC FOUNDATIONAL SKILLS:
       - Extract and include basic prerequisite skills written by students and junior-to-senior candidates (programming languages: Python, Java, JavaScript, C++, SQL; web basics: HTML5, CSS3, REST APIs; tools: Git, Linux; core concepts: Data Structures, Object-Oriented Programming).

    3. EXPAND AND INFER IMPLICIT TECH STACK REQUIREMENTS:
       - Generate a rich, broad list of 15 to 30 concrete, market-aligned technical skills (libraries, languages, frameworks, tools) expected for this role.

    Required JSON keys:
    - job_position_name (str)
    - educational_requirements (str): Extract from anywhere in the text.
    - experience_requirement (str): Extract from anywhere in the text.
    - age_requirement (str)
    - explicit_skills (list of str): Concrete technical keywords, libraries, tools, and languages mentioned directly or implied by JD requirements.
    - inferred_skills (list of str): A broad list of 10-25+ concrete technical skills, languages, tools, frameworks, and libraries implied by modern industry tech stack standards.
    - skills_required (list of str): The COMBINED list of all explicit AND market-inferred concrete technical skills (deduplicated clean keywords).
    - responsibilities (list of str): Extract all job duties from the entire text.

    Job Description Text:
    {text}
    """
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    chain = prompt | active_llm

    try:
        text = text[:4000]
        response = chain.invoke({"text": text})
        logger.info("LLM JD parsing request sent.")
        content = response.content.strip()
        parsed = json.loads(content)

        explicit = process_and_decompose_skills(parsed.get("explicit_skills", []))
        inferred = process_and_decompose_skills(parsed.get("inferred_skills", []))
        skills_req = process_and_decompose_skills(parsed.get("skills_required", []))
        
        combined = list(dict.fromkeys(explicit + inferred + skills_req))
        parsed["skills_required"] = combined
        parsed["explicit_skills"] = explicit if explicit else combined
        parsed["inferred_skills"] = inferred
        
        logger.info(f"Successfully parsed LLM JD JSON. Total concrete skills: {len(combined)}.")
        return parsed
    except Exception as e:
        logger.error(f"Error parsing job description with LLM: {e}")
        return {"error": str(e)}
