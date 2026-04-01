import os
import sys
import json
import re

# Add parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parser.llm_parser import llm, parse_resume
from langchain_core.prompts import PromptTemplate

dummy_resume = """
John Doe
Software Engineer
Skills: Python, Java, SQL
Experience: 5 years at TechCorp
Education: B.Sc. in Computer Science
"""

print("Testing direct parse_resume function...")
try:
    result = parse_resume(dummy_resume)
    print("Parsed Result:", result)
except Exception as e:
    print("Error in parse_resume:", e)
    
print("\nTesting raw LLM invoke...")
prompt_template = """
You are an expert HR parsed. Extract the following information from the provided resume text and return it as a valid JSON object ONLY, with no preamble or explanation.
Do not add formatting like ```json ... ```, just return the raw JSON string.

Required JSON keys:
- career_objective (str)
- skills (list of str)

Resume Text:
{text}
"""
try:
    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    chain = prompt | llm
    response = chain.invoke({"text": dummy_resume})
    print("RAW CONTENT:", repr(response.content))
    
    content = response.content.strip()
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        print("MATCH GROUP:", repr(match.group()))
        parsed = json.loads(match.group())
        print("JSON LOADS SUCCESS:", parsed)
    else:
        print("NO REGEX MATCH")
except Exception as e:
    print("Raw LLM Invoke Error:", e)
