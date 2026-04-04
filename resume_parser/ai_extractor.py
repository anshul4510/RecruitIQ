import json
import logging
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


logger = logging.getLogger(__name__)


from utils.education_processor import EducationProcessor

class ResumeAIExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an AI resume parser.
                Extract structured information from the resume text.

                Return ONLY a JSON object in the exact following format, filling in the attributes based on the text.
                DO NOT return any markdown formatting or explanations.
                {{
                    "name": "",
                    "email": "",
                    "phone": "",
                    "address": "",
                    "career_objective": "",
                    "skills": [],
                    "educational_institution_name": [],
                    "degree_names": [],
                    "passing_years": [],
                    "educational_results": [],
                    "result_types": [],
                    "major_field_of_studies": [],
                    "professional_company_names": [],
                    "company_urls": [],
                    "start_dates": [],
                    "end_dates": [],
                    "related_skills_in_job": [],
                    "positions": [],
                    "locations": [],
                    "responsibilities": [],
                    "extra_curricular_activity_types": [],
                    "extra_curricular_organization_names": [],
                    "extra_curricular_organization_links": [],
                    "role_positions": [],
                    "languages": [],
                    "proficiency_levels": [],
                    "certification_providers": [],
                    "certification_skills": [],
                    "online_links": [],
                    "issue_dates": [],
                    "expiry_dates": [],
                    "experience_years": 0,
                    "projects_count": 0
                }}

                Rules:
                - Extract ALL skills extensively: including required, preferred, additional, tools, or any kind of skills mentioned anywhere in the resume. 
                - Fields expecting lists should contain a list of strings. If no data is found, return an empty list or string accordingly.
                - experience_years must be a number (estimate conservatively if unclear).
                - projects_count must be a number.
                """
            ),
            ("human", "{resume_text}")
        ])

    def extract(self, resume_text: str) -> dict:
        logger.info(f"LLM extraction request sent for resume text.")
        response = self.llm.invoke(
            self.prompt.format_messages(
                resume_text=resume_text[:12000]
            )
        )

        try:
            content = response.content.strip()
            parsed_json = json.loads(content)
            
            # --- Hybrid Extraction Enhancement ---
            # Use deterministic processor to verify/augment education details
            edu_details = EducationProcessor.extract_education_simple(resume_text)
            if edu_details:
                # If LLM missed degrees or institutions, fill them from regex extractor
                if not parsed_json.get('degree_names') or len(parsed_json['degree_names']) == 0:
                    parsed_json['degree_names'] = [e['raw_degree'] for e in edu_details]
                if not parsed_json.get('educational_institution_name') or len(parsed_json['educational_institution_name']) == 0:
                    parsed_json['educational_institution_name'] = list(set([e['institution'] for e in edu_details if e['institution'] != "Unknown Institution"]))
                if not parsed_json.get('major_field_of_studies') or len(parsed_json['major_field_of_studies']) == 0:
                    parsed_json['major_field_of_studies'] = list(set([e['major'] for e in edu_details]))

            logger.info("Successfully parsed LLM response JSON and augmented with Regex Processor.")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Fallback to pure regex extraction if LLM fails entirely
            edu_details = EducationProcessor.extract_education_simple(resume_text)
            return {
                "name": "Unknown",
                "email": "",
                "phone": "",
                "address": "",
                "degree_names": [e['raw_degree'] for e in edu_details] if edu_details else [],
                "educational_institution_name": [e['institution'] for e in edu_details] if edu_details else [],
                "major_field_of_studies": [e['major'] for e in edu_details] if edu_details else [],
                "skills": [],
                "experience_years": 0,
                "projects_count": 0
            }
