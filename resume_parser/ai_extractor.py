import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


class ResumeAIExtractor:
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an AI resume parser.
                Extract structured information from the resume text.

                Return ONLY valid JSON in the exact following format, filling in the attributes based on the text:
                {{
                    "name": "",
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
                    "related_skils_in_job": [],
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
                - do NOT add explanations.
                - do NOT return text outside JSON.
                """
            ),
            ("human", "{resume_text}")
        ])

    def extract(self, resume_text: str) -> dict:
        response = self.llm.invoke(
            self.prompt.format_messages(
                resume_text=resume_text[:12000]
            )
        )

        try:
            import re
            content = response.content.strip()
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "name": "Unknown",
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
                "related_skils_in_job": [],
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
            }
