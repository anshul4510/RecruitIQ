import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


class ResumeAIExtractor:
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b"):
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
                Extract structured information from resume text.

                Return ONLY valid JSON in the following format:
                {{
                "name": "",
                "skills": [],
                "experience_years": number,
                "education": "",
                "projects_count": number,
                "certifications": [],
                "salary_expectation": number
                }}

                Rules:
                - skills must be lowercase skill names
                - certifications must be a list of certification names
                - experience_years must be a number (estimate conservatively if unclear)
                - projects_count must be a number (estimate if unclear)
                - education must be highest degree only (B.Sc, B.Tech, MBA, M.Tech, PhD)
                - salary_expectation must be a NUMBER only (annual), 0 if not mentioned
                - do NOT add explanations
                - do NOT return text outside JSON
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
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "name": "Unknown",
                "skills": [],
                "experience_years": 0,
                "education": "",
                "projects_count": 0,
                "certifications": [],
                "salary_expectation": 0
            }
