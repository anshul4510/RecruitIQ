import pandas as pd
from resume_parser.pdf_loader import extract_text_from_multiple_pdfs
from resume_parser.ai_extractor import ResumeAIExtractor
from preprocessing.feature_engineering import preprocess_resumes


def process_uploaded_resumes(uploaded_files, api_key):
    extracted = extract_text_from_multiple_pdfs(uploaded_files)
    extractor = ResumeAIExtractor(api_key=api_key)

    structured_resumes = []

    for item in extracted:
        parsed = extractor.extract(item["text"])
        structured_resumes.append({
            "Name": parsed.get("name", "Unknown"),
            "Skills": ", ".join(parsed.get("skills", [])),
            "Experience (Years)": parsed.get("experience_years", 0),
            "Education": parsed.get("education", ""),
            "Projects Count": parsed.get("projects_count", 0),
            "Certifications": ", ".join(parsed.get("certifications", [])),
            "Salary Expectation (₹)": parsed.get("salary_expectation", 0),
            "Source File": item["filename"]
        })

    df = pd.DataFrame(structured_resumes)
    return preprocess_resumes(df)
