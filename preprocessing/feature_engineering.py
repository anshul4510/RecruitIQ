import pandas as pd
from preprocessing.clean_text import clean_skills, clean_column


EDUCATION_MAPPING = {
    "phd": 4,
    "m.tech": 3,
    "mba": 3,
    "m.sc": 3,
    "b.tech": 2,
    "b.sc": 2
}


def encode_education(education: str) -> int:
    if not isinstance(education, str):
        return 1

    education = education.lower()
    for key, value in EDUCATION_MAPPING.items():
        if key in education:
            return value
    return 1


def preprocess_resumes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_skills"] = df["Skills"].apply(clean_skills)
    df["clean_education"] = clean_column(df, "Education")
    if "Job Role" in df.columns:
        df["clean_job_role"] = clean_column(df, "Job Role")
    else:
        df["clean_job_role"] = ""
    df["experience_years"] = df["Experience (Years)"]
    df["projects_count"] = df.get("Projects Count", 0)
    df["education_level"] = df["clean_education"].apply(encode_education)
    df["skills_text"] = df["clean_skills"].apply(lambda x: " ".join(x))

    df["resume_text"] = (
        df["skills_text"] + " " +
        df["clean_education"] + " " +
        df["clean_job_role"]
    )

    return df
