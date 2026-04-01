# RecruitIQ — AI-Powered Resume Screening & Ranking System

RecruitIQ is an AI-driven Applicant Tracking System (ATS) that intelligently screens, ranks, and shortlists candidates based on job requirements using advanced LLMs, semantic embeddings, and explainable scoring logic.  
It is designed to reduce recruiter effort, improve decision transparency, and support data-driven hiring.

---

## Key Features

### Intelligent Resume & Job Description Parsing
- Extracts structured information from PDF resumes using the `openai/gpt-oss-120b` LLM via the Groq API
- Robust JSON extraction with built-in API rate limit handling
- Automatically identifies:
  - Name and contact info
  - Extensive skill sets
  - Years of experience
  - Education level
  - Certifications
  - Project count
  - Salary expectations
  - Preferred qualifications (from job descriptions)

### PDF OCR Functionality
- Integrated robust OCR module to reliably extract text from job requirement PDF files and scanned resumes, ensuring no candidate data is missed

### Semantic Resume Matching
- Uses sentence embeddings to compute semantic similarity between:
  - Job description
  - Candidate resumes
- Captures meaning beyond keyword matching
- Ranks candidates by true relevance to the role

### Predictive Modeling & Explainable Scoring
Each candidate is scored and ranked using a trained **XGBoost model** combined with a transparent feature engineering pipeline:
- Semantic similarity score
- Skill match percentage
- Experience alignment
- Education level weighting

Recruiters can clearly see why a candidate ranks higher or lower based on explainable XGBoost match scores.

### Skill Gap Analysis
- Displays matched skills and missing required skills
- Helps recruiters identify immediate fit and upskilling potential

### Recruiter Workflow Support
- Interactive Streamlit dashboard with a decision dropdown per candidate:
  - Unreviewed
  - Shortlisted
  - Rejected
- No page reloads or lag
- Decisions persist during the session

### Export Shortlisted Candidates
- Download only shortlisted candidates as CSV
- Includes rank, scores, skills, education, certifications, and salary expectations

### Smart Filters
Recruiters can filter candidates by:
- Minimum final score
- Experience range
- Skill match percentage
- Education level

### Hiring Funnel Dashboard
At-a-glance metrics:
- Total candidates uploaded
- Shortlisted count
- Rejected count
- Unreviewed count

---

## Scoring Logic (Explainable AI)

Final candidate rank and match score is generated using an XGBoost machine learning model that relies on engineered features, including:

| Component        | Description                                         |
|------------------|-----------------------------------------------------|
| Semantic Match   | Meaning similarity between resume and job description |
| Skill Match      | Overlap between required and candidate skills       |
| Experience Score | Alignment with required experience                  |
| Education Score  | Degree-based weighting                              |

This ensures fair, transparent, and bias-aware ranking.

---

## Tech Stack

**Frontend & UI**
- Streamlit (Interactive Dashboard)

**AI / ML**
- Groq API (`openai/gpt-oss-120b` model for robust LLM parsing)
- XGBoost (Candidate ranking & match scoring)
- Sentence Transformers (Semantic embeddings)
- Cosine Similarity

**Data Processing & Document Parsing**
- Pandas (Custom feature engineering)
- PDF OCR (Text extraction for job descriptions and resumes)
- Custom JSON extraction & API rate limit handlers

---
