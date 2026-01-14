# RecruitIQ — AI-Powered Resume Screening & Ranking System

RecruitIQ is an AI-driven Applicant Tracking System (ATS) that intelligently screens, ranks, and shortlists candidates based on job requirements using LLMs, semantic embeddings, and explainable scoring logic.  
It is designed to reduce recruiter effort, improve decision transparency, and support data-driven hiring.

---

## Key Features

### Intelligent Resume Parsing
- Extracts structured information from PDF resumes using an LLM (Groq)
- Automatically identifies:
  - Name
  - Skills
  - Years of experience
  - Education level
  - Certifications
  - Project count
  - Salary expectations

### Semantic Resume Matching
- Uses sentence embeddings to compute semantic similarity between:
  - Job description
  - Candidate resumes
- Captures meaning beyond keyword matching
- Ranks candidates by true relevance to the role

### Explainable Candidate Scoring
Each candidate is scored using a transparent, weighted system:
- Semantic similarity score
- Skill match percentage
- Experience alignment
- Education level weighting

Recruiters can clearly see why a candidate ranks higher or lower.

### Skill Gap Analysis
- Displays matched skills and missing required skills
- Helps recruiters identify immediate fit and upskilling potential

### Recruiter Workflow Support
- Decision dropdown per candidate:
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

Final candidate score is computed using a weighted combination of:

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
- Streamlit

**AI / ML**
- Groq LLM (Resume Parsing)
- Sentence Transformers (Embeddings)
- Cosine Similarity

**Data Processing**
- Pandas
- Custom feature engineering pipeline

---


