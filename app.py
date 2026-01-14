import streamlit as st
import pandas as pd
import os
from services.resume_service import process_uploaded_resumes
from services.jd_service import build_jd_payload
from models.embeddings import ResumeEmbedder
from models.similarity import ResumeSimilarityEngine
from models.scorer import ResumeScorer
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="RecruitIQ", layout="wide")
st.title("RecruitIQ : AI-Powered Resume Screening System")
st.sidebar.header("Job Requirements")

jd_text = st.sidebar.text_area(
    "Job Description",
    value="Looking for a Data Scientist with Python, SQL, Machine Learning"
)
jd_skills = st.sidebar.text_input(
    "Required Skills (comma separated)",
    value="python, sql, machine learning"
)

required_experience = st.sidebar.slider("Required Experience (Years)", 0, 20, 3)
top_k = st.sidebar.slider("Top Candidates", 1, 50, 10)

st.sidebar.markdown("---")
st.sidebar.header("📤 Upload Resumes")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("⬅ Upload resumes to start screening.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("☰ Filters")
min_score = st.sidebar.slider("Minimum Final Score", 0, 100, 0)
min_exp, max_exp = st.sidebar.slider("Experience Range", 0, 20, (0, 20))
min_skill_match = st.sidebar.slider("Minimum Skill Match (%)", 0, 100, 0)

jd_payload = build_jd_payload(jd_text, jd_skills)

if "processed_df" not in st.session_state:
    st.session_state.processed_df = process_uploaded_resumes(
        uploaded_files=uploaded_files,
        api_key=os.getenv("GROQ_API_KEY")
    )

df = st.session_state.processed_df

@st.cache_resource
def load_embedder():
    return ResumeEmbedder()
embedder = load_embedder()

if "resume_embeddings" not in st.session_state:
    st.session_state.resume_embeddings = embedder.embed_texts(df["resume_text"].tolist())

resume_embeddings = st.session_state.resume_embeddings

engine = ResumeSimilarityEngine(resume_embeddings)
jd_embedding = embedder.embed_texts([jd_payload["text"]])

if "results_df" not in st.session_state:
    scores, indices = engine.search(jd_embedding, top_k=min(top_k, len(df)))
    scorer = ResumeScorer(
        required_experience=required_experience,
        jd_skills=jd_payload["skills"]
    )

    ranked_rows = []
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        row = df.iloc[idx]
        breakdown = scorer.final_score(
            semantic_similarity=score,
            resume_skills=row["clean_skills"],
            resume_exp=row["experience_years"],
            education_level=row["education_level"]
        )
        ranked_rows.append({
            "Rank": rank,
            "Name": row["Name"],
            "Final Score": breakdown["final_score"],
            "Skill Match (%)": breakdown["skill_score"],
            "Semantic Match (%)": breakdown["semantic_score"],
            "Education Score (%)": breakdown["education_score"],
            "Experience (Years)": row["experience_years"],
            "Education": row["Education"],
            "Projects Count": row["projects_count"],
            "Skills": row["Skills"],
            "Certifications": row["Certifications"],
            "Salary Expectation (₹)": row["Salary Expectation (₹)"],
            "Source File": row["Source File"]
        })

    st.session_state.results_df = pd.DataFrame(ranked_rows)

results_df = st.session_state.results_df

education_options = sorted(results_df["Education"].dropna().unique())
education_filter = st.sidebar.multiselect(
    "Education Level", education_options, education_options
)

filtered_df = results_df[
    (results_df["Final Score"] >= min_score) &
    (results_df["Experience (Years)"].between(min_exp, max_exp)) &
    (results_df["Skill Match (%)"] >= min_skill_match) &
    (results_df["Education"].isin(education_filter))
]

statuses = []
for _, r in filtered_df.iterrows():
    status = st.session_state.get(f"decision_{r['Source File']}", "Unreviewed")
    statuses.append(status)
filtered_df["Status"] = statuses

col1, col2, col3, col4 = st.columns(4)
col1.metric("Uploaded", len(filtered_df))
col2.metric("Shortlisted", sum(filtered_df["Status"] == "Shortlisted"))
col3.metric("Rejected", sum(filtered_df["Status"] == "Rejected"))
col4.metric("Unreviewed", sum(filtered_df["Status"] == "Unreviewed"))

st.subheader("👤 Ranked Candidates")
st.dataframe(
    filtered_df[
        ["Rank", "Name", "Status", "Final Score", "Experience (Years)", "Education", "Skill Match (%)"]
    ],
    use_container_width=True,
    hide_index=True
)

st.markdown("### ℹ️ Candidate Details")
jd_skill_set = set(jd_payload["skills"])

for _, row in filtered_df.iterrows():
    candidate_id = row["Source File"]
    decision_key = f"decision_{candidate_id}"
    status = st.session_state.get(decision_key, "Unreviewed")

    with st.expander(f"#{row['Rank']} • {row['Name']} — {status}"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Education:** {row['Education']}")
            st.markdown(f"**Experience:** {row['Experience (Years)']} years")
            st.markdown(f"**Projects:** {row['Projects Count']}")
            st.markdown(f"**Skills:** {row['Skills']}")
            st.markdown(f"**Certifications:** {row['Certifications'] or 'Not specified'}")
            st.markdown(f"**Salary Expectation:** {row['Salary Expectation (₹)']}")
            resume_skills = set(s.strip().lower() for s in row["Skills"].split(","))
            st.markdown("**Skill Gap Analysis**")
            st.success(f"Matched: {', '.join(jd_skill_set & resume_skills) or 'None'}")
            st.error(f"Missing: {', '.join(jd_skill_set - resume_skills) or 'None'}")
        with col2:
            st.markdown("**Score Breakdown**")
            st.progress(row["Semantic Match (%)"] / 100)
            st.progress(row["Skill Match (%)"] / 100)
            st.progress(row["Education Score (%)"] / 100)
        st.selectbox(
            "Decision",
            ["Unreviewed", "Shortlisted", "Rejected"],
            index=["Unreviewed", "Shortlisted", "Rejected"].index(status),
            key=decision_key
        )

shortlisted_df = filtered_df[
    filtered_df.apply(
        lambda r: st.session_state.get(
            f"decision_{r['Source File']}", "Unreviewed"
        ) == "Shortlisted",
        axis=1
    )
]

st.download_button(
    "📥 Download Shortlisted Candidates",
    shortlisted_df.to_csv(index=False),
    file_name="shortlisted_candidates.csv"
)
