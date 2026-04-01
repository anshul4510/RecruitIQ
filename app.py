import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from services.resume_service import process_uploaded_resumes
from services.jd_service import build_jd_payload
from models.embeddings import ResumeEmbedder
from models.similarity import ResumeSimilarityEngine
from models.scorer import ResumeScorer
from dotenv import load_dotenv
load_dotenv()
from utils.jd_ocr import extract_jd_text

st.set_page_config(page_title="RecruitIQ", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Premium Master-Detail UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .card-container {
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        background-color: white;
        transition: all 0.2s;
    }
    .card-container:hover {
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        border-color: #cbd5e1;
    }
    .card-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #1e293b;
        margin-bottom: 5px;
    }
    .card-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 10px;
    }
    .badge {
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
        display: inline-block;
    }
    .badge-match { background-color: #d1fae5; color: #065f46; }
    .badge-missing { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: white;
        padding-top: 10px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #4F46E5;
        font-weight: 700;
    }
    /* Download Button Styling */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'active_candidate_idx' not in st.session_state:
    st.session_state.active_candidate_idx = None
if 'decisions' not in st.session_state:
    st.session_state.decisions = {}

# --- SIDEBAR: Requirements & Filters ---
st.sidebar.header("Job Requirements")
jd_file = st.sidebar.file_uploader("Optional: Upload JD PDF", type=["pdf"])
if jd_file is not None:
    if st.sidebar.button("Extract text from JD PDF"):
        with st.spinner("Running OCR..."):
            extracted = extract_jd_text(jd_file)
            if extracted and not extracted.startswith("Error:"):
                st.session_state["jd_text_cache"] = extracted
            else:
                st.sidebar.error(extracted or "Failed to extract text.")

default_jd = st.session_state.get("jd_text_cache", "Looking for a Data Scientist with Python, SQL, Machine Learning")
jd_text = st.sidebar.text_area("Job Description", value=default_jd, height=150)
jd_skills = st.sidebar.text_input("Required Skills (comma separated)", value="python, sql, machine learning")

required_experience = st.sidebar.slider("Required Experience (Years)", 0, 20, 3)

st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.header("☰ Filters")
min_score = st.sidebar.slider("Minimum Final Score", 0, 100, 0)
min_exp, max_exp = st.sidebar.slider("Experience Range", 0, 20, (0, 20))
min_skill_match = st.sidebar.slider("Minimum Skill Match (%)", 0, 100, 0)

if not uploaded_files:
    st.title("RecruitIQ : AI-Powered Resume Screening System")
    st.info("⬅ Upload Job Description (optional) and Resumes from the sidebar to start screening candidates!")
    st.stop()

# --- PROCESSING ---
@st.cache_data(show_spinner="Extracting Resume Features...")
def cached_process_resumes(files, api_key):
    # This assumes process_uploaded_resumes caches or is lightweight
    return process_uploaded_resumes(files, api_key=api_key)

@st.cache_resource
def load_embedder():
    return ResumeEmbedder()

df = cached_process_resumes(uploaded_files, os.getenv("GROQ_API_KEY"))
jd_payload = build_jd_payload(jd_text, jd_skills)

embedder = load_embedder()

if "resume_embeddings" not in st.session_state or len(st.session_state.get('processed_files', [])) != len(uploaded_files):
    # Cache invalidation based on file count
    with st.spinner("Generating Semantic Embeddings..."):
        st.session_state.resume_embeddings = embedder.embed_texts(df["resume_text"].tolist())
        st.session_state.processed_files = [f.name for f in uploaded_files]

engine = ResumeSimilarityEngine(st.session_state.resume_embeddings)
jd_embedding = embedder.embed_texts([jd_payload["text"]])

# Only top 50 in the engine for speed
scores, indices = engine.search(jd_embedding, top_k=min(50, len(df)))
scorer = ResumeScorer(required_experience=required_experience, jd_skills=jd_payload["skills"])

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
        "Source File": row["Source File"],
        "Original_Idx": idx
    })

results_df = pd.DataFrame(ranked_rows)

education_options = sorted(results_df["Education"].dropna().unique())
education_filter = st.sidebar.multiselect("Education Level", education_options, default=education_options)

filtered_df = results_df[
    (results_df["Final Score"] >= min_score) &
    (results_df["Experience (Years)"].between(min_exp, max_exp)) &
    (results_df["Skill Match (%)"] >= min_skill_match) &
    (results_df["Education"].isin(education_filter))
]

# Quick Stats / Sticky Header effect
with st.container():
    st.markdown("<div class='sticky-header'>", unsafe_allow_html=True)
    st.title("RecruitIQ ATS Dashboard")
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
    c1.metric("Candidates", len(filtered_df))
    
    shortlisted = sum(1 for _, r in filtered_df.iterrows() if st.session_state.decisions.get(r['Source File']) == 'Shortlisted')
    rejected = sum(1 for _, r in filtered_df.iterrows() if st.session_state.decisions.get(r['Source File']) == 'Rejected')
    
    c2.metric("Shortlisted", shortlisted)
    c3.metric("Rejected", rejected)
    c4.metric("Pending", len(filtered_df) - shortlisted - rejected)
    
    with c5:
        # Export logic
        short_df = filtered_df[filtered_df.apply(lambda r: st.session_state.decisions.get(r['Source File']) == 'Shortlisted', axis=1)]
        st.download_button("📥 Export Shortlisted (CSV)", short_df.to_csv(index=False), "shortlisted.csv", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# --- MASTER-DETAIL LAYOUT ---
col_list, col_gap, col_detail = st.columns([5, 0.2, 7])

# Logic for buttons
def set_decision(source_file, decision):
    st.session_state.decisions[source_file] = decision

def set_active(idx):
    st.session_state.active_candidate_idx = idx

with col_list:
    st.subheader("📋 Candidate List")
    if filtered_df.empty:
        st.warning("No candidates match the current filters.")
    else:
        # Load in batches (e.g., top 15 for speed)
        batch_df = filtered_df.head(15)
        if len(filtered_df) > 15:
            st.caption(f"Showing top 15 of {len(filtered_df)} candidates. Adjust filters to refine.")
            
        jd_skill_set = set(s.strip().lower() for s in jd_payload["skills"])
        
        for _, row in batch_df.iterrows():
            source_file = row["Source File"]
            status = st.session_state.decisions.get(source_file, "Unreviewed")
            idx = row["Original_Idx"]
            
            # Use columns to build the card structure seamlessly inside a border
            with st.container(border=True):
                icon = "🟢" if status == "Shortlisted" else "🔴" if status == "Rejected" else "⚪"
                
                # Header showing Rank, Name and decision icon
                st.markdown(f"**#{row['Rank']} | {row['Name']}** {icon}")
                st.progress(row["Final Score"] / 100.0, text=f"Fit Score: {row['Final Score']:.1f}")
                
                st.markdown(f"<span style='font-size: 0.85rem; color: #64748b;'>{row['Experience (Years)']} yrs exp | {row['Education']}</span>", unsafe_allow_html=True)
                
                # Render skills
                resume_skills_set = set(s.strip().lower() for s in str(row["Skills"]).split(",") if s.strip())
                matched = jd_skill_set & resume_skills_set
                html_pills = "".join([f"<span class='badge badge-match'>{s.title()}</span>" for s in list(matched)[:3]])
                if len(matched) > 3: 
                    html_pills += f"<span class='badge badge-match'>+{len(matched)-3}</span>"
                st.markdown(html_pills, unsafe_allow_html=True)
                
                # Buttons row using callbacks to avoid full re-runs on pure inputs
                bc1, bc2, bc3 = st.columns([1,1,1])
                with bc1:
                    st.button("✅ Shortlist", key=f"sl_{source_file}", on_click=set_decision, args=(source_file, "Shortlisted"), use_container_width=True)
                with bc2:
                    st.button("❌ Reject", key=f"rj_{source_file}", on_click=set_decision, args=(source_file, "Rejected"), use_container_width=True)
                with bc3:
                    st.button("🔎 View", key=f"vp_{source_file}", on_click=set_active, args=(idx,), use_container_width=True)

with col_detail:
    st.subheader("🎯 Insights & Analysis")
    if st.session_state.active_candidate_idx is None:
        st.info("👈 Select 'View' (🔎) on a candidate from the list to see in-depth insights.")
    else:
        active_idx = st.session_state.active_candidate_idx
        active_rows = results_df[results_df["Original_Idx"] == active_idx]
        
        if active_rows.empty:
            st.warning("Selected candidate isn't in current filter view.")
        else:
            row = active_rows.iloc[0]
            
            with st.container(border=True):
                st.markdown(f"## {row['Name']}")
                ac1, ac2, ac3 = st.columns(3)
                status = st.session_state.decisions.get(row['Source File'], 'Unreviewed')
                icon = "🟢" if status == "Shortlisted" else "🔴" if status == "Rejected" else "⚪"
                ac1.markdown(f"**Status:** {icon} {status}")
                ac2.markdown(f"**Experience:** {row['Experience (Years)']} yrs")
                ac3.markdown(f"**Expected:** {row['Salary Expectation (₹)']}")
                
                st.markdown("---")
                
                # Deep Skill Dive
                jd_skill_set = set(s.strip().lower() for s in jd_payload["skills"])
                resume_skills_set = set(s.strip().lower() for s in str(row["Skills"]).split(",") if s.strip())
                matched = jd_skill_set & resume_skills_set
                missing = jd_skill_set - resume_skills_set
                
                st.markdown("#### Skill Gap Analysis")
                matched_html = " ".join([f"<span class='badge badge-match'>{s.title()}</span>" for s in matched])
                missing_html = " ".join([f"<span class='badge badge-missing'>{s.title()}</span>" for s in missing])
                
                st.markdown(f"**Matched Required Skills:**<br>{matched_html if matched_html else 'None'}", unsafe_allow_html=True)
                st.markdown(f"<br>**Missing Requirements:**<br>{missing_html if missing_html else 'None'}", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Visualizations
                tc1, tc2 = st.columns(2)
                with tc1:
                    fig_radar = px.line_polar(
                        r=[row["Semantic Match (%)"], row["Skill Match (%)"], row["Education Score (%)"], 100], 
                        theta=["Semantics", "Skills", "Education", "Max"], 
                        line_close=True
                    )
                    fig_radar.update_traces(fill='toself', fillcolor='rgba(16, 185, 129, 0.4)', line_color='#10b981', line_width=2)
                    fig_radar.update_layout(
                        margin=dict(l=30, r=30, t=40, b=20), 
                        height=250,
                        title=dict(text="Fit Profile", font=dict(family="Inter, sans-serif", size=16, color="#1e293b")),
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100], color="#64748b"))
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                with tc2:
                    final = float(row["Final Score"])
                    sem = float(row["Semantic Match (%)"]) * 0.4
                    skl = float(row["Skill Match (%)"]) * 0.4
                    exp = min(float(row["Experience (Years)"]) * 2.0, 20.0)
                    base = final - sem - skl - exp
                    if base < 0: base = 0
                    
                    fig_water = go.Figure(go.Waterfall(
                        orientation="v",
                        measure=["absolute", "relative", "relative", "relative", "total"],
                        x=["Base", "Semantics", "Skills", "Exp", "Final"],
                        y=[base, sem, skl, exp, final],
                        connector={"line":{"color":"rgb(203, 213, 225)", "dash":"solid"}},
                        decreasing={"marker":{"color":"#ef4444", "line":{"color":"#b91c1c", "width":1}}},
                        increasing={"marker":{"color":"#10b981", "line":{"color":"#047857", "width":1}}},
                        totals={"marker":{"color":"#4F46E5", "line":{"color":"#3730a3", "width":1}}},
                        textposition="outside",
                        text=[f"{y:.1f}" for y in [base, sem, skl, exp, final]]
                    ))
                    fig_water.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20), 
                        height=250, 
                        showlegend=False,
                        title=dict(text="Score Explainability", font=dict(family="Inter, sans-serif", size=16, color="#1e293b")),
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_water, use_container_width=True)
                
                strongest_feat = "Skill Match" if row['Skill Match (%)'] > row['Semantic Match (%)'] else "Semantic Profile"
                st.info(f"💡 **AI Summary:** Scored {row['Final Score']:.1f}/100. Strongest feature: {strongest_feat}.")
