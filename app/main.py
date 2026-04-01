import os
import sys
import json
import streamlit as st
import pandas as pd

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.pdf_utils import extract_text_from_pdf
from parser.llm_parser import parse_job_description
from resume_parser.ai_extractor import ResumeAIExtractor
from model.predict import ResumeScorerModel
from ranking.scorer import rank_candidates, filter_candidates
from explainability.explainer import generate_explanation, analyze_resume_quality

st.set_page_config(page_title="RecruitIQ", layout="wide")

st.title("RecruitIQ: AI-Powered Resume Screening")

# Application State
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "jd_parsed" not in st.session_state:
    st.session_state.jd_parsed = None

# Sidebar for Uploads
with st.sidebar:
    st.header("Upload Files")
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
    resume_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process & Rank Candidates"):
        if not jd_file or not resume_files:
            st.error("Please upload both JD and at least one Resume.")
        else:
            with st.spinner(f"Extracting Job Description features..."):
                jd_text = extract_text_from_pdf(jd_file)
                if not jd_text:
                    st.error("Could not extract any text from the Job Description PDF. It might be a scanned image.")
                else:
                    st.session_state.jd_parsed = parse_job_description(jd_text)
                    if not st.session_state.jd_parsed:
                        st.error("LLM failed to parse Job Description. This is likely a Groq API Rate Limit or invalid key.")
                    elif "error" in st.session_state.jd_parsed:
                        st.error(f"Groq API Error on Job Description: {st.session_state.jd_parsed['error']}")
            
            scorer = ResumeScorerModel()
            
            candidates = []
            progress_bar = st.progress(0)
            
            for i, res_file in enumerate(resume_files):
                with st.spinner(f"Processing Resume {i+1}/{len(resume_files)}: {res_file.name}..."):
                    res_text = extract_text_from_pdf(res_file)
                    if not res_text:
                        st.warning(f"No text extracted from {res_file.name}.")
                        res_parsed = {}
                    else:
                        extractor = ResumeAIExtractor(api_key=os.getenv("GROQ_API_KEY"))
                        res_parsed = extractor.extract(res_text)
                        if not res_parsed or res_parsed.get("name") == "Unknown":
                            st.warning(f"LLM possibly failed to parse {res_file.name}. Check Groq API Key or Rate Limit.")

                    score, match_breakdown = scorer.predict_score(res_parsed, st.session_state.jd_parsed)
                    explanation = generate_explanation(res_parsed, st.session_state.jd_parsed)
                    quality = analyze_resume_quality(res_parsed)

                    candidate_data = {
                        "filename": res_file.name,
                        "resume_json": res_parsed,
                        "score": score,
                    "match_breakdown": match_breakdown,
                    "explanation": explanation,
                    "quality": quality
                }
                candidates.append(candidate_data)
                progress_bar.progress((i + 1) / len(resume_files))
                
            ranked = rank_candidates(candidates)
            st.session_state.candidates = ranked
            st.success("Processing Complete!")

# Dashboard
if st.session_state.candidates and st.session_state.jd_parsed is not None:
    # Filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum Match Score", 0.0, 1.0, 0.0, 0.05)
    with col2:
        default_skills = ", ".join(st.session_state.jd_parsed.get('skills_required', []))
        req_skills_input = st.text_input("Required Skills (comma separated)", value=default_skills)
        req_skills = [s.strip() for s in req_skills_input.split(',')] if req_skills_input else None
        
    filtered_candidates = filter_candidates(st.session_state.candidates, min_score, req_skills)
    
    st.subheader(f"Ranked Candidates ({len(filtered_candidates)})")
    
    for rank, cand in enumerate(filtered_candidates, 1):
        with st.expander(f"{rank}. {cand['filename']} - Score: {cand['score']:.2f}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Key Skills:**")
                skills = cand['resume_json'].get('skills', [])
                st.write(", ".join(skills[:10]) if isinstance(skills, list) else "N/A")
            with c2:
                st.markdown("**Missing Skills:**")
                missing = cand['explanation'].get('missing_skills', [])
                st.write(", ".join(missing) if isinstance(missing, list) else "None identified")
            with c3:
                st.markdown("**Resume Quality Score:**")
                st.write(f"{cand['quality'].get('quality_score', 'N/A')}/10")
                
            st.markdown("---")
            st.markdown("**Match Explanation (LLM):**")
            st.write(cand['explanation'].get('match_reason', ''))
            
            st.markdown("**Strength Summary:**")
            st.write(cand['explanation'].get('strength_summary', ''))
            
            st.markdown("---")
            st.markdown("**Match Breakdown (Features):**")
            st.json(cand['match_breakdown'])
            
            st.markdown("**Extracted Resume Data:**")
            st.json(cand['resume_json'])
            
    # Download as CSV functionality
    st.subheader("Export Results")
    export_data = []
    for c in filtered_candidates:
        export_data.append({
            "Filename": c['filename'],
            "Score": c['score'],
            "Quality_Score": c['quality'].get('quality_score'),
            "Strength": c['explanation'].get('strength_summary')
        })
    df_export = pd.DataFrame(export_data)
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='resume_ranking_results.csv',
        mime='text/csv',
    )
elif not st.session_state.candidates:
    st.info("Upload a Job Description and Resumes from the sidebar to start screening candidates!")
