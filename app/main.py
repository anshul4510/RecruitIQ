import os
import sys
import json
import time
import streamlit as st
import pandas as pd
import plotly.express as px

# Sync Streamlit Secrets into os.environ if running on Streamlit Cloud
try:
    if hasattr(st, "secrets"):
        for sec_key in ["OPENAI_API_KEY", "GROQ_API_KEY"]:
            if sec_key in st.secrets and not os.getenv(sec_key):
                os.environ[sec_key] = str(st.secrets[sec_key])
except Exception:
    pass

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.pdf_utils import extract_text_from_pdf
from parser.llm_parser import parse_job_description
from resume_parser.ai_extractor import ResumeAIExtractor
from model.predict import ResumeScorerModel
from ranking.scorer import rank_candidates, filter_candidates
from explainability.explainer import analyze_and_explain, generate_outreach_email
from utils.logging_config import setup_logging
from utils.cache_manager import pipeline_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize logging
logger = setup_logging(__name__)
logger.info("RecruitIQ Application Started")

st.set_page_config(page_title="RecruitIQ", layout="wide")

# --- Premium Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    h1, h2, h3, h4 {
        color: #f1f5f9 !important;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Metric Enhancements */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(99, 102, 241, 0.2);
    }
    
    /* Expander Styling */
    .stExpander {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
        transition: all 0.3s ease;
    }
    .stExpander:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    /* Horizontal Rule */
    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* ----------- EXPLANATION BOXES ----------- */
    .box {
        background: rgba(30, 41, 59, 0.4) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s ease;
    }
    .box:hover {
        background: rgba(30, 41, 59, 0.6) !important;
        border-color: rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Box Accents based on container keys */
    div[key^="explain_"] .box { border-left: 4px solid #6366f1 !important; }
    div[key^="strength_"] .box { border-left: 4px solid #22c55e !important; }
    div[key^="weakness_"] .box { border-left: 4px solid #ef4444 !important; }

    /* Keyed Containers Styling */
    div[class*="st-key-metrics_funnel"], div[key="metrics_funnel"] {
        background: rgba(30, 41, 59, 0.3) !important;
        padding: 20px !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin-bottom: 20px !important;
    }
    
    div[class*="st-key-compare_selection"], div[key="compare_selection"] {
        background: rgba(30, 41, 59, 0.3) !important;
        padding: 15px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin-bottom: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("RecruitIQ: AI-Powered Resume Screening")

# Application State
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "jd_parsed" not in st.session_state:
    st.session_state.jd_parsed = None
if "candidate_status" not in st.session_state:
    st.session_state.candidate_status = {}
if "compare_mode" not in st.session_state:
    st.session_state.compare_mode = False
if "open_jd_dialog" not in st.session_state:
    st.session_state.open_jd_dialog = False

# JD Breakdown Modal
if hasattr(st, "dialog"):
    @st.dialog("Job Description Breakdown", width="large")
    def show_jd_modal():
        jd = st.session_state.get("jd_parsed")
        if not jd:
            st.warning("No Job Description processed yet.")
            return
        
        pos_name = jd.get("job_position_name", "Job Position")
        st.markdown(f"## {pos_name}")
        st.markdown("---")
        
        exp_req = str(jd.get("experience_requirement", "Not specified"))
        edu_req = str(jd.get("educational_requirements", "Not specified"))
        sal_req = str(jd.get("salary_range", "Not specified"))
        age_req = str(jd.get("age_requirement", "Any"))
        
        def clean_metric(val, max_len=25):
            if not val or str(val).lower() in ["none", "null", "n/a", ""]:
                return "Not specified"
            v = str(val).strip()
            if len(v) > max_len:
                return v[:max_len] + "..."
            return v

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Experience Needed", clean_metric(exp_req))
        col2.metric("Education Needed", clean_metric(edu_req))
        col3.metric("Salary Range", clean_metric(sal_req))
        col4.metric("Age Threshold", clean_metric(age_req))
        
        if len(exp_req) > 25 or len(edu_req) > 25 or (sal_req and len(sal_req) > 25):
            with st.expander("Full Requirements & Salary Notes", expanded=False):
                st.markdown(f"**Experience:** {exp_req}")
                st.markdown(f"**Education:** {edu_req}")
                st.markdown(f"**Salary / Pay:** {sal_req}")
        
        st.markdown("---")
        st.markdown("### Required & Market Tech Stack Skills")
        
        from parser.llm_parser import process_and_decompose_skills
        
        explicit = process_and_decompose_skills(jd.get("explicit_skills", []))
        inferred = process_and_decompose_skills(jd.get("inferred_skills", []))
        all_req = process_and_decompose_skills(jd.get("skills_required", []))
        
        if not explicit and not inferred and all_req:
            explicit = all_req
        
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Explicit & Basic Skills:**")
            if explicit:
                st.write(", ".join(explicit))
            else:
                st.write("N/A")
        with sc2:
            st.markdown("**AI Market-Inferred Skills:**")
            if inferred:
                st.write(", ".join(inferred))
            else:
                st.caption("None inferred.")
                
        if all_req:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**All Combined Skills:**")
            st.write(", ".join(all_req))
                
        resps = jd.get("responsibilities", [])
        if resps:
            st.markdown("---")
            st.markdown("### Key Responsibilities & Role Duties")
            for r in resps:
                st.markdown(f"- {r}")

        st.markdown("---")
        if st.button("Close Breakdown", key="btn_close_jd_dialog"):
            st.session_state.open_jd_dialog = False
            st.rerun()

    if st.session_state.open_jd_dialog:
        show_jd_modal()

# Sidebar for Uploads
with st.sidebar:
    st.header("Upload Files")
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
    resume_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if st.session_state.get("jd_parsed") and "error" not in st.session_state.jd_parsed:
        if st.button("View JD Breakdown", use_container_width=True, help="Open structured summary of the parsed Job Description"):
            st.session_state.open_jd_dialog = True
            st.rerun()
        st.markdown("---")
        
    if st.button("Process & Rank Candidates"):
        if not jd_file or not resume_files:
            st.error("Please upload both JD and at least one Resume.")
            logger.warning("Attempted to process without uploading JD or resumes.")
        else:
            logger.info(f"Starting processing pipeline for {len(resume_files)} resumes.")
            t_pipeline_start = time.time()
            runtime_stats = {
                "OCR_Latency": [],
                "LLM_Inference": [],
                "Embedding_Generation": [],
                "XGBoost_Inference": [],
                "Total_Pipeline": []
            }
            
            @st.cache_data(show_spinner=False)
            def cached_parse_jd(text):
                return parse_job_description(text)

            with st.spinner(f"Extracting Job Description features..."):
                t_jd_start = time.time()
                jd_text = extract_text_from_pdf(jd_file)
                t_jd_ocr = time.time()
                runtime_stats["OCR_Latency"].append(t_jd_ocr - t_jd_start)
                
                if not jd_text:
                    st.error("Could not extract any text from the Job Description PDF. It might be a scanned image.")
                else:
                    st.session_state.jd_parsed = cached_parse_jd(jd_text)
                    t_jd_llm = time.time()
                    runtime_stats["LLM_Inference"].append(t_jd_llm - t_jd_ocr)
                    
                    if not st.session_state.jd_parsed:
                        st.error("LLM failed to parse Job Description. This is likely an API Rate Limit or invalid key.")
                    elif "error" in st.session_state.jd_parsed:
                        st.error(f"API Error on Job Description: {st.session_state.jd_parsed['error']}")
                    else:
                        logger.info("Successfully parsed Job Description.")
            
            scorer = ResumeScorerModel()
            
            # --- NEW: JD Pre-embedding Optimization ---
            jd_embeddings = {}
            try:
                if st.session_state.jd_parsed and "error" not in st.session_state.jd_parsed:
                    logger.info("Pre-calculating JD embeddings for optimization.")
                    jd_resp = " ".join(st.session_state.jd_parsed.get('responsibilities', []))
                    jd_pos = st.session_state.jd_parsed.get('job_position_name', '')
                    jd_embeddings = {
                        'responsibilities': scorer.sbert.encode(jd_resp, convert_to_tensor=True),
                        'job_position_name': scorer.sbert.encode(jd_pos, convert_to_tensor=True)
                    }
            except Exception as e:
                logger.warning(f"JD Pre-embedding failed: {e}. Falling back to default scoring.")
            
            candidates = []
            progress_bar = st.progress(0)
            
            # Additional stats for explainer
            runtime_stats["Explainer_Latency"] = []
            
            def process_single_resume(res_file, jd_parsed, jd_txt, jd_embeds):
                t_res_start = time.time()
                try:
                    res_text = extract_text_from_pdf(res_file)
                    t_res_ocr = time.time()
                    
                    if not res_text:
                        return None, (t_res_ocr - t_res_start, 0, 0, 0, 0, time.time() - t_res_start), "No text extracted"

                    # --- GAUNTLET 1: Early Rejection Gatekeeper ---
                    if scorer.early_rejection_check(res_text, jd_txt):
                        logger.info(f"Early Rejection triggered for {res_file.name}. Bypassing pipeline.")
                        candidate_data = {
                            "filename": res_file.name,
                            "resume_json": {"name": "Rejected Candidate (Poor Match)"},
                            "score": 0.05,
                            "match_breakdown": {"Overall Score": 5.0},
                            "explanation": {"match_reason": "Instantly rejected due to extreme lack of keyword similarity.", "missing_skills": ["Multiple Core Skills"], "strength_summary": "N/A"},
                            "quality": {"quality_score": 0, "feedback": "Irrelevant document."},
                            "processing_time": time.time() - t_res_start
                        }
                        return candidate_data, (t_res_ocr - t_res_start, 0, 0, 0, 0, candidate_data["processing_time"]), None
                        
                    # --- GAUNTLET 2: MD5 Pipeline Cache ---
                    cached_data = pipeline_cache.get(res_text, jd_txt)
                    if cached_data:
                        # Append the filename to cached data in case it was renamed but identical
                        import copy
                        cached_data = copy.deepcopy(cached_data)
                        cached_data["filename"] = res_file.name
                        cached_data["processing_time"] = time.time() - t_res_start
                        return cached_data, (0, 0, 0, 0, 0, cached_data["processing_time"]), None


                    # Main Pipeline Parse
                    extractor = ResumeAIExtractor(api_key=os.getenv("OPENAI_API_KEY"))
                    res_parsed = extractor.extract(res_text)
                    if not res_parsed or res_parsed.get("name") == "Unknown":
                        logger.warning(f"LLM possibly failed to parse {res_file.name}.")
                        
                    t_res_llm = time.time()

                    # Scoring
                    score, match_breakdown, emb_time, xgb_time = scorer.predict_score(res_parsed, jd_parsed, jd_embeddings=jd_embeds)
                    
                    # --- PASS 1: Generate Dummy Explanations (LLM Deferred to Pass 2) ---
                    t_exp_start = time.time()
                    explanation = {"match_reason": "Pending Top-N Reasoning Pass.", "missing_skills": [], "strength_summary": "Skipped to save API costs."}
                    quality = {"quality_score": 0, "feedback": "Pending."}
                    t_exp_end = time.time()
                    exp_time = t_exp_end - t_exp_start

                    candidate_data = {
                        "filename": res_file.name,
                        "resume_json": res_parsed,
                        "raw_score": score,
                        "score": score,
                        "match_breakdown": match_breakdown,
                        "explanation": explanation,
                        "quality": quality,
                        "processing_time": time.time() - t_res_start,
                        "stage_times": {
                            "OCR": t_res_ocr - t_res_start,
                            "LLM": t_res_llm - t_res_ocr,
                            "Embedding": emb_time,
                            "XGBoost": xgb_time,
                            "Explainer": exp_time
                        }
                    }
                    
                    # Update Cache
                    pipeline_cache.set(res_text, jd_txt, candidate_data)
                    return candidate_data, (t_res_ocr - t_res_start, t_res_llm - t_res_ocr, emb_time, xgb_time, exp_time, candidate_data["processing_time"]), None
                except Exception as e:
                    import traceback
                    logger.error(f"Error processing {res_file.name}: {e}")
                    logger.error(traceback.format_exc())
                    return None, (time.time() - t_res_start, 0, 0, 0, 0, time.time() - t_res_start), f"Error: {str(e)}"

            with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                # Safely capture jd_parsed to avoid st.session_state thread issues
                jd_init = st.session_state.jd_parsed
                futures = {executor.submit(process_single_resume, res_file, jd_init, jd_text, jd_embeddings): res_file for res_file in resume_files}
                
                unranked = []
                for i, future in enumerate(as_completed(futures)):
                    candidate_data, stats, reason = future.result()
                    if candidate_data:
                        candidates.append(candidate_data)
                    else:
                        unranked.append({"filename": futures[future].name, "reason": reason or "Unknown failure"})
                    
                    # Unpack stats
                    runtime_stats["OCR_Latency"].append(stats[0])
                    runtime_stats["LLM_Inference"].append(stats[1])
                    runtime_stats["Embedding_Generation"].append(stats[2])
                    runtime_stats["XGBoost_Inference"].append(stats[3])
                    runtime_stats["Explainer_Latency"].append(stats[4])
                    runtime_stats["Total_Pipeline"].append(stats[5])
                    
                    progress_bar.progress((i + 1) / len(resume_files))
            
            ranked = rank_candidates(candidates)
            
            # --- PASS 2: Top-N LLM ATS Reasoning ---
            top_n = min(20, len(ranked))
            if top_n > 0:
                with st.spinner(f"Initiating ATS Reasoning Phase on Top {top_n} profiles..."):
                    for rank_idx in range(top_n):
                        cand = ranked[rank_idx]
                        if cand.get("score", 0) > 0.05: # Skip heavily rejected
                            # Force analysis via the new LLM schema
                            analysis = analyze_and_explain(cand["resume_json"], st.session_state.jd_parsed)
                            cand["explanation"] = {
                                "match_reason": analysis.get("reasoning", ""),
                                "missing_skills": analysis.get("weaknesses", []),
                                "strength_summary": ", ".join(analysis.get("strengths", []))
                            }
                            # Hack quality to map recommendations into UI
                            rec = analysis.get("recommendation", "Consider")
                            cand["quality"] = {
                                "quality_score": 9 if "Strong" in rec else (6 if "Consider" in rec else 3),
                                "feedback": rec
                            }
                            
                            ats_score = analysis.get("ats_score", 0.0)
                            
                            # Determine base score (ensure we aren't adding bonus to an already-bonused cached score)
                            base_score = cand.get("raw_score", cand["score"])
                            
                            # Integrate LLM layer back into final score (15% weight)
                            cand["score"] = min(1.0, base_score + (0.15 * float(ats_score)))
                            
            # Re-rank after applying final ATS modifiers
            ranked = rank_candidates(ranked)
            st.session_state.candidates = ranked
            st.session_state.unranked_resumes = unranked
            st.session_state.runtime_stats = runtime_stats
            st.session_state.total_time = time.time() - t_pipeline_start
            st.success("Processing Complete!")
            logger.info(f"Pipeline completed. Ranked {len(ranked)} candidates.")

# Dashboard
if st.session_state.candidates and st.session_state.jd_parsed is not None:


    with st.container(key="metrics_funnel"):
        funnel_cols = st.columns(4)
        total_cands = len(st.session_state.candidates)
        shortlisted = sum(1 for stat in st.session_state.candidate_status.values() if stat == "Shortlisted")
        rejected = sum(1 for stat in st.session_state.candidate_status.values() if stat == "Rejected")
        unreviewed = total_cands - shortlisted - rejected

        funnel_cols[0].metric("Total Ranked", total_cands, help="Ranked candidates screened")
        funnel_cols[1].metric("Shortlisted", shortlisted, help="Candidates moved forward")
        funnel_cols[2].metric("Rejected", rejected, help="Candidates dropped")
        funnel_cols[3].metric("Unreviewed", unreviewed, help="Candidates needing manual review")

    # Show unranked resumes if any
    unranked = st.session_state.get('unranked_resumes', [])
    if unranked:
        with st.expander("⚠️ Unranked Resumes (Parsing/Processing Failures)", expanded=True):
            for item in unranked:
                st.info(f"**{item['filename']}** – {item['reason']}")
        st.markdown("---")
    else:
        st.markdown("---")
    
    with st.sidebar:
        if "total_time" in st.session_state:
            st.sidebar.caption(f"&#9202; Total Efficiency: {st.session_state.total_time:.2f}s")
            
        st.markdown("---")
        st.header("Screening Filters")
        
        inferred_skills_list = st.session_state.jd_parsed.get('inferred_skills', []) if st.session_state.jd_parsed else []
        if inferred_skills_list:
            with st.expander("AI Market-Inferred Skills", expanded=False):
                st.caption("Skills automatically inferred based on modern tech stack & market standards:")
                st.write(", ".join(inferred_skills_list))

        min_score = st.slider("Minimum Match Score", 0.0, 1.0, 0.0, 0.05)
        default_skills = ", ".join(st.session_state.jd_parsed.get('skills_required', []))
        req_skills_input = st.text_input("Required Skills (comma separated)", value="", placeholder=f"e.g. {default_skills}" if default_skills else "e.g. Python, SQL")
        req_skills = [s.strip() for s in req_skills_input.split(',') if s.strip()] if req_skills_input and req_skills_input.strip() else None
        
        filtered_candidates = filter_candidates(st.session_state.candidates, min_score, req_skills)
        
        if filtered_candidates:
            st.markdown("---")
            st.header("Bulk Actions")
            k_val = st.number_input("Auto-Shortlist Top K Candidates:", min_value=1, max_value=len(filtered_candidates), value=min(5, len(filtered_candidates)), step=1)
            if st.button(f"Shortlist Top {k_val}", type="primary", use_container_width=True):
                for cand in filtered_candidates[:k_val]:
                    st.session_state.candidate_status[cand['filename']] = "Shortlisted"
                st.rerun()

    def status_sort_key(c):
        status = st.session_state.candidate_status.get(c['filename'], "Unreviewed")
        if status == "Shortlisted": return 0
        if status == "Rejected": return 2
        return 1
        
    filtered_candidates.sort(key=lambda c: (status_sort_key(c), -c.get('score', 0)))
    
    # Prepare Data for Excel Sheet Export
    export_data = []
    for c in filtered_candidates:
        res = c.get('resume_json', {})
        c_name = res.get('name')
        if not c_name or c_name.lower() == 'unknown':
            c_name = c['filename']
        
        status = st.session_state.candidate_status.get(c['filename'], "Unreviewed")
        export_data.append({
            "Candidate Name": c_name,
            "Shortlisted Status": status,
            "Email ID": res.get('email', 'N/A'),
            "Phone/Mobile Number": res.get('phone', 'N/A')
        })
    df_export = pd.DataFrame(export_data)
    csv = df_export.to_csv(index=False).encode('utf-8')
    

    
    header_col1, header_col2, header_col3 = st.columns([3, 1.2, 1])
    with header_col1:
        if not filtered_candidates:
            st.warning("No candidates match the current filters.")
        else:
            list_title = "Ranked Candidates" if not st.session_state.compare_mode else "Comparing Candidates"
            st.subheader(f"{list_title} ({len(filtered_candidates)})")
            if st.button("View JD Breakdown", key="main_header_jd_btn", help="Open parsed Job Description details"):
                st.session_state.open_jd_dialog = True
                st.rerun()
            
    with header_col2:
        compare_label = "✖ Close Compare" if st.session_state.compare_mode else "🔍 Compare"
        if st.button(compare_label, use_container_width=True):
            st.session_state.compare_mode = not st.session_state.compare_mode
            if not st.session_state.compare_mode:
                st.session_state.do_compare = False
            st.rerun()

    with header_col3:
        st.download_button(
            label="📄 Export",
            data=csv,
            file_name='candidate_details.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    st.markdown("---")

    if st.session_state.get("compare_mode", False):
        def render_candidate_comparison_view():
            candidates = st.session_state.candidates
            if len(candidates) < 2:
                st.warning("Not enough candidates to compare.")
                return

            # Selection UI
            options = {}
            for c in candidates:
                name = c.get('resume_json', {}).get('name')
                if not name or name.lower() == 'unknown':
                    name = c.get('filename', 'Unknown')
                if name in options:
                    name = f"{name} ({c.get('filename')})"
                options[name] = c

            names = list(options.keys())

            with st.container(key="compare_selection"):
                sel_col1, sel_col2, btn_col = st.columns([2, 2, 1])
                with sel_col1:
                    cand_a_name = st.selectbox("Select Candidate A", options=names, index=0)
                with sel_col2:
                    cand_b_name = st.selectbox("Select Candidate B", options=names, index=min(1, len(names)-1))
                with btn_col:
                    st.write("")
                    st.write("")
                    compare_now = st.button("Compare Now", use_container_width=True)
                    
            if "do_compare" not in st.session_state:
                st.session_state.do_compare = False
                
            if compare_now:
                st.session_state.do_compare = True
                
            if not st.session_state.do_compare:
                st.info("Select two candidates and click 'Compare Now' to see the side-by-side analysis.")
                return

            if cand_a_name == cand_b_name:
                st.warning("Please select two different candidates for comparison.")
                return

            cand_a = options[cand_a_name]
            cand_b = options[cand_b_name]

            col_a, col_b = st.columns(2)

            score_a = cand_a.get('score', 0)
            score_b = cand_b.get('score', 0)

            with col_a:
                if score_a > score_b:
                    st.markdown("### 🏆 Better Fit")
                else:
                    st.markdown("### &nbsp;")
                st.markdown(f"### {cand_a_name}")
                st.markdown("---")

            with col_b:
                if score_b > score_a:
                    st.markdown("### 🏆 Better Fit")
                else:
                    st.markdown("### &nbsp;")
                st.markdown(f"### {cand_b_name}")
                st.markdown("---")

            metrics_to_compare = [
                ("Final Score", lambda c: c.get('score', 0)),
                ("Core Skill Coverage", lambda c: c.get('match_breakdown', {}).get('core_skill_coverage', 0)),
                ("Experience Match", lambda c: c.get('match_breakdown', {}).get('experience_years_score', 0)),
                ("Semantic Relevance", lambda c: c.get('match_breakdown', {}).get('resume_jd_embedding_score', 0)),
                ("Impact Score", lambda c: c.get('match_breakdown', {}).get('impact_score', 0)),
                ("Leadership Score", lambda c: c.get('match_breakdown', {}).get('leadership_experience_score', 0)),
                ("Job Stability", lambda c: c.get('match_breakdown', {}).get('job_stability_score', 0)),
                ("Education Match", lambda c: c.get('match_breakdown', {}).get('education_match_score', 0))
            ]

            for metric_name, extractor in metrics_to_compare:
                val_a = extractor(cand_a)
                val_b = extractor(cand_b)

                diff_a = val_a - val_b
                diff_b = val_b - val_a

                delta_str_a = f"{diff_a:.2f}" if diff_a != 0 else None
                delta_str_b = f"{diff_b:.2f}" if diff_b != 0 else None

                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric(metric_name, f"{val_a:.2f}", delta=delta_str_a)
                with mc2:
                    st.metric(metric_name, f"{val_b:.2f}", delta=delta_str_b)

            st.markdown("---")

            def draw_explanations(cand, col, suffix):
                with col:
                    with st.container(key=f"explain_comp_{suffix}"):
                        st.markdown(f'''
                        <div class="box">
                            <b>Match Explanation:</b><br>
                            {cand.get('explanation', {}).get('match_reason', '')}
                        </div>
                        ''', unsafe_allow_html=True)
                    with st.container(key=f"strength_comp_{suffix}"):
                        st.markdown(f'''
                        <div class="box">
                            <b>Strengths:</b><br>
                            {cand.get('explanation', {}).get('strength_summary', '')}
                        </div>
                        ''', unsafe_allow_html=True)
                    with st.container(key=f"weakness_comp_{suffix}"):
                        miss_skills = cand.get('explanation', {}).get('missing_skills', [])
                        st.markdown(f'''
                        <div class="box">
                            <b>Missing Skills / Gaps:</b><br>
                            {", ".join(miss_skills) if miss_skills else "None found."}
                        </div>
                        ''', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            draw_explanations(cand_a, c1, "A")
            draw_explanations(cand_b, c2, "B")

            st.markdown("---")
            st.markdown("#### Skills Comparison")

            skills_a_list = cand_a.get('resume_json', {}).get('skills', [])
            skills_b_list = cand_b.get('resume_json', {}).get('skills', [])
            if not isinstance(skills_a_list, list): skills_a_list = []
            if not isinstance(skills_b_list, list): skills_b_list = []

            skills_a = set(s.strip().lower() for s in skills_a_list if isinstance(s, str))
            skills_b = set(s.strip().lower() for s in skills_b_list if isinstance(s, str))
            orig_skills_a = {s.strip().lower(): s.strip() for s in skills_a_list if isinstance(s, str)}
            orig_skills_b = {s.strip().lower(): s.strip() for s in skills_b_list if isinstance(s, str)}

            common_skills = skills_a.intersection(skills_b)
            unique_a = skills_a - skills_b
            unique_b = skills_b - skills_a

            common_html = ", ".join([f'<span style="color: #22c55e;">{orig_skills_a[s]}</span>' for s in common_skills])
            unique_a_html = ", ".join([f'<span style="color: #eab308;">{orig_skills_a[s]}</span>' for s in unique_a])
            unique_b_html = ", ".join([f'<span style="color: #eab308;">{orig_skills_b[s]}</span>' for s in unique_b])

            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"**Shared Skills:**<br>{common_html if common_html else 'None'}", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Unique Skills:**<br>{unique_a_html if unique_a_html else 'None'}", unsafe_allow_html=True)
            with sc2:
                st.markdown(f"**Shared Skills:**<br>{common_html if common_html else 'None'}", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Unique Skills:**<br>{unique_b_html if unique_b_html else 'None'}", unsafe_allow_html=True)
        
        render_candidate_comparison_view()
    
    if not st.session_state.compare_mode:
        for display_rank, cand in enumerate(filtered_candidates, 1):
            candidate_name = cand['resume_json'].get('name')
            if not candidate_name or candidate_name.lower() == 'unknown':
                candidate_name = cand['filename']
                
            score = cand['score']
            status = st.session_state.candidate_status.get(cand['filename'], "Unreviewed")
            
            status_badge = "Shortlisted" if status == "Shortlisted" else ("Rejected" if status == "Rejected" else "")
            
            if status == "Shortlisted":
                exp_title = f":green[{display_rank}. {candidate_name} | Match: {score:.2f} | {status_badge}]"
            elif status == "Rejected":
                exp_title = f":red[{display_rank}. {candidate_name} | Match: {score:.2f} | {status_badge}]"
            else:
                exp_title = f"{display_rank}. {candidate_name} | Match: {score:.2f}"
                
            with st.container(key=f"card_{cand['filename']}"):
                with st.expander(exp_title):
                    c_email = cand['resume_json'].get('email', 'N/A')
                    c_phone = cand['resume_json'].get('phone', 'N/A')
                    
                    st.markdown(f"**Name:** {candidate_name} &nbsp;&nbsp;|&nbsp;&nbsp; **Rank:** {display_rank} &nbsp;&nbsp;|&nbsp;&nbsp; **Score:** {score:.2f}")
                    st.markdown(f"**Email:** {c_email} &nbsp;&nbsp;|&nbsp;&nbsp; **Phone:** {c_phone}")
                    
                    btn_cols = st.columns([1, 1, 2, 2])
                    if btn_cols[0].button("Shortlist", key=f"sl_{cand['filename']}", help="Mark candidate as shortlisted"):
                        st.session_state.candidate_status[cand['filename']] = "Shortlisted"
                        st.rerun()
                    if btn_cols[1].button("Reject", key=f"rj_{cand['filename']}", help="Mark candidate as rejected"):
                        st.session_state.candidate_status[cand['filename']] = "Rejected"
                        st.rerun()
                    
                    if btn_cols[2].button("✉️ Draft Outreach", key=f"outreach_{cand['filename']}", help="Generate personalized outreach email"):
                        with st.spinner("Drafting personalized email..."):
                            email_draft = generate_outreach_email(cand['resume_json'], st.session_state.jd_parsed, cand.get('explanation'))
                            st.session_state[f"draft_{cand['filename']}"] = email_draft
                            
                    if f"draft_{cand['filename']}" in st.session_state:
                        st.info("✉️ **Generated Outreach Draft:**")
                        st.text_area("Copy Text", st.session_state[f"draft_{cand['filename']}"], height=200, key=f"ta_{cand['filename']}")
                        
                    st.markdown("---")
                    
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown("**Key Skills Found:**")
                        skills = cand['resume_json'].get('skills', [])
                        st.write(", ".join(skills[:12]) if isinstance(skills, list) else "N/A")
                    with c2:
                        st.markdown("**Resume Quality:**")
                        st.write(f"Score: **{cand['quality'].get('quality_score', 'N/A')}/10**")
                        st.caption(f"Signal: {cand['quality'].get('feedback', 'Calculated')}")
                        
                    st.markdown("---")
                    
                    with st.container(key=f"explain_{cand['filename']}"):
                        st.markdown(f"""
                        <div class="box">
                            <b>Match Explanation:</b><br>
                            {cand['explanation'].get('match_reason', '')}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with st.container(key=f"strength_{cand['filename']}"):
                        st.markdown(f"""
                        <div class="box">
                            <b>Strengths:</b><br>
                            {cand['explanation'].get('strength_summary', '')}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with st.container(key=f"weakness_{cand['filename']}"):
                        st.markdown(f"""
                        <div class="box">
                            <b>Missing Skills / Gaps:</b><br>
                            {", ".join(cand['explanation'].get('missing_skills', []))}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("**Match Breakdown (Key Recruiter Signals):**")
                    breakdown = cand['match_breakdown']
                    if breakdown and "rejection_reasons" in breakdown:
                        st.markdown(f"**Rejection Reasons:** {', '.join(breakdown['rejection_reasons'])}")
                         # Show only high-signal metrics for decision making
                    b_col1, b_col2, b_col3 = st.columns(3)
                    b_col1.metric("Core Skill Coverage", f"{breakdown.get('core_skill_coverage', 0):.2f}", help="Percentage of MUST-HAVE skills identified.")
                    b_col2.metric("Experience Match", f"{breakdown.get('experience_years_score', 0):.2f}", help="Alignment with required tenure.")
                    b_col3.metric("Semantic Relevance", f"{breakdown.get('resume_jd_embedding_score', 0):.2f}", help="Deep contextual overlap between JD and Resume.")
                    
                    b_col4, b_col5, b_col6 = st.columns(3)
                    b_col4.metric("Impact Score", f"{breakdown.get('impact_score', 0):.2f}", help="Frequency of measurable results (%, $, scaling).")
                    b_col5.metric("Career Progression", f"{breakdown.get('role_progression_score', 0):.2f}", help="Trajectory of seniority and growth.")
                    b_col6.metric("Job Stability", f"{breakdown.get('job_stability_score', 0):.2f}", help="Consistency of tenure across past roles.")
        
                    b_col7, b_col8, b_col9 = st.columns(3)
                    b_col7.metric("Tech Complexity", f"{breakdown.get('responsibility_complexity_score', 0):.2f}", help="Depth and density of technical responsibilities.")
                    b_col8.metric("Leadership Signal", f"{breakdown.get('leadership_experience_score', 0):.2f}", help="Mentorship, ownership, and management keywords.")
                    b_col9.metric("Education Match", f"{breakdown.get('education_match_score', 0):.2f}", help="Alignment with required degree and field.")
                    
                    st.markdown("---")
        
                    st.markdown("**Extracted Resume Data Details:**")
                    res_data = cand['resume_json']
                    
                    def clean_list_str(val):
                        if isinstance(val, list):
                            return val
                        if isinstance(val, str) and val.strip().startswith('['):
                            import ast
                            try:
                                parsed = ast.literal_eval(val)
                                if isinstance(parsed, list):
                                    return parsed
                            except:
                                cleaned = val.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
                                return [s.strip() for s in cleaned.split(',') if s.strip()]
                        return val
                        
                    groups = {
                        "Contact Info": ["email", "phone", "address", "online_links"],
                        "Objective": ["career_objective"],
                        "Skills & Languages": ["skills", "related_skils_in_job", "languages", "proficiency_levels"],
                        "Experience": ["professional_company_names", "company_urls", "positions", "role_positions", "start_dates", "end_dates", "locations", "responsibilities", "experience_years", "projects_count"],
                        "Education": ["educational_institution_name", "degree_names", "major_field_of_studies", "passing_years", "educational_results", "result_types"],
                        "Certifications": ["certification_providers", "certification_skills", "issue_dates", "expiry_dates"],
                        "Extra Curricular": ["extra_curricular_activity_types", "extra_curricular_organization_names", "extra_curricular_organization_links"]
                    }
                    
                    processed_keys = set(["name"])
                    
                    for group_name, keys in groups.items():
                        group_data = {}
                        for k in keys:
                            val = res_data.get(k)
                            if val not in [None, "", [], {}, "N/A", "Unknown", "[]", "['']"]:
                                group_data[k] = clean_list_str(val)
                                processed_keys.add(k)
                        
                        if group_data:
                            st.markdown(f"#### :gray[{group_name}]")
                            for key, value in group_data.items():
                                clean_key = key.replace('_', ' ').title()
                                
                                if isinstance(value, list):
                                    if len(value) > 0 and isinstance(value[0], str) and len(value[0]) > 60:
                                        st.markdown(f":blue[**{clean_key}:**]")
                                        for item in value:
                                            st.markdown(f"- {item}")
                                    else:
                                        st.markdown(f":blue[**{clean_key}:**] {', '.join(map(str, value))}")
                                elif isinstance(value, dict):
                                    st.markdown(f":blue[**{clean_key}:**]")
                                    st.json(value)
                                else:
                                    st.markdown(f":blue[**{clean_key}:**] {value}")
                                    
                    # Render unmapped extracted fields
                    for key, value in res_data.items():
                        if key not in processed_keys and value not in [None, "", [], {}, "N/A", "Unknown", "[]", "['']"]:
                            val = clean_list_str(value)
                            clean_key = key.replace('_', ' ').title()
                            if isinstance(val, list):
                                st.markdown(f":blue[**{clean_key}:**] {', '.join(map(str, val))}")
                            elif isinstance(value, dict):
                                st.markdown(f":blue[**{clean_key}:**]")
                                st.json(value)
                            else:
                                st.markdown(f":blue[**{clean_key}:**] {val}")
            

elif not st.session_state.candidates:
    st.info("Upload a Job Description and Resumes from the sidebar to start screening candidates!")
