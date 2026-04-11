import os
import sys
import json
import time
import streamlit as st
import pandas as pd
import plotly.express as px

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

st.title("RecruitIQ: AI-Powered Resume Screening")

# Application State
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "jd_parsed" not in st.session_state:
    st.session_state.jd_parsed = None
if "candidate_status" not in st.session_state:
    st.session_state.candidate_status = {}

# Sidebar for Uploads
with st.sidebar:
    st.header("Upload Files")
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
    resume_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    
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
    funnel_cols = st.columns(4)
    total_cands = len(st.session_state.candidates)
    shortlisted = sum(1 for stat in st.session_state.candidate_status.values() if stat == "Shortlisted")
    rejected = sum(1 for stat in st.session_state.candidate_status.values() if stat == "Rejected")
    unreviewed = total_cands - shortlisted - rejected

    funnel_cols[0].metric("Total Ranked", total_cands, help="Ranked candidates screened")
    funnel_cols[1].metric("Shortlisted", shortlisted, help="Candidates moved forward")
    funnel_cols[2].metric("Rejected", rejected, help="Candidates dropped")
    funnel_cols[3].metric("Unreviewed", unreviewed, help="Candidates needing manual review")

    st.markdown("---")
    
    # Show unranked resumes at the bottom
    unranked = st.session_state.get('unranked_resumes', [])
    if unranked:
        with st.expander("⚠️ Unranked Resumes (Parsing/Processing Failures)", expanded=True):
            for item in unranked:
                st.info(f"**{item['filename']}** – {item['reason']}")
    
    st.markdown("---")
    
    with st.sidebar:
        if "total_time" in st.session_state:
            st.sidebar.caption(f"&#9202; Total Efficiency: {st.session_state.total_time:.2f}s")
        st.markdown("---")
        st.header("Screening Filters")
        min_score = st.slider("Minimum Match Score", 0.0, 1.0, 0.0, 0.05)
        default_skills = ", ".join(st.session_state.jd_parsed.get('skills_required', []))
        req_skills_input = st.text_input("Required Skills (comma separated)", value=default_skills)
        req_skills = [s.strip() for s in req_skills_input.split(',')] if req_skills_input else None
        
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
        
    filtered_candidates.sort(key=status_sort_key)
    
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
    
    header_col1, header_col2 = st.columns([5, 1])
    with header_col1:
        if not filtered_candidates:
            st.warning("No candidates match the current filters.")
        else:
            st.subheader(f"Ranked Candidates ({len(filtered_candidates)})")
            
    with header_col2:
        st.download_button(
            label="📄 Export Data",
            data=csv,
            file_name='candidate_details.csv',
            mime='text/csv'
        )
    
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
