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
    funnel_cols = st.columns(4)
    total_cands = len(st.session_state.candidates)
    shortlisted = sum(1 for stat in st.session_state.candidate_status.values() if stat == "Shortlisted")
    rejected = sum(1 for stat in st.session_state.candidate_status.values() if stat == "Rejected")
    unreviewed = total_cands - shortlisted - rejected
    
    funnel_cols[0].metric("Total Candidates", total_cands, help="Total candidates screened")
    funnel_cols[1].metric("Shortlisted", shortlisted, help="Candidates moved forward")
    funnel_cols[2].metric("Rejected", rejected, help="Candidates dropped")
    funnel_cols[3].metric("Unreviewed", unreviewed, help="Candidates needing manual review")
    
    st.markdown("---")
    
    with st.sidebar:
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
            
            btn_cols = st.columns([1, 1, 4])
            if btn_cols[0].button("Shortlist", key=f"sl_{cand['filename']}", help="Mark candidate as shortlisted"):
                st.session_state.candidate_status[cand['filename']] = "Shortlisted"
                st.rerun()
            if btn_cols[1].button("Reject", key=f"rj_{cand['filename']}", help="Mark candidate as rejected"):
                st.session_state.candidate_status[cand['filename']] = "Rejected"
                st.rerun()
                
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
            st.markdown("**Match Breakdown (Features):**")
            breakdown = cand['match_breakdown']
            if breakdown:
                strongest_key = max(breakdown, key=breakdown.get)
                strongest_val = breakdown[strongest_key]
                st.markdown(f"**Strongest Factor:** {strongest_key.replace('_', ' ').title()} ({strongest_val:.2f})", help="This was the highest individual matching feature.")
            b_col1, b_col2, b_col3 = st.columns(3)
            b_col1.metric("Skill Overlap", f"{breakdown.get('skill_overlap_score', 0):.2f}")
            b_col2.metric("Education Match", f"{breakdown.get('education_match_score', 0):.2f}")
            b_col3.metric("Experience Match", f"{breakdown.get('experience_years_score', breakdown.get('experience_match_score', 0)):.2f}")
            
            b_col4, b_col5, b_col6 = st.columns(3)
            b_col4.metric("Responsibility Sim", f"{breakdown.get('responsibility_similarity_score', 0):.2f}")
            b_col5.metric("Language Match", f"{breakdown.get('language_match_score', 0):.2f}")
            b_col6.metric("Certification Match", f"{breakdown.get('certification_match_score', 0):.2f}")

            b_col7, b_col8, b_col9 = st.columns(3)
            b_col7.metric("Projects Score", f"{breakdown.get('projects_count_score', 0):.2f}")
            b_col8.metric("Seniority Match", f"{breakdown.get('seniority_match_score', 0):.2f}")
            b_col9.metric("Skill Breadth", f"{breakdown.get('skill_breadth_score', 0):.2f}")

            b_col10, b_col11, b_col12 = st.columns(3)
            b_col10.metric("Job Stability", f"{breakdown.get('job_stability_score', 0):.2f}")
            b_col11.metric("Online Presence", f"{breakdown.get('online_presence_score', 0):.2f}")
            b_col12.metric("Resp. Depth", f"{breakdown.get('responsibility_depth_score', 0):.2f}")
            
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
