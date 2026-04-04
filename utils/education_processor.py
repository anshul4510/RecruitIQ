import re
import pandas as pd

class EducationProcessor:
    """
    Expert system for extracting, normalizing, and scoring education details from resumes.
    Hybrid approach: Regex + Metadata-driven normalization.
    """
    
    # Standardized Hierarchy
    DEGREE_HIERARCHY = {
        "PHD": 5,
        "MASTER": 4,
        "BACHELOR": 3,
        "DIPLOMA": 2,
        "HIGHSCHOOL": 1,
        "NONE": 0
    }

    # Robust Regex Patterns for Degree Extraction
    DEGREE_PATTERNS = {
        "PHD": r'\b(ph\.?d|doctorate|doctor of philosophy|d\.phil)\b',
        "MASTER": r'\b(m\.?tech|m\.?s|m\.?sc|m\.?a|mba|mca|pgdm|master\'?s?|post grad)\b',
        "BACHELOR": r'\b(b\.?tech|b\.?e|b\.?sc|b\.?a|bba|bca|b\.?com|b\.?ed|bachelor\'?s?|grad)\b',
        "DIPLOMA": r'\b(diploma|polytechnic|iti|associate degree)\b',
        "HIGHSCHOOL": r'\b(high school|12th|10th|hsc|ssc|intermediate|senior secondary|matriculation)\b'
    }

    # Semantic Major Groups
    MAJOR_GROUPS = {
        "COMPUTER_SCIENCE": ['computer science', 'cse', 'it', 'information technology', 'software engineering', 'computer applications', 'cs', 'data science', 'ai', 'ml', 'artificial intelligence'],
        "ENGINEERING": ['mechanical', 'civil', 'electrical', 'ece', 'electronics', 'ee', 'eee', 'me', 'ce', 'instrumentation', 'production'],
        "MANAGEMENT": ['management', 'business', 'marketing', 'finance', 'hr', 'operations', 'admin'],
        "SCIENCE": ['physics', 'chemistry', 'mathematics', 'statistics', 'biology', 'biotechnology']
    }

    @staticmethod
    def normalize_degree(text):
        """Maps raw degree text to hierarchy level."""
        if not text:
            return "NONE", 0
        
        text = str(text).lower()
        for level, pattern in EducationProcessor.DEGREE_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return level, EducationProcessor.DEGREE_HIERARCHY[level]
        
        return "NONE", 0

    @staticmethod
    def normalize_major(text):
        """Groups majors into semantic buckets."""
        if not text:
            return "GENERAL"
            
        text = str(text).lower()
        for group, synonyms in EducationProcessor.MAJOR_GROUPS.items():
            if any(syn in text for syn in synonyms):
                return group
        return text.upper()

    @staticmethod
    def extract_education_simple(text):
        """
        Extracts education details using regex + patterns.
        Used as a high-speed alternative/fallback to LLM.
        """
        if not text:
            return []
            
        # 1. Extract Years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        
        # 2. Extract Degrees
        extracted = []
        lines = text.split('\n')
        for line in lines:
            level, rank = EducationProcessor.normalize_degree(line)
            if rank > 0:
                major = "General"
                for group, synonyms in EducationProcessor.MAJOR_GROUPS.items():
                    if any(syn in line.lower() for syn in synonyms):
                        major = group
                        break
                
                # Simple Institution heuristic (lines containing 'university', 'college', 'iit', 'nit')
                inst = "Unknown Institution"
                inst_match = re.search(r'\b([A-Z][a-zA-Z\s]+(University|College|Institute|IIT|NIT|School))\b', line)
                if inst_match:
                    inst = inst_match.group(0)
                
                extracted.append({
                    "degree_level": level,
                    "rank": rank,
                    "raw_degree": line.strip()[:50],
                    "major": major,
                    "institution": inst
                })
        
        return extracted

    @staticmethod
    def calculate_match_score(candidate_degrees, jd_requirement):
        """
        Computes the final education match score (0-1).
        Hierarchical: Higher degree always satisfies lower degree requirements.
        """
        jd_edu_str = str(jd_requirement).lower()
        
        # JD level extraction
        jd_level, jd_rank = EducationProcessor.normalize_degree(jd_edu_str)
        # If JD has no specific requirement, return 1.0 (Fair Match)
        if jd_rank == 0:
            return 1.0
            
        if not candidate_degrees:
            return 0.0

        best_score = 0.0
        
        jd_major_group = EducationProcessor.normalize_major(jd_edu_str)
        
        for cand_edu in candidate_degrees:
            # Handle both raw strings and dicts from extraction
            if isinstance(cand_edu, str):
                c_level, c_rank = EducationProcessor.normalize_degree(cand_edu)
                c_major_group = EducationProcessor.normalize_major(cand_edu)
            else:
                c_level = cand_edu.get('degree_level', 'NONE')
                c_rank = cand_edu.get('rank', EducationProcessor.DEGREE_HIERARCHY.get(c_level, 0))
                c_major_group = EducationProcessor.normalize_major(cand_edu.get('major', 'None'))

            # Degree Match Logic (60%)
            degree_score = 0.0
            if c_rank >= jd_rank:
                degree_score = 1.0
            elif c_rank == jd_rank - 1:
                degree_score = 0.5
            
            # Field Similarity Logic (40%)
            field_score = 0.0
            if jd_major_group == "GENERAL":
                field_score = 1.0 # JD didn't specify major
            elif c_major_group == jd_major_group:
                field_score = 1.0
            elif (jd_major_group == "COMPUTER_SCIENCE" and c_major_group == "ENGINEERING") or \
                 (jd_major_group == "ENGINEERING" and c_major_group == "COMPUTER_SCIENCE"):
                field_score = 0.7 # Related technical overlap
            
            total = (degree_score * 0.6) + (field_score * 0.4)
            best_score = max(best_score, total)
            
        return round(best_score, 2)
