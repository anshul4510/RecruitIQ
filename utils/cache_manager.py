import hashlib
import logging

logger = logging.getLogger(__name__)

class ProcessCache:
    """
    In-memory L1 cache using md5 hashing to bypass the entire pipeline 
    if an exact pairing of Resume and Job Description is re-uploaded.
    """
    def __init__(self):
        self._cache = {}

    def _generate_hash(self, resume_text: str, jd_text: str) -> str:
        # Standardize strings to prevent trailing space false negatives
        rt = resume_text.strip() if resume_text else ""
        jt = jd_text.strip() if jd_text else ""
        combined = f"{rt}|||{jt}".encode('utf-8')
        return hashlib.md5(combined).hexdigest()

    def get(self, resume_text: str, jd_text: str):
        key = self._generate_hash(resume_text, jd_text)
        if key in self._cache:
            logger.info("⚡ L1 Cache HIT: Bypassing entire processing pipeline.")
            return self._cache[key]
        return None

    def set(self, resume_text: str, jd_text: str, candidate_data: dict):
        key = self._generate_hash(resume_text, jd_text)
        self._cache[key] = candidate_data
        logger.info("L1 Cache SET: Candidate data successfully cached.")

# Global singleton instance (persists during Streamlit session if cached in state, or within the module namespace for basic async loops)
pipeline_cache = ProcessCache()
