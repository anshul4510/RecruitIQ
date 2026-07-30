import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranking.scorer import rank_candidates, filter_candidates

def test_rank_candidates_sorting():
    candidates = [
        {"filename": "cand1.pdf", "score": 0.45},
        {"filename": "cand2.pdf", "score": 0.85},
        {"filename": "cand3.pdf", "score": 0.65},
    ]
    ranked = rank_candidates(candidates)
    scores = [c["score"] for c in ranked]
    assert scores == [0.85, 0.65, 0.45]

def test_filter_candidates_no_filter():
    candidates = [
        {"filename": "cand1.pdf", "score": 0.45, "resume_json": {"skills": ["Python"]}},
        {"filename": "cand2.pdf", "score": 0.85, "resume_json": {"skills": ["Java"]}},
        {"filename": "cand3.pdf", "score": 0.65, "resume_json": {"skills": ["C++"]}},
    ]
    filtered = filter_candidates(candidates, min_score=0.0, required_skills=None)
    assert len(filtered) == 3

def test_filter_candidates_with_skills():
    candidates = [
        {"filename": "cand1.pdf", "score": 0.45, "resume_json": {"skills": ["Python", "Django"]}},
        {"filename": "cand2.pdf", "score": 0.85, "resume_json": {"skills": ["Java", "Spring"]}},
    ]
    filtered = filter_candidates(candidates, min_score=0.0, required_skills=["Python"])
    assert len(filtered) == 1
    assert filtered[0]["filename"] == "cand1.pdf"
