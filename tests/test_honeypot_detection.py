"""
test_honeypot_detection.py -- unit tests for the honeypot filter

Tests are anchored to REAL patterns found in the dataset sample
(e.g., CAND_0000031 with 88mo Pinecone vs 6.0 YOE).

Run:
    python -m pytest tests/ -v
  or:
    python tests/test_honeypot_detection.py
"""

import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scoring.honeypot_filter import check_honeypot, HARD_HONEYPOT_THRESHOLD


def _make_candidate(
    yoe=6.0,
    skills=None,
    career=None,
    salary_min=20.0,
    salary_max=35.0,
) -> dict:
    """Build a minimal candidate dict for testing."""
    if skills is None:
        skills = [{"name": "Python", "proficiency": "advanced",
                   "duration_months": 48, "endorsements": 10}]
    if career is None:
        career = [{"duration_months": 36, "title": "ML Engineer",
                   "company": "Acme", "description": "Built ML systems"}]
    return {
        "candidate_id": "CAND_TEST",
        "profile": {"years_of_experience": yoe, "current_title": "ML Engineer",
                    "headline": "Test", "summary": "Test"},
        "career_history": career,
        "skills": skills,
        "redrob_signals": {
            "skill_assessment_scores": {},
            "expected_salary_range_inr_lpa": {"min": salary_min, "max": salary_max},
            "recruiter_response_rate": 0.7,
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "interview_completion_rate": 0.8,
            "applications_submitted_30d": 2,
            "saved_by_recruiters_30d": 1,
            "notice_period_days": 30,
        },
    }


# ── H1: Skill duration > YOE * 12 + 6 ───────────────────────────────────────

def test_h1_skill_duration_exceeds_yoe():
    """CAND_0000031 pattern: 88 months Pinecone vs 6.0 YOE (max=78 months)"""
    c = _make_candidate(
        yoe=6.0,
        skills=[{"name": "Pinecone", "proficiency": "expert",
                 "duration_months": 88, "endorsements": 5}],
    )
    result = check_honeypot(c)
    assert result["is_honeypot"], f"Expected honeypot=True, got {result}"
    assert any("H1" in f for f in result["flags"]), f"Expected H1 flag, got {result['flags']}"
    print(f"[PASS] test_h1: score={result['honeypot_score']}, flags={result['flags']}")


def test_h1_borderline_no_flag():
    """6.0 YOE, 70 months = within buffer (max=78). Should NOT flag."""
    c = _make_candidate(
        yoe=6.0,
        skills=[{"name": "Python", "proficiency": "advanced",
                 "duration_months": 70, "endorsements": 5}],
    )
    result = check_honeypot(c)
    assert not result["is_honeypot"], f"Expected clean, got {result}"
    print(f"[PASS] test_h1_borderline: score={result['honeypot_score']}")


# -- H2: Salary min > max -----------------------------------------------------

def test_h2_inverted_salary():
    """min=15.5 > max=13.9 — exact pattern from data sample"""
    c = _make_candidate(salary_min=15.5, salary_max=13.9)
    result = check_honeypot(c)
    assert result["is_honeypot"], f"Expected honeypot=True, got {result}"
    assert any("H2" in f for f in result["flags"]), f"Expected H2 flag"
    print(f"[PASS] test_h2: flags={result['flags']}")


def test_h2_valid_salary():
    """min=20 < max=35 — normal range, no flag."""
    c = _make_candidate(salary_min=20.0, salary_max=35.0)
    result = check_honeypot(c)
    assert not any("H2" in f for f in result["flags"]), f"Unexpected H2 flag: {result['flags']}"
    print(f"[PASS] test_h2_valid")


# -- H3: Expert proficiency with zero evidence ---------------------------------

def test_h3_expert_zero_duration():
    """Expert + 0 months + 0 endorsements + no assessment = suspicious."""
    c = _make_candidate(
        yoe=3.0,
        skills=[{"name": "FAISS", "proficiency": "expert",
                 "duration_months": 0, "endorsements": 0}],
    )
    result = check_honeypot(c)
    assert any("H3" in f for f in result["flags"]), f"Expected H3 flag, got {result['flags']}"
    print(f"[PASS] test_h3: score={result['honeypot_score']}")


def test_h3_expert_with_assessment_no_flag():
    """Expert + 2 months but has an assessment score → NOT flagged."""
    c = _make_candidate(
        yoe=3.0,
        skills=[{"name": "Pinecone", "proficiency": "expert",
                 "duration_months": 2, "endorsements": 0}],
    )
    # Add assessment score to signals
    c["redrob_signals"]["skill_assessment_scores"] = {"Pinecone": 82.0}
    result = check_honeypot(c)
    assert not any("H3" in f for f in result["flags"]), f"Unexpected H3 flag: {result['flags']}"
    print(f"[PASS] test_h3_with_assessment")


# -- H4: Career total >> YOE --------------------------------------------------

def test_h4_career_total_hard():
    """Career total = 300 months vs YOE=6 (72 months) → ratio=4.17 > 2.0"""
    c = _make_candidate(
        yoe=6.0,
        career=[
            {"duration_months": 150, "title": "Eng", "company": "A", "description": "x"},
            {"duration_months": 150, "title": "Eng", "company": "B", "description": "x"},
        ],
    )
    result = check_honeypot(c)
    assert result["is_honeypot"], f"Expected honeypot=True, got {result}"
    assert any("H4" in f for f in result["flags"]), f"Expected H4 flag"
    print(f"[PASS] test_h4_hard: score={result['honeypot_score']}")


def test_h4_career_total_reasonable():
    """Career total = 80 months vs YOE=6 (72 months) → ratio=1.11, normal."""
    c = _make_candidate(
        yoe=6.0,
        career=[
            {"duration_months": 40, "title": "Eng", "company": "A", "description": "x"},
            {"duration_months": 40, "title": "Eng", "company": "B", "description": "x"},
        ],
    )
    result = check_honeypot(c)
    assert not any("H4" in f for f in result["flags"]), f"Unexpected H4 flag: {result['flags']}"
    print(f"[PASS] test_h4_reasonable")


# -- Clean candidate — no flags ------------------------------------------------

def test_clean_candidate():
    """A realistic, clean candidate should score honeypot=0 and is_honeypot=False."""
    c = _make_candidate(
        yoe=7.0,
        skills=[
            {"name": "Python", "proficiency": "expert", "duration_months": 60, "endorsements": 25},
            {"name": "Elasticsearch", "proficiency": "advanced", "duration_months": 36, "endorsements": 12},
            {"name": "FAISS", "proficiency": "intermediate", "duration_months": 24, "endorsements": 5},
        ],
        career=[
            {"duration_months": 36, "title": "ML Engineer", "company": "Startup",
             "description": "Built and deployed semantic search system using Elasticsearch + FAISS."},
            {"duration_months": 48, "title": "Data Scientist", "company": "ProductCo",
             "description": "Shipped recommendation system serving 1M users."},
        ],
        salary_min=30.0,
        salary_max=50.0,
    )
    result = check_honeypot(c)
    assert not result["is_honeypot"], f"Expected clean, got {result}"
    assert result["honeypot_score"] == 0.0, f"Expected score=0, got {result['honeypot_score']}"
    print(f"[PASS] test_clean: score={result['honeypot_score']}, flags={result['flags']}")


# -- H5: Multiple expert skills with zero duration ----------------------------

def test_h5_many_expert_zero_duration():
    """3+ expert skills all with duration=0 → clear fabrication pattern."""
    c = _make_candidate(
        yoe=5.0,
        skills=[
            {"name": "Pinecone", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
            {"name": "FAISS", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
            {"name": "Weaviate", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
        ],
    )
    result = check_honeypot(c)
    assert any("H5" in f for f in result["flags"]), f"Expected H5 flag, got {result['flags']}"
    print(f"[PASS] test_h5: score={result['honeypot_score']}")


if __name__ == "__main__":
    tests = [
        test_h1_skill_duration_exceeds_yoe,
        test_h1_borderline_no_flag,
        test_h2_inverted_salary,
        test_h2_valid_salary,
        test_h3_expert_zero_duration,
        test_h3_expert_with_assessment_no_flag,
        test_h4_career_total_hard,
        test_h4_career_total_reasonable,
        test_clean_candidate,
        test_h5_many_expert_zero_duration,
    ]

    passed = failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
