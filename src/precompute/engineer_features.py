"""
engineer_features.py — Stage B precompute (Step 2, the core work)

Reads candidates.jsonl + artifacts/skill_taxonomy.json
Produces artifacts/features.parquet with one row per candidate.

ALL computation is vectorised pandas/numpy — no per-candidate Python loops
in the hot path. Runs once offline; rank.py only loads the parquet.

Columns produced
────────────────
candidate_id
yoe                         years_of_experience
# Skill evidence (one per must-have category)
skill_embeddings_retrieval  0-1
skill_vector_db             0-1
skill_python                0-1
skill_ranking_eval          0-1
skill_evidence_score        0-1  weighted average of above 4
# Nice-to-have bonus
nice_to_have_score          0-1
# Domain fit from career_history descriptions
domain_fit_score            0-1
# Trajectory
trajectory_score            0-1
yoe_band_score              0-1  (peaks at 6-8 yrs)
is_pure_consulting          bool
consulting_penalty          0-1  (0.0 = hard zero, 0.75 = one role)
title_chaser_flag           bool
research_only_flag          bool
# Honeypot
is_honeypot                 bool
honeypot_score              0-1
# Behavioral
behavioral_multiplier       0.30-1.00
# Location
location_fit                0-1
# Disqualifier aggregate
disqualifier_penalty        0-1
# Final (pre-embedding, no behavioral applied yet — applied in rank.py)
base_fit_no_emb             0-1  (used when embeddings not available)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Make src importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.scoring.honeypot_filter import check_honeypot
from src.scoring.behavioral_multiplier import compute_behavioral_multiplier

# ── Constants ─────────────────────────────────────────────────────────────────

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "tata consultancy services",
    "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "techmahindra",
    "hexaware", "mphasis",
}

PRODUCTION_WORDS = {
    "shipped", "deployed", "deployment", "production", "prod",
    "launched", "launch", "scaled", "serving", "serving infrastructure",
    "real users", "at scale", "maintained and", "led development", "developed and",
    "inference pipeline", "mlops", "built a ranking", "deployed model"
}

DOMAIN_WORDS = {
    "search", "ranking", "ranker", "retrieval", "recommendation",
    "recommender", "embeddings", "embedding", "semantic search", "vector search",
    "reranking", "rerank", "re-rank", "llm", "rag", "fine-tune", "fine-tuning",
    "ndcg", "mrr", "a/b test", "ab test", "information retrieval",
    "candidate matching", "job matching",
    "hybrid search", "dense retrieval", "sparse retrieval",
    "bert", "transformer", "sentence transformers", "sentence-transformers"
}

RESEARCH_WORDS = {
    "published paper", "research paper", "arxiv", "research lab",
    "phd research", "thesis", "academic", "research intern",
    "research scientist", "research engineer", "ablation",
    "state of the art", "sota", "benchmark dataset",
}

# Pre-compile regex patterns with word boundaries
_PROD_PATTERNS = [re.compile(r'\b' + re.escape(w) + r'\b') for w in PRODUCTION_WORDS]
_DOMAIN_PATTERNS = [re.compile(r'\b' + re.escape(w) + r'\b') for w in DOMAIN_WORDS]
_RESEARCH_PATTERNS = [re.compile(r'\b' + re.escape(w) + r'\b') for w in RESEARCH_WORDS]

SEO_NEGATIVE_PATTERNS = [
    re.compile(r'\bsearch engine optimization\b'),
    re.compile(r'\bseo\b'),
    re.compile(r'\bfirst page\b'),
    re.compile(r'\bgoogle ranking\b'),
    re.compile(r'\bkeyword\b'),
    re.compile(r'\bblog post\b')
]

PROFICIENCY_WEIGHT = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.80, "expert": 1.00}

LOCATION_SCORES = {
    "pune": 1.0, "noida": 1.0,
    "hyderabad": 0.85, "mumbai": 0.85, "delhi": 0.85,
    "delhi ncr": 0.85, "gurugram": 0.85, "gurgaon": 0.85,
    "bengaluru": 0.85, "bangalore": 0.85, "new delhi": 0.85,
    "chennai": 0.70, "kolkata": 0.70, "ahmedabad": 0.70,
    "jaipur": 0.65, "kochi": 0.65, "coimbatore": 0.65,
}

MUST_HAVE_WEIGHTS = {
    "embeddings_retrieval": 0.30,
    "vector_db_hybrid_search": 0.25,
    "python": 0.20,
    "ranking_evaluation": 0.25,
}

NICE_HAVE_WEIGHTS = {
    "llm_finetuning": 0.25,
    "learning_to_rank": 0.25,
    "hr_tech": 0.20,
    "distributed_systems": 0.20,
    "open_source": 0.10,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lower(s) -> str:
    return str(s).lower().strip() if s else ""


def _contains_any(text: str, compiled_patterns: list[re.Pattern]) -> bool:
    tl = text.lower()
    return any(p.search(tl) for p in compiled_patterns)


def _count_matches(text: str, compiled_patterns: list[re.Pattern]) -> int:
    tl = text.lower()
    return sum(1 for p in compiled_patterns if p.search(tl))


# ── Skill Evidence Score ──────────────────────────────────────────────────────

def compute_skill_evidence(skills: list, taxonomy: dict, assessment_scores: dict) -> dict:
    """
    For each must-have category, score 0-1 combining:
      - proficiency weight
      - duration_months (log-scaled, saturates at 48mo)
      - endorsements (log-scaled, saturates at 50)
      - assessment score bonus (strongest anti-stuffing signal)
    """
    cat_scores: dict[str, float] = {cat: 0.0 for cat in MUST_HAVE_WEIGHTS}
    nice_cat_scores: dict[str, float] = {cat: 0.0 for cat in NICE_HAVE_WEIGHTS}

    for sk in skills:
        name = sk.get("name", "")
        proficiency = _lower(sk.get("proficiency", "beginner"))
        duration = max(sk.get("duration_months", 0) or 0, 0)
        endorsements = max(sk.get("endorsements", 0) or 0, 0)

        # Taxonomy lookup — try exact, then normalized
        tax_entry = taxonomy.get(name) or taxonomy.get(name.lower())
        if not tax_entry:
            continue

        categories: list = tax_entry.get("categories", [])
        if not categories:
            continue

        # Component scores
        prof_w = PROFICIENCY_WEIGHT.get(proficiency, 0.25)
        dur_w = np.log1p(min(duration, 48)) / np.log1p(48)   # 0-1, saturates at 48mo
        end_w = np.log1p(min(endorsements, 50)) / np.log1p(50)  # 0-1, saturates at 50

        # Assessment bonus: objective platform score trumps self-report
        assess_score = assessment_scores.get(name, -1)
        if assess_score >= 0:
            assess_w = assess_score / 100.0
        else:
            assess_w = 0.0

        # Combined skill score — assessment bonus lifts significantly
        if assess_score >= 0:
            skill_score = 0.20 * prof_w + 0.25 * dur_w + 0.15 * end_w + 0.40 * assess_w
        else:
            skill_score = 0.35 * prof_w + 0.45 * dur_w + 0.20 * end_w

        # Assign to categories — take max across multiple skills in same category
        for cat in categories:
            if cat in cat_scores:
                cat_scores[cat] = max(cat_scores[cat], skill_score)
            if cat in nice_cat_scores:
                nice_cat_scores[cat] = max(nice_cat_scores[cat], skill_score)

    # Weighted average for must-haves
    skill_evidence = sum(
        MUST_HAVE_WEIGHTS[cat] * cat_scores[cat] for cat in MUST_HAVE_WEIGHTS
    )

    # Weighted average for nice-to-haves (bonus, capped at 1.0)
    nice_score = sum(
        NICE_HAVE_WEIGHTS[cat] * nice_cat_scores[cat] for cat in NICE_HAVE_WEIGHTS
    )

    result = {cat: round(cat_scores[cat], 4) for cat in MUST_HAVE_WEIGHTS}
    result["skill_evidence_score"] = round(min(skill_evidence, 1.0), 4)
    result["nice_to_have_score"] = round(min(nice_score, 1.0), 4)
    result.update({f"nice_{cat}": round(nice_cat_scores[cat], 4) for cat in NICE_HAVE_WEIGHTS})
    return result


# ── Domain Fit Score ─────────────────────────────────────────────────────────

def compute_domain_fit(career_history: list, current_title: str, summary: str) -> float:
    """
    Scan career_history[].description for:
      - Production evidence words (shipped, deployed, at scale...)
      - Domain-specific terms (ranking, retrieval, embeddings...)
    Also checks title family and summary for supporting signal.

    A candidate with right title but empty descriptions scores LOWER than
    one with "Data Engineer" title but descriptions full of ML pipeline work.
    This is the JD's Tier-5 mechanism.

    Domain fit is penalised hard if title is in a totally non-technical family
    (HR Manager, Accountant, Sales, Content Writer, Graphic Designer, etc.)
    AND career descriptions contain no technical domain language at all.
    """
    NON_TECH_TITLES = {
        "hr manager", "human resources", "accountant", "accounting",
        "content writer", "copywriter", "graphic designer", "designer",
        "sales executive", "sales manager", "marketing manager",
        "customer support", "customer service", "civil engineer",
        "mechanical engineer", "operations manager", "business development",
    }

    # Gate: is the candidate in a non-technical title family?
    title_lower = _lower(current_title)
    is_non_tech_title = any(nt in title_lower for nt in NON_TECH_TITLES)

    # Scan all career descriptions + summary
    all_text = " ".join(
        [r.get("description", "") or "" for r in career_history]
        + [summary or ""]
    ).lower()

    # Negative context guard for SEO false positives
    has_seo_context = any(p.search(all_text) for p in SEO_NEGATIVE_PATTERNS)
    
    prod_count = _count_matches(all_text, _PROD_PATTERNS)
    domain_count = _count_matches(all_text, _DOMAIN_PATTERNS)
    research_count = _count_matches(all_text, _RESEARCH_PATTERNS)

    # If it looks like SEO and the title is non-technical, aggressively zero the domain count
    # Since "search" and "ranking" are in DOMAIN_WORDS
    if has_seo_context and is_non_tech_title:
        domain_count = 0

    # Penalty for non-tech title with zero domain signal
    if is_non_tech_title and domain_count == 0:
        return 0.02   # hard gate — matches JD's "Marketing Manager is not a fit"

    # Base score: combination of production evidence + domain terms
    prod_score = min(prod_count / 6.0, 1.0)   # saturates at 6 production hits
    domain_score = min(domain_count / 5.0, 1.0)  # saturates at 5 domain hits

    # Research penalty: purely academic language with no production language
    if research_count > 0 and prod_count == 0:
        research_penalty = max(0.0, 1.0 - research_count * 0.15)
    else:
        research_penalty = 1.0

    # Title bonus: ML/AI/Search titles boost the score
    TECH_TITLE_WORDS = {
        "machine learning", "ml engineer", "ai engineer", "data scientist",
        "search engineer", "ranking", "recommendation", "nlp", "research scientist",
        "applied scientist", "software engineer", "data engineer", "backend engineer",
        "full stack", "platform engineer",
    }
    title_bonus = 0.10 if any(tw in title_lower for tw in TECH_TITLE_WORDS) else 0.0

    raw = (0.45 * prod_score + 0.45 * domain_score + 0.10 * title_bonus) * research_penalty
    return round(min(max(raw, 0.0), 1.0), 4)


# ── Trajectory Score ─────────────────────────────────────────────────────────

def compute_trajectory(
    yoe: float,
    career_history: list,
    current_title: str,
    jd: dict,
) -> dict:
    """
    Returns:
      yoe_band_score       float 0-1
      is_pure_consulting   bool
      consulting_penalty   float 0-1
      title_chaser_flag    bool
      research_only_flag   bool
      trajectory_score     float 0-1
    """
    # ── YOE band score (peaks at 6-8, graceful decay) ────────────────────
    if 6 <= yoe <= 8:
        yoe_band = 1.00
    elif 5 <= yoe < 6:
        yoe_band = 0.90
    elif 8 < yoe <= 9:
        yoe_band = 0.90
    elif 4 <= yoe < 5:
        yoe_band = 0.75
    elif 9 < yoe <= 11:
        yoe_band = 0.75
    elif 3 <= yoe < 4:
        yoe_band = 0.55
    elif 11 < yoe <= 13:
        yoe_band = 0.60
    elif yoe < 3:
        yoe_band = max(0.20, yoe / 3.0 * 0.55)
    else:
        yoe_band = max(0.30, 0.60 - (yoe - 13) * 0.04)

    # ── Consulting career check ───────────────────────────────────────────
    companies = [_lower(r.get("company", "")) for r in career_history]
    consulting_roles = [
        c for c in companies
        if any(firm in c for firm in CONSULTING_FIRMS)
    ]
    is_pure_consulting = len(consulting_roles) == len(companies) and len(companies) > 0

    if is_pure_consulting:
        consulting_penalty = 0.0    # hard zero for entire career at consulting
    elif len(consulting_roles) > 0:
        ratio = len(consulting_roles) / max(len(companies), 1)
        consulting_penalty = max(0.50, 1.0 - ratio * 0.50)
    else:
        consulting_penalty = 1.0

    # ── Title chaser detection ────────────────────────────────────────────
    SENIORITY = {
        "intern": 0, "junior": 1, "associate": 2, "": 3,
        "mid": 3, "senior": 4, "staff": 5, "principal": 6,
        "director": 7, "vp": 8, "head": 7, "chief": 9,
    }

    def _seniority(title_str: str) -> int:
        tl = _lower(title_str)
        for word, level in sorted(SENIORITY.items(), key=lambda x: -x[1]):
            if word and word in tl:
                return level
        return 3  # default mid-level

    if len(career_history) >= 3:
        sorted_roles = sorted(career_history, key=lambda r: r.get("start_date", ""))
        seniority_levels = [_seniority(r.get("title", "")) for r in sorted_roles]
        durations = [r.get("duration_months", 12) or 12 for r in sorted_roles]
        median_tenure = float(np.median(durations)) if durations else 24.0

        # Monotonically increasing seniority + short median tenure = chaser
        is_escalating = all(
            seniority_levels[i] <= seniority_levels[i + 1]
            for i in range(len(seniority_levels) - 1)
        ) and seniority_levels[-1] > seniority_levels[0]

        title_chaser = is_escalating and median_tenure < 16
    else:
        title_chaser = False

    # ── Research-only flag ────────────────────────────────────────────────
    all_desc = " ".join(r.get("description", "") or "" for r in career_history).lower()
    research_count = _count_matches(all_desc, RESEARCH_WORDS)
    prod_count = _count_matches(all_desc, PRODUCTION_WORDS)
    research_only = research_count >= 3 and prod_count == 0

    # ── Aggregate trajectory score ────────────────────────────────────────
    penalty = 1.0
    if title_chaser:
        penalty *= 0.65
    if research_only:
        penalty *= 0.50

    trajectory_score = round(yoe_band * penalty, 4)

    return {
        "yoe_band_score": round(yoe_band, 4),
        "is_pure_consulting": is_pure_consulting,
        "consulting_penalty": round(consulting_penalty, 4),
        "title_chaser_flag": title_chaser,
        "research_only_flag": research_only,
        "trajectory_score": trajectory_score,
    }


# ── Location Fit ──────────────────────────────────────────────────────────────

def compute_location_fit(location: str, country: str, willing_to_relocate: bool) -> float:
    loc_lower = _lower(location)
    country_lower = _lower(country)

    # Check known cities
    for city, score in LOCATION_SCORES.items():
        if city in loc_lower:
            base = score
            break
    else:
        # India but unknown city
        if "india" in country_lower or "in" == country_lower:
            base = 0.55
        else:
            base = 0.35   # outside India

    # Relocation boost
    if willing_to_relocate and base < 1.0:
        base = min(base + 0.10, 1.0)

    return round(base, 4)


# ── Disqualifier Penalty ──────────────────────────────────────────────────────

def compute_disqualifier_penalty(
    career_history: list,
    current_title: str,
    summary: str,
    yoe: float,
    is_pure_consulting: bool,
    research_only: bool,
    title_chaser: bool,
) -> float:
    """
    Returns a multiplier 0.0–1.0 representing how disqualified a candidate is.
    0.0 = hard disqualifier (entire consulting career, no product exp)
    1.0 = no disqualifiers
    """
    penalty = 1.0

    # Hard: entire career at consulting firms
    if is_pure_consulting:
        return 0.0

    # Hard: research-only
    if research_only:
        penalty *= 0.10

    # Soft: title chaser
    if title_chaser:
        penalty *= 0.65

    # CV/Speech/Robotics only without NLP/IR
    all_text = " ".join(
        [r.get("description", "") or "" for r in career_history]
        + [summary or ""]
    ).lower()
    cv_terms = {"computer vision", "object detection", "image segmentation",
                "speech recognition", "asr", "tts", "robotics", "ros", "slam"}
    nlp_terms = {"nlp", "natural language", "ranking", "retrieval", "search",
                 "recommendation", "information retrieval", "text classification"}
    has_cv_only = _contains_any(all_text, cv_terms) and not _contains_any(all_text, nlp_terms)
    if has_cv_only:
        penalty *= 0.45

    # LangChain-only + very low YOE
    langchain_kws = {"langchain", "llamaindex", "llama index", "openai api", "chatgpt api"}
    if _contains_any(all_text, langchain_kws) and yoe < 2.0:
        penalty *= 0.35

    return round(max(0.0, min(penalty, 1.0)), 4)


# ── Main Feature Engineering Loop ────────────────────────────────────────────

def engineer_features(
    candidates_path: str,
    taxonomy_path: str,
    jd_path: str,
    out_path: str,
) -> None:
    print(f"Loading taxonomy from {taxonomy_path}")
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy: dict = json.load(f)

    print(f"Loading JD from {jd_path}")
    with open(jd_path, "r", encoding="utf-8") as f:
        jd: dict = yaml.safe_load(f)

    rows = []
    print(f"Engineering features from {candidates_path}")

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Candidates"):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = c.get("candidate_id", "")
            profile = c.get("profile", {})
            career = c.get("career_history", [])
            skills = c.get("skills", [])
            signals = c.get("redrob_signals", {})
            assessment_scores = signals.get("skill_assessment_scores", {}) or {}

            yoe = float(profile.get("years_of_experience", 0) or 0)
            current_title = profile.get("current_title", "") or ""
            summary = profile.get("summary", "") or ""
            location = profile.get("location", "") or ""
            country = profile.get("country", "") or ""
            willing_relocate = bool(signals.get("willing_to_relocate", False))

            # ── Skill evidence ────────────────────────────────────────────
            skill_feats = compute_skill_evidence(skills, taxonomy, assessment_scores)

            # ── Domain fit ────────────────────────────────────────────────
            domain_fit = compute_domain_fit(career, current_title, summary)

            # ── Trajectory ────────────────────────────────────────────────
            traj = compute_trajectory(yoe, career, current_title, jd)

            # ── Honeypot ──────────────────────────────────────────────────
            hp = check_honeypot(c)

            # ── Behavioral ────────────────────────────────────────────────
            beh = compute_behavioral_multiplier(signals)

            # ── Location ──────────────────────────────────────────────────
            loc = compute_location_fit(location, country, willing_relocate)

            # ── Disqualifier ──────────────────────────────────────────────
            disq = compute_disqualifier_penalty(
                career, current_title, summary, yoe,
                traj["is_pure_consulting"],
                traj["research_only_flag"],
                traj["title_chaser_flag"],
            )

            # ── Base fit (no embedding axis yet) ─────────────────────────
            w = jd.get("score_weights", {})
            sw = float(w.get("skill_evidence", 0.35))
            dw = float(w.get("domain_fit", 0.30))
            tw = float(w.get("trajectory", 0.15))
            lw = float(w.get("location_fit", 0.10))
            # embedding weight redistributed to skill+domain when embeddings absent
            ew = float(w.get("embedding_sim", 0.10))
            redistribute = ew  # will be filled in rank.py if embeddings available

            base_fit_no_emb = (
                (sw + redistribute * 0.5) * skill_feats["skill_evidence_score"]
                + (dw + redistribute * 0.5) * domain_fit
                + tw * traj["trajectory_score"]
                + lw * loc
            )
            base_fit_no_emb = round(min(max(base_fit_no_emb, 0.0), 1.0), 6)

            # Nice-to-have bonus (up to +0.05 on base fit)
            nth_bonus = round(skill_feats["nice_to_have_score"] * 0.05, 6)

            row = {
                "candidate_id": cid,
                "yoe": yoe,
                # skill evidence
                "skill_embeddings_retrieval": skill_feats.get("embeddings_retrieval", 0.0),
                "skill_vector_db": skill_feats.get("vector_db_hybrid_search", 0.0),
                "skill_python": skill_feats.get("python", 0.0),
                "skill_ranking_eval": skill_feats.get("ranking_evaluation", 0.0),
                "skill_evidence_score": skill_feats["skill_evidence_score"],
                "nice_to_have_score": skill_feats["nice_to_have_score"],
                # domain fit
                "domain_fit_score": domain_fit,
                # trajectory
                "yoe_band_score": traj["yoe_band_score"],
                "trajectory_score": traj["trajectory_score"],
                "is_pure_consulting": traj["is_pure_consulting"],
                "consulting_penalty": traj["consulting_penalty"],
                "title_chaser_flag": traj["title_chaser_flag"],
                "research_only_flag": traj["research_only_flag"],
                # honeypot
                "is_honeypot": hp["is_honeypot"],
                "honeypot_score": hp["honeypot_score"],
                # behavioral
                "behavioral_multiplier": beh,
                # location
                "location_fit": loc,
                # disqualifier
                "disqualifier_penalty": disq,
                # base fit
                "base_fit_no_emb": base_fit_no_emb,
                "nth_bonus": nth_bonus,
                # raw signals for reasoning
                "current_title": current_title,
                "location": location,
                "country": country,
                "notice_period_days": int(signals.get("notice_period_days", 30) or 30),
                "recruiter_response_rate": float(signals.get("recruiter_response_rate", 0.5) or 0.5),
                "open_to_work": bool(signals.get("open_to_work_flag", False)),
                "last_active_date": str(signals.get("last_active_date", "") or ""),
                "github_activity_score": float(signals.get("github_activity_score", -1) or -1),
                "willing_to_relocate": willing_relocate,
                "profile_completeness": float(signals.get("profile_completeness_score", 50) or 50),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nFeatures saved to {out_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Honeypots detected: {df['is_honeypot'].sum()}")
    print(f"  Pure consulting (hard disq): {df['is_pure_consulting'].sum()}")
    print(f"  Title chasers: {df['title_chaser_flag'].sum()}")
    print(f"  Research only: {df['research_only_flag'].sum()}")
    print(f"\nTop 10 by base_fit_no_emb:")
    top = df.nlargest(10, "base_fit_no_emb")[
        ["candidate_id", "current_title", "yoe", "skill_evidence_score",
         "domain_fit_score", "trajectory_score", "base_fit_no_emb"]
    ]
    print(top.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--taxonomy", default="artifacts/skill_taxonomy.json")
    parser.add_argument("--jd", default="job_requirements.yaml")
    parser.add_argument("--out", default="artifacts/features.parquet")
    args = parser.parse_args()
    engineer_features(args.candidates, args.taxonomy, args.jd, args.out)


if __name__ == "__main__":
    main()
