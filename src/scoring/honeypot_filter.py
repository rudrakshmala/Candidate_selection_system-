"""
honeypot_filter.py — isolated, unit-tested honeypot/consistency filter

Detects candidates with logically impossible or highly suspicious profiles.
Called from engineer_features.py and rank.py.

Honeypot patterns (from spec + data inspection):
  H1: skill duration_months > years_of_experience * 12 + 6
  H2: expected_salary_range min > max
  H3: proficiency='expert' AND duration_months <= 3 AND endorsements == 0
      AND no skill_assessment_scores entry for that skill
  H4: sum(career_history.duration_months) >> years_of_experience * 12
      (> 1.4x = moderate flag, > 2.0x = hard flag)
  H5: Multiple skills with duration_months=0 but proficiency='expert'
  H6: years_of_experience=0 but claims advanced/expert skills with long duration

Returns:
  is_honeypot: bool  — True = hard exclude from top-100 pool
  honeypot_score: float  — 0.0 (clean) → 1.0 (definitely fake)
  flags: list[str]  — human-readable reasons
"""

from __future__ import annotations

from typing import Any


CAREER_TOTAL_RATIO_HARD = 2.0   # career_total > YOE*12 * this  → hard flag
CAREER_TOTAL_RATIO_SOFT = 1.4   # career_total > YOE*12 * this  → soft flag
SKILL_DURATION_BUFFER = 6       # months buffer for rounding (spec says +6)
EXPERT_SHORT_MONTHS = 3         # 'expert' with <= this months is suspicious
HARD_HONEYPOT_THRESHOLD = 0.5   # honeypot_score above this → is_honeypot=True


def check_honeypot(candidate: dict[str, Any]) -> dict[str, Any]:
    """
    Returns:
        {
            "is_honeypot": bool,
            "honeypot_score": float,  # 0-1
            "flags": list[str],
        }
    """
    flags: list[str] = []
    score_components: list[float] = []

    yoe: float = candidate.get("profile", {}).get("years_of_experience", 0.0)
    yoe_months: float = yoe * 12.0
    skills: list[dict] = candidate.get("skills", [])
    career: list[dict] = candidate.get("career_history", [])
    signals: dict = candidate.get("redrob_signals", {})
    assessment_scores: dict = signals.get("skill_assessment_scores", {})
    salary: dict = signals.get("expected_salary_range_inr_lpa", {})

    # ── H1: Skill duration exceeds total experience ────────────────────────
    # Relaxed: skill duration more than 2x total experience + 24mo buffer
    h1_extreme = False
    h1_soft = False
    for sk in skills:
        skill_name = sk.get("name", "?")
        sk_dur = sk.get("duration_months", 0) or 0
        if sk_dur > yoe_months * 2 + 24:
            flags.append(f"H1-extreme: skill '{skill_name}' duration={sk_dur}mo > 2x YOE ({yoe_months}mo) + 24")
            h1_extreme = True
        elif sk_dur > yoe_months + 48:
            flags.append(f"H1-soft: skill '{skill_name}' duration={sk_dur}mo > YOE ({yoe_months}mo) + 48")
            h1_soft = True
            
    if h1_extreme:
        score_components.append(0.8)
    elif h1_soft:
        score_components.append(0.25)

    # ── H2: Salary range inverted (min > max) ─────────────────────────────
    # Common data entry error (19% of candidates), so reduced to very soft penalty
    sal_min = salary.get("min", 0.0)
    sal_max = salary.get("max", 0.0)
    if sal_min > 0 and sal_max > 0 and sal_min > sal_max:
        flags.append(
            f"H2: salary min={sal_min} > max={sal_max} (inverted range)"
        )
        score_components.append(0.15)

    # ── H3: 'expert' with near-zero evidence and no assessment score ───────
    h3_count = 0
    for sk in skills:
        skill_name = sk.get("name", "?")
        proficiency = sk.get("proficiency", "").lower()
        sk_dur = sk.get("duration_months", 0) or 0
        endorsements = sk.get("endorsements", 0) or 0
        has_assessment = skill_name in assessment_scores

        if (
            proficiency == "expert"
            and sk_dur <= EXPERT_SHORT_MONTHS
            and endorsements == 0
            and not has_assessment
        ):
            flags.append(
                f"H3: '{skill_name}' expert proficiency but "
                f"dur={sk_dur}mo, endorsements=0, no assessment"
            )
            h3_count += 1

    if h3_count > 0:
        score_components.append(min(0.4 + h3_count * 0.15, 0.85))

    # ── H4: Career history total >> years_of_experience ───────────────────
    if career and yoe_months > 0:
        career_total = sum(r.get("duration_months", 0) or 0 for r in career)
        ratio = career_total / yoe_months if yoe_months > 0 else 1.0
        if ratio > CAREER_TOTAL_RATIO_HARD:
            flags.append(
                f"H4: career_total={career_total}mo vs yoe_months={yoe_months:.0f} "
                f"(ratio={ratio:.2f}, hard threshold={CAREER_TOTAL_RATIO_HARD})"
            )
            score_components.append(0.85)
        elif ratio > CAREER_TOTAL_RATIO_SOFT:
            flags.append(
                f"H4-soft: career_total={career_total}mo vs yoe_months={yoe_months:.0f} "
                f"(ratio={ratio:.2f}, soft threshold={CAREER_TOTAL_RATIO_SOFT})"
            )
            score_components.append(0.35)

    # ── H5: Multiple expert skills with zero duration ──────────────────────
    zero_dur_experts = [
        sk.get("name", "?")
        for sk in skills
        if sk.get("proficiency", "").lower() == "expert"
        and (sk.get("duration_months", 0) or 0) == 0
    ]
    if len(zero_dur_experts) >= 3:
        flags.append(
            f"H5: {len(zero_dur_experts)} expert skills with duration=0: "
            f"{zero_dur_experts[:5]}"
        )
        score_components.append(0.60)

    # ── H6: Zero YOE but advanced/expert long-duration skills ─────────────
    if yoe <= 0.5:
        long_advanced = [
            sk.get("name", "?")
            for sk in skills
            if sk.get("proficiency", "").lower() in ("advanced", "expert")
            and (sk.get("duration_months", 0) or 0) > 12
        ]
        if long_advanced:
            flags.append(
                f"H6: YOE={yoe} but {len(long_advanced)} advanced/expert skills "
                f"with >12mo duration: {long_advanced[:5]}"
            )
            score_components.append(0.70)

    # ── Aggregate ──────────────────────────────────────────────────────────
    if not score_components:
        honeypot_score = 0.0
    else:
        # Max of components, boosted by count
        honeypot_score = min(
            max(score_components) + (len(score_components) - 1) * 0.05, 1.0
        )

    is_honeypot = honeypot_score >= HARD_HONEYPOT_THRESHOLD

    return {
        "is_honeypot": is_honeypot,
        "honeypot_score": round(honeypot_score, 4),
        "flags": flags,
    }


def batch_check_honeypots(candidates: list[dict[str, Any]]) -> list[dict]:
    """Process a list of candidates, return parallel list of honeypot results."""
    return [check_honeypot(c) for c in candidates]
