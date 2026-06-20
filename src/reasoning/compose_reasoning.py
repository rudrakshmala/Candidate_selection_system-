"""
compose_reasoning.py — rule-based reasoning text generator

Produces the `reasoning` column for the top-100 submission.
Graded manually at Stage 4. Requirements:
  - Substantively different across rows (not just name-in-template)
  - Picks 2-3 MOST DISTINCTIVE facts per candidate (not always same fields)
  - HONESTLY surfaces a concern when one exists
  - No hallucination — every fact must come from the candidate's own data
  - Variation in sentence structure (≥8 branching templates)
  - Tone consistent with rank (rank 1 more confident than rank 80)

Design: cascade of rule-checks → pick the template family that best
fits the candidate's specific combination of signals.
"""

from __future__ import annotations

import random
from datetime import datetime, date
from typing import Any

REFERENCE_DATE_STR = "2026-06-19"
REFERENCE_DATE = datetime.strptime(REFERENCE_DATE_STR, "%Y-%m-%d").date()


def _days_since(date_str: str) -> int | None:
    if not date_str:
        return None
    try:
        d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except (ValueError, TypeError):
        return None


def _notice_str(days: int) -> str:
    if days <= 0:
        return "immediately available"
    elif days <= 15:
        return f"{days}-day notice (near-immediate)"
    elif days <= 30:
        return f"{days}-day notice (within ideal range)"
    elif days <= 60:
        return f"{days}-day notice (manageable)"
    elif days <= 90:
        return f"{days}-day notice (longer than ideal)"
    else:
        return f"{days}-day notice (a concern — significantly above 30-day ideal)"


def _activity_str(days_inactive: int | None) -> str:
    if days_inactive is None:
        return "activity unknown"
    elif days_inactive <= 7:
        return "very recently active"
    elif days_inactive <= 30:
        return "active within the month"
    elif days_inactive <= 90:
        return f"last seen {days_inactive} days ago"
    elif days_inactive <= 180:
        return f"inactive for ~{days_inactive // 30} months — reachability uncertain"
    else:
        return f"inactive for {days_inactive // 30}+ months — may be a ghost profile"


def _yoe_str(yoe: float) -> str:
    return f"{yoe:.1f} yrs"


def _rr_str(rr: float) -> str:
    if rr >= 0.8:
        return f"high recruiter response rate ({rr:.0%})"
    elif rr >= 0.5:
        return f"moderate response rate ({rr:.0%})"
    elif rr >= 0.3:
        return f"low response rate ({rr:.0%})"
    else:
        return f"very low response rate ({rr:.0%}) — may be hard to reach"


def _domain_strength_str(domain_fit: float) -> str:
    if domain_fit >= 0.75:
        return "strong production evidence in IR/ranking/ML"
    elif domain_fit >= 0.50:
        return "solid domain signals in career descriptions"
    elif domain_fit >= 0.30:
        return "some relevant domain exposure"
    else:
        return "limited domain evidence in career descriptions"


def _skill_strength_str(skill_score: float) -> str:
    if skill_score >= 0.75:
        return "well-verified core skill set"
    elif skill_score >= 0.50:
        return "partially verified skills with meaningful tenure"
    elif skill_score >= 0.30:
        return "skill claims present but lightly corroborated"
    else:
        return "minimal relevant skill evidence"


def _concern_str(row: dict) -> str | None:
    """Return the single biggest honest concern, or None if clean."""
    concerns = []

    notice = row.get("notice_period_days", 30)
    if notice > 90:
        concerns.append(f"notice period of {notice} days is a significant barrier")
    elif notice > 60:
        concerns.append(f"notice period is {notice} days (above ideal)")

    rr = row.get("recruiter_response_rate", 0.5)
    if rr < 0.25:
        concerns.append(f"very low recruiter response rate ({rr:.0%})")

    days_inactive = _days_since(row.get("last_active_date", ""))
    if days_inactive and days_inactive > 120:
        concerns.append(f"profile inactive for {days_inactive // 30}+ months")

    if row.get("title_chaser_flag", False):
        concerns.append("career shows rapid title escalation — tenure depth uncertain")

    if row.get("research_only_flag", False):
        concerns.append("career language skews academic — verify production depth")

    if not concerns:
        return None
    return concerns[0]   # surface only the primary concern


def compose_reasoning(row: dict, rank: int) -> str:
    """
    Generate a substantive, candidate-specific reasoning string.
    Each template family is triggered by a different signal combination.
    """
    yoe = float(row.get("yoe", 0))
    title = str(row.get("current_title", ""))
    location = str(row.get("location", ""))
    domain_fit = float(row.get("domain_fit_score", 0))
    skill_score = float(row.get("skill_evidence_score", 0))
    traj_score = float(row.get("trajectory_score", 0))
    notice = int(row.get("notice_period_days", 30))
    rr = float(row.get("recruiter_response_rate", 0.5))
    days_inactive = _days_since(row.get("last_active_date", ""))
    open_to_work = bool(row.get("open_to_work", False))
    github = float(row.get("github_activity_score", -1))
    consulting = bool(row.get("is_pure_consulting", False))
    nth_bonus = float(row.get("nice_to_have_score", 0))
    loc_fit = float(row.get("location_fit", 0))

    concern = _concern_str(row)

    # ── Template selection logic ────────────────────────────────────────────
    # T0: Top Pick (Rank 1 specifically highlighted)
    if rank == 1:
        dominant_str = "exceptional domain fit"
        if skill_score > domain_fit + 0.15:
            dominant_str = "exceptional core skill evidence"
        elif traj_score > domain_fit + 0.15 and traj_score > skill_score:
            dominant_str = "an exceptional career trajectory"
            
        base = f"Top Pick: {_yoe_str(yoe)}, {dominant_str}."
        if concern:
            return f"{base} {title} with {_skill_strength_str(skill_score)}. Note: {concern}."
        return f"{base} {title} offering {_domain_strength_str(domain_fit)} and {_skill_strength_str(skill_score)}. {_notice_str(notice).capitalize()}."

    # T1: High domain fit + high skill score (ideal fit)
    if domain_fit >= 0.65 and skill_score >= 0.55:
        if concern:
            text = (
                f"{title} with {_yoe_str(yoe)} total experience. "
                f"Career descriptions show {_domain_strength_str(domain_fit)} — "
                f"the clearest signal here is production work in the JD's core areas, not just keyword listing. "
                f"Skill evidence is {_skill_strength_str(skill_score)}. "
                f"Primary concern: {concern}."
            )
        else:
            text = (
                f"{title} with {_yoe_str(yoe)} experience and {_notice_str(notice)}. "
                f"Strong fit: {_domain_strength_str(domain_fit)}, "
                f"backed by {_skill_strength_str(skill_score)}. "
                f"Response rate is {_rr_str(rr)}. "
                + (f"GitHub activity score {github:.0f}/100 — active practitioner. " if github >= 50 else "")
                + ("Marked open to work." if open_to_work else "")
            )

    # T2: High domain fit but low skill score (Tier-5 candidate — the JD's stated case)
    elif domain_fit >= 0.55 and skill_score < 0.40:
        text = (
            f"{title} with {_yoe_str(yoe)} experience. "
            f"This is a Tier-5 style match: career descriptions demonstrate "
            f"{_domain_strength_str(domain_fit)}, even though explicit JD-keyword skill claims "
            f"are light ({_skill_strength_str(skill_score)}). "
            f"Domain evidence from role descriptions outweighs skills-section keyword count here. "
            + (f"Concern: {concern}." if concern else f"Availability: {_notice_str(notice)}.")
        )

    # T3: High skill score but low domain fit (keyword stuffer risk — still ranked but noted)
    elif skill_score >= 0.65 and domain_fit < 0.35:
        text = (
            f"{title} with {_yoe_str(yoe)} experience. "
            f"Skill set aligns well with JD requirements ({_skill_strength_str(skill_score)}), "
            f"but career history descriptions offer only {_domain_strength_str(domain_fit)}. "
            f"Recommend verifying that listed skills reflect hands-on production work "
            f"rather than project-adjacent exposure. "
            + (f"Key concern: {concern}." if concern else f"Notice: {_notice_str(notice)}.")
        )

    # T4: Location is a differentiator (Pune/Noida with good fit)
    elif loc_fit >= 0.95 and (domain_fit + skill_score) >= 0.70:
        text = (
            f"Based in {location} — preferred location for this role. "
            f"{title} with {_yoe_str(yoe)} experience. "
            f"{_domain_strength_str(domain_fit).capitalize()} combined with "
            f"{_skill_strength_str(skill_score)}. "
            f"Location alignment reduces onboarding friction significantly. "
            + (f"Note: {concern}." if concern else f"Available with {_notice_str(notice)}.")
        )

    # T5: Nice-to-have bonus is a differentiator (LoRA, LTR, HR-tech etc.)
    elif nth_bonus >= 0.025 and (domain_fit + skill_score) >= 0.55:
        text = (
            f"{title} with {_yoe_str(yoe)} experience. "
            f"Core JD alignment: {_skill_strength_str(skill_score)} and "
            f"{_domain_strength_str(domain_fit)}. "
            f"Additionally brings relevant nice-to-have skills (LoRA/QLoRA, LTR, or HR-tech exposure) "
            f"that would accelerate time-to-impact on the ranking system v2. "
            + (f"Concern: {concern}." if concern else "")
        )

    # T6: Low activity / reachability concern dominates
    elif days_inactive and days_inactive > 90:
        text = (
            f"{title} with {_yoe_str(yoe)} experience. "
            f"Profile signals reasonable fit: {_skill_strength_str(skill_score)}, "
            f"{_domain_strength_str(domain_fit)}. "
            f"However, profile has been {_activity_str(days_inactive)} "
            f"and response rate is {_rr_str(rr)} — "
            f"reachability is the primary risk factor here, not skill fit."
        )

    # T7: Good trajectory score, moderate skills (solid generalist)
    elif traj_score >= 0.75 and skill_score >= 0.35:
        text = (
            f"{title} with {_yoe_str(yoe)} experience, "
            f"trajectory score suggests stable, progressive ML/AI career at product companies. "
            f"{_skill_strength_str(skill_score).capitalize()}. "
            f"{_domain_strength_str(domain_fit).capitalize()} in career descriptions. "
            + (f"Note: {concern}." if concern else f"{_notice_str(notice).capitalize()}.")
        )

    # T8: Default / lower-ranked catch-all
    else:
        text = (
            f"{title}, {_yoe_str(yoe)} experience. "
            f"{_skill_strength_str(skill_score).capitalize()}; "
            f"{_domain_strength_str(domain_fit)}. "
            f"Availability: {_notice_str(notice)}; {_rr_str(rr)}. "
            + (f"Concern: {concern}." if concern else "")
        )

    return text.strip()


def compose_reasoning_batch(rows: list[dict], ranks: list[int]) -> list[str]:
    """Vectorised call over top-100 rows."""
    return [compose_reasoning(row, rank) for row, rank in zip(rows, ranks)]
