"""
behavioral_multiplier.py — Stage B/C behavioral signal combiner

Computes a multiplicative modifier (0.30 – 1.00) from platform activity signals.
Applied on top of base_fit so a low-behavior candidate is down-weighted but
NOT zeroed out — the JD says "down-weight appropriately," not disqualify.

Signals used:
  - recruiter_response_rate   (0-1)
  - last_active_date recency  (days since today)
  - open_to_work_flag         (bool)
  - interview_completion_rate (0-1)
  - notice_period_days        (graded penalty beyond 30d)
  - applications_submitted_30d (mild positive signal)
  - offer_acceptance_rate     (optional context)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

REFERENCE_DATE_STR = "2026-06-19"   # today — update if re-running later
REFERENCE_DATE = datetime.strptime(REFERENCE_DATE_STR, "%Y-%m-%d").date()

MULTIPLIER_MIN = 0.30
MULTIPLIER_MAX = 1.00


def _parse_date(d: Any) -> date | None:
    if d is None:
        return None
    if isinstance(d, date):
        return d
    try:
        return datetime.strptime(str(d), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _recency_score(last_active_date: Any) -> float:
    """
    Score 0-1 based on days since last active.
    0–14d  → 1.00
    15–30d → 0.90
    31–60d → 0.75
    61–90d → 0.55
    91–180d→ 0.35
    181d+  → 0.15
    """
    d = _parse_date(last_active_date)
    if d is None:
        return 0.40   # unknown → mild penalty

    days = (REFERENCE_DATE - d).days
    if days <= 0:
        return 1.00
    elif days <= 14:
        return 1.00
    elif days <= 30:
        return 0.90
    elif days <= 60:
        return 0.75
    elif days <= 90:
        return 0.55
    elif days <= 180:
        return 0.35
    else:
        return 0.15


def _notice_multiplier(notice_days: int | None) -> float:
    """
    Graded penalty for long notice periods.
    JD: sub-30d ideal; 30+ still in scope but bar gets higher.
    """
    if notice_days is None:
        return 0.85   # unknown → slight penalty

    nd = int(notice_days)
    if nd <= 30:
        return 1.00
    elif nd <= 60:
        return 0.90
    elif nd <= 90:
        return 0.78
    elif nd <= 120:
        return 0.68
    else:
        return 0.60


def compute_behavioral_multiplier(signals: dict[str, Any]) -> float:
    """
    Combines all behavioral signals into a single multiplier in [0.30, 1.00].

    Parameters
    ----------
    signals : dict
        The `redrob_signals` sub-object from a candidate record.

    Returns
    -------
    float in [MULTIPLIER_MIN, MULTIPLIER_MAX]
    """
    # ── Component scores (all 0-1) ──────────────────────────────────────────
    rr = float(signals.get("recruiter_response_rate", 0.5) or 0.5)
    rr_score = min(max(rr, 0.0), 1.0)

    recency_score = _recency_score(signals.get("last_active_date"))

    open_to_work = bool(signals.get("open_to_work_flag", False))
    otw_score = 1.00 if open_to_work else 0.75

    icr = float(signals.get("interview_completion_rate", 0.7) or 0.7)
    icr_score = min(max(icr, 0.0), 1.0)

    # Applications submitted in 30d — mild positive signal (active job seeker)
    apps = int(signals.get("applications_submitted_30d", 0) or 0)
    apps_score = min(apps / 5.0, 1.0)   # saturates at 5 applications

    # Saved by recruiters — market validation
    saved = int(signals.get("saved_by_recruiters_30d", 0) or 0)
    saved_score = min(saved / 3.0, 1.0)  # saturates at 3

    # Weighted behavioral base (before notice penalty)
    behavioral_base = (
        0.35 * rr_score
        + 0.25 * recency_score
        + 0.15 * otw_score
        + 0.10 * icr_score
        + 0.08 * apps_score
        + 0.07 * saved_score
    )

    # ── Notice period — multiplicative penalty on top ─────────────────────
    notice_days = signals.get("notice_period_days")
    notice_mult = _notice_multiplier(notice_days)

    raw = behavioral_base * notice_mult

    # ── Clamp to [MIN, MAX] ───────────────────────────────────────────────
    multiplier = max(MULTIPLIER_MIN, min(MULTIPLIER_MAX, raw))
    return round(multiplier, 4)
