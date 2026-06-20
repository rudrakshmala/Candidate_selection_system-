"""
base_fit.py — standalone base fit scorer (thin wrapper, used for clarity)

Wraps the vectorised score computation logic so it can be imported
and unit-tested independently from rank.py.
"""

import numpy as np
import pandas as pd

SCORE_WEIGHTS = {
    "skill_evidence": 0.35,
    "domain_fit":     0.30,
    "trajectory":     0.15,
    "embedding_sim":  0.10,
    "location_fit":   0.10,
}


def compute_base_fit(
    df: pd.DataFrame,
    emb_sims: np.ndarray | None = None,
) -> pd.Series:
    """
    Pure base fit before behavioral multiplier and disqualifier penalty.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix from features.parquet
    emb_sims : np.ndarray or None
        Precomputed cosine similarities to JD embedding (optional)

    Returns
    -------
    pd.Series of float in [0, 1]
    """
    w = SCORE_WEIGHTS
    emb_w = w["embedding_sim"]

    if emb_sims is not None:
        base = (
            w["skill_evidence"] * df["skill_evidence_score"]
            + w["domain_fit"]   * df["domain_fit_score"]
            + w["trajectory"]   * df["trajectory_score"]
            + emb_w             * pd.Series(emb_sims, index=df.index)
            + w["location_fit"] * df["location_fit"]
        )
    else:
        half_e = emb_w / 2.0
        base = (
            (w["skill_evidence"] + half_e) * df["skill_evidence_score"]
            + (w["domain_fit"] + half_e)   * df["domain_fit_score"]
            + w["trajectory"]              * df["trajectory_score"]
            + w["location_fit"]            * df["location_fit"]
        )

    # Nice-to-have bonus (capped at +0.05)
    base = base + df["nth_bonus"].clip(0, 0.05)

    return base.clip(0.0, 1.0)
