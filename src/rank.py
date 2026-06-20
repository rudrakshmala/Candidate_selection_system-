"""
rank.py — Stage C: THE TIMED ENTRY POINT

Hard constraints (must never be violated):
  - Runtime ≤ 5 minutes wall-clock
  - Memory ≤ 16 GB RAM
  - CPU only — no GPU
  - Zero network/API calls
  - Reads ONLY cached artifacts from Stage B (no recomputation)

Usage:
    python src/rank.py \
        --candidates data/candidates.jsonl \
        --features  artifacts/features.parquet \
        --out       submission.csv

Optional (if Stage B produced these):
    --embeddings  artifacts/candidate_embeddings.npy
    --emb-ids     artifacts/candidate_ids_emb.json
    --jd-emb      artifacts/jd_embedding.npy
    --model       artifacts/reranker_model.txt

Output format (validate_submission.py compatible):
    candidate_id,rank,score,reasoning
    CAND_XXXXXXX,1,0.9412,"..."
    ...  (exactly 100 data rows)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reasoning.compose_reasoning import compose_reasoning_batch

# ── Constants ─────────────────────────────────────────────────────────────────
TOP_N = 100
SCORE_WEIGHTS = {
    "skill_evidence": 0.30,
    "domain_fit":     0.35,
    "trajectory":     0.15,
    "embedding_sim":  0.10,
    "location_fit":   0.10,
}


# ── Score computation ─────────────────────────────────────────────────────────

def compute_final_scores(
    df: pd.DataFrame,
    emb_sims: np.ndarray | None,
) -> pd.Series:
    """
    Fully vectorised score computation — no Python loops over candidates.

    final_score = base_fit * behavioral_multiplier * (0 if honeypot) * disqualifier_penalty
    """
    w = SCORE_WEIGHTS
    emb_w = w["embedding_sim"]

    if emb_sims is not None:
        # Full formula with embedding axis
        base_fit = (
            w["skill_evidence"] * df["skill_evidence_score"]
            + w["domain_fit"]   * df["domain_fit_score"]
            + w["trajectory"]   * df["trajectory_score"]
            + emb_w             * pd.Series(emb_sims, index=df.index)
            + w["location_fit"] * df["location_fit"]
        )
    else:
        # Redistribute embedding weight to skill + domain
        half_e = emb_w / 2.0
        base_fit = (
            (w["skill_evidence"] + half_e) * df["skill_evidence_score"]
            + (w["domain_fit"] + half_e)   * df["domain_fit_score"]
            + w["trajectory"]              * df["trajectory_score"]
            + w["location_fit"]            * df["location_fit"]
        )

    # Nice-to-have bonus (up to +0.05)
    base_fit = base_fit + df["nth_bonus"].clip(0, 0.05)

    # Hard honeypot gate
    honeypot_mask = df["is_honeypot"].astype(float)   # 1.0 = honeypot
    base_fit = base_fit * (1.0 - honeypot_mask)

    # Hard consulting disqualifier (consulting_penalty already 0.0 for pure consulting)
    base_fit = base_fit * df["consulting_penalty"]

    # Disqualifier penalty (soft disqualifiers)
    base_fit = base_fit * df["disqualifier_penalty"]

    # Behavioral multiplier (0.30-1.00, never zeroes out)
    final_score = base_fit * df["behavioral_multiplier"]

    return final_score.clip(0.0, 1.0)


# ── Embedding similarity ──────────────────────────────────────────────────────

def load_embedding_similarities(
    features_df: pd.DataFrame,
    emb_path: str,
    ids_path: str,
    jd_emb_path: str,
) -> np.ndarray | None:
    """
    Load precomputed embeddings and return cosine similarity to JD embedding.
    Returns None if any artifact is missing (graceful fallback).
    """
    for p in [emb_path, ids_path, jd_emb_path]:
        if not Path(p).exists():
            print(f"  [emb] {p} not found — skipping embedding axis")
            return None

    print("  Loading embeddings...")
    candidate_embs = np.load(emb_path)          # (N, D) float32, pre-normalised
    jd_emb = np.load(jd_emb_path)               # (1, D)

    with open(ids_path, "r", encoding="utf-8") as f:
        emb_ids = json.load(f)                  # list of candidate_ids

    # Build id → row index map
    id_to_emb_idx = {cid: i for i, cid in enumerate(emb_ids)}

    # Align with features_df order
    idxs = [id_to_emb_idx.get(cid, -1) for cid in features_df["candidate_id"]]
    valid = [i for i in idxs if i >= 0]
    if len(valid) == 0:
        print("  [emb] No ID matches — skipping embedding axis")
        return None

    aligned_embs = np.zeros((len(features_df), candidate_embs.shape[1]), dtype=np.float32)
    for row_i, emb_i in enumerate(idxs):
        if emb_i >= 0:
            aligned_embs[row_i] = candidate_embs[emb_i]

    # Cosine similarity = dot product (already normalised)
    sims = aligned_embs @ jd_emb[0]  # shape (N,)
    print(f"  Embedding similarities computed. Shape: {sims.shape}")
    return sims


# ── LightGBM re-ranker ────────────────────────────────────────────────────────

LGBM_FEATURE_COLS = [
    "skill_embeddings_retrieval", "skill_vector_db", "skill_python",
    "skill_ranking_eval", "skill_evidence_score", "nice_to_have_score",
    "domain_fit_score", "yoe_band_score", "trajectory_score",
    "behavioral_multiplier", "location_fit", "disqualifier_penalty",
    "yoe", "profile_completeness", "github_activity_score",
    "recruiter_response_rate", "notice_period_days",
]


def apply_lgbm_reranker(
    df: pd.DataFrame,
    base_scores: pd.Series,
    model_path: str,
) -> pd.Series:
    """
    Apply LightGBM re-ranker if model exists.
    Blends rule-based and learned scores (70/30) so the rule-based signal
    remains dominant and interpretable.
    """
    if not Path(model_path).exists():
        print("  [lgbm] model not found — using rule-based scores only")
        return base_scores

    try:
        import lightgbm as lgb
    except ImportError:
        print("  [lgbm] lightgbm not installed — using rule-based scores only")
        return base_scores

    print("  Loading LightGBM model...")
    model = lgb.Booster(model_file=model_path)

    X = df[LGBM_FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    lgbm_scores = model.predict(X)

    # Normalise to [0, 1]
    mn, mx = lgbm_scores.min(), lgbm_scores.max()
    if mx > mn:
        lgbm_scores_norm = (lgbm_scores - mn) / (mx - mn)
    else:
        lgbm_scores_norm = lgbm_scores

    # Blend: 70% rule-based + 30% learned
    blended = 0.70 * base_scores.values + 0.30 * lgbm_scores_norm

    # Re-apply hard gates (honeypot/consulting — must not be overridden by learned model)
    blended[df["is_honeypot"].values] = 0.0
    blended[df["is_pure_consulting"].values] = 0.0

    print(f"  LightGBM blended scores applied (70/30 rule-based + learned)")
    return pd.Series(blended, index=df.index).clip(0.0, 1.0)


# ── Submission writer ─────────────────────────────────────────────────────────

def write_submission(
    top100_df: pd.DataFrame,
    final_scores: pd.Series,
    out_path: str,
) -> None:
    """Write exactly 100 data rows + header in the required format."""
    # Sort by score desc, ties broken by candidate_id asc
    result = top100_df.copy()
    result["_score"] = final_scores.loc[top100_df.index].values
    result = result.sort_values(
        by=["_score", "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)

    # Assign ranks 1-N (graceful degradation for small <100 pools in sandbox)
    result["rank"] = range(1, len(result) + 1)

    # Normalise scores to be strictly non-increasing
    # (small epsilon to handle floating point ties)
    scores = result["_score"].values.copy()
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            scores[i] = scores[i - 1]
    result["score"] = np.round(scores, 6)

    # Generate reasoning
    rows_for_reasoning = result.to_dict(orient="records")
    ranks_for_reasoning = list(result["rank"])
    reasoning_texts = compose_reasoning_batch(rows_for_reasoning, ranks_for_reasoning)
    result["reasoning"] = reasoning_texts

    # Validate row count before writing
    assert len(result) <= TOP_N, f"Expected up to {TOP_N} rows, got {len(result)}"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for _, row in result.iterrows():
            writer.writerow([
                row["candidate_id"],
                int(row["rank"]),
                f"{row['score']:.6f}",
                row["reasoning"],
            ])

    print(f"\nSubmission written to {out_path}")
    print(f"  Rows: {len(result)} (expected 100)")
    print(f"  Score range: {result['score'].min():.4f} – {result['score'].max():.4f}")

    # Quick honeypot spot-check on top-100
    honeypot_in_top = result[result["is_honeypot"] == True]
    if not honeypot_in_top.empty:
        print(f"\n  [WARN] WARNING: {len(honeypot_in_top)} honeypot(s) slipped into top-100!")
        print(honeypot_in_top[["candidate_id", "rank", "score", "honeypot_score"]].to_string())
    else:
        print("  [OK] Honeypot check: 0 honeypots in top-100")

    # Show top 10
    print("\nTop 10 candidates:")
    cols = ["candidate_id", "rank", "score", "current_title", "yoe",
            "skill_evidence_score", "domain_fit_score"]
    print(result[cols].head(10).to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rank candidates — timed Stage C entry point")
    parser.add_argument("--candidates", default="data/candidates.jsonl",
                        help="Path to candidates.jsonl (used only to validate IDs exist)")
    parser.add_argument("--features",   default="artifacts/features.parquet")
    parser.add_argument("--embeddings", default="artifacts/candidate_embeddings.npy")
    parser.add_argument("--emb-ids",    default="artifacts/candidate_ids_emb.json")
    parser.add_argument("--jd-emb",     default="artifacts/jd_embedding.npy")
    parser.add_argument("--model",      default="artifacts/reranker_model.txt")
    parser.add_argument("--out",        default="submission.csv")
    args = parser.parse_args()

    wall_start = time.time()
    print("=" * 60)
    print("RANK.PY — Stage C Timed Entry Point")
    print("=" * 60)

    # ── 1. Load pre-built feature matrix ─────────────────────────────────
    print(f"\n[1/6] Loading features from {args.features}")
    if not Path(args.features).exists():
        print(f"ERROR: {args.features} not found. Run Stage B first:")
        print("  python src/precompute/build_skill_taxonomy.py")
        print("  python src/precompute/engineer_features.py")
        sys.exit(1)

    df = pd.read_parquet(args.features)
    print(f"  Loaded {len(df)} candidates, {df.shape[1]} features")

    # ── 2. Validate candidate IDs exist in source file (fast check) ──────
    print(f"\n[2/6] Validating candidate IDs...")
    known_ids = set(df["candidate_id"].values)
    print(f"  {len(known_ids)} unique candidate IDs in features")

    # ── 3. Embedding similarities (optional) ─────────────────────────────
    print(f"\n[3/6] Loading embedding similarities (optional)...")
    emb_sims = load_embedding_similarities(
        df, args.embeddings, args.emb_ids, args.jd_emb
    )

    # ── 4. Compute base scores (fully vectorised) ─────────────────────────
    print(f"\n[4/6] Computing final scores (vectorised)...")
    base_scores = compute_final_scores(df, emb_sims)
    df["base_score"] = base_scores
    print(f"  Score stats: min={base_scores.min():.4f}, max={base_scores.max():.4f}, "
          f"mean={base_scores.mean():.4f}")

    # ── 5. Apply LightGBM re-ranker (optional) ───────────────────────────
    print(f"\n[5/6] Applying LightGBM re-ranker (optional)...")
    final_scores = apply_lgbm_reranker(df, base_scores, args.model)
    df["final_score"] = final_scores

    # ── 6. Select top-100, write submission ───────────────────────────────
    print(f"\n[6/6] Selecting top {TOP_N} and writing submission...")

    # Hard-exclude honeypots and hard disqualifiers before selecting top-100
    eligible = df[final_scores > 0.0].copy()
    print(f"  Eligible (score > 0): {len(eligible)} candidates")
    print(f"  Hard-excluded (score = 0): {len(df) - len(eligible)} candidates")

    if len(eligible) < TOP_N:
        print(f"  WARNING: Only {len(eligible)} eligible candidates — padding from excluded pool")
        # Fallback padding: strictly exclude honeypots from being padded back in
        excluded = df[(final_scores <= 0.0) & (~df["is_honeypot"])].nlargest(TOP_N - len(eligible), "base_score")
        if not excluded.empty:
            eligible = pd.concat([eligible, excluded])

    top100 = eligible.nlargest(TOP_N, "final_score")

    write_submission(top100, final_scores, args.out)

    wall_elapsed = time.time() - wall_start
    print(f"\n{'=' * 60}")
    print(f"Total wall time: {wall_elapsed:.1f}s ({wall_elapsed / 60:.2f} min)")
    if wall_elapsed > 300:
        print("  [WARN] WARNING: Exceeded 5-minute wall-clock limit!")
    else:
        print(f"  [OK] Within 5-minute budget ({300 - wall_elapsed:.0f}s remaining)")
    print("=" * 60)


if __name__ == "__main__":
    main()
