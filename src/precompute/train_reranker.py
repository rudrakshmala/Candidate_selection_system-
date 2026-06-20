"""
train_reranker.py — Stage B precompute (optional, Step 4)

Trains a LightGBM lambdarank model using:
  - Features from artifacts/features.parquet
  - Weak-supervision labels from base_fit_no_emb (rule-based score)

This is the "learned re-ranker on top of rule-based scorer" pattern.
The model captures non-linear feature interactions the linear blend misses.

Output: artifacts/reranker_model.txt  (LightGBM text model)

Run once:
    pip install lightgbm
    python src/precompute/train_reranker.py \
        --features artifacts/features.parquet \
        --out artifacts/reranker_model.txt

At rank time: model.predict(feature_matrix) in <1 second for 100k rows.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not installed. Run: pip install lightgbm")
    raise

FEATURE_COLS = [
    "skill_embeddings_retrieval",
    "skill_vector_db",
    "skill_python",
    "skill_ranking_eval",
    "skill_evidence_score",
    "nice_to_have_score",
    "domain_fit_score",
    "yoe_band_score",
    "trajectory_score",
    "behavioral_multiplier",
    "location_fit",
    "disqualifier_penalty",
    "yoe",
    "profile_completeness",
    "github_activity_score",
    "recruiter_response_rate",
    "notice_period_days",
]


def train_reranker(features_path: str, out_path: str) -> None:
    print(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    print(f"  Shape: {df.shape}")

    # Filter out hard disqualifiers — don't train on garbage
    df_clean = df[
        (~df["is_honeypot"]) & (~df["is_pure_consulting"])
    ].copy()
    print(f"  After hard disqualifier filter: {len(df_clean)} rows")

    # Weak supervision: use base_fit_no_emb as label
    # Convert to discrete relevance scores 0-4 for lambdarank
    labels_cont = df_clean["base_fit_no_emb"].values
    labels_discrete = np.clip(
        np.floor(labels_cont * 5).astype(int), 0, 4
    )  # 0=irrelevant … 4=highly relevant

    X = df_clean[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    y = labels_discrete

    # Single group (all 100k as one "query") for lambdarank
    group = [len(X)]

    print(f"  Training LightGBM lambdarank on {len(X)} rows, {X.shape[1]} features")

    train_data = lgb.Dataset(X, label=y, group=group)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 100],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "n_estimators": 300,
        "feature_name": FEATURE_COLS,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(50)],
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(out_path)
    print(f"\nModel saved to {out_path}")

    # Feature importance
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=FEATURE_COLS,
    ).sort_values(ascending=False)
    print("\nFeature importance (gain):")
    print(importance.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="artifacts/features.parquet")
    parser.add_argument("--out", default="artifacts/reranker_model.txt")
    args = parser.parse_args()
    train_reranker(args.features, args.out)


if __name__ == "__main__":
    main()
