"""
compute_embeddings.py — Stage B precompute (optional, Step 3)

Encodes candidate headline + summary + top-5 skill names using
bge-small-en-v1.5 (CPU, fully local, ~130 MB model).
Also encodes the JD text.

Outputs:
  artifacts/candidate_embeddings.npy   shape (N, 384), float32
  artifacts/candidate_ids_emb.json     ordered list of candidate_ids matching rows
  artifacts/jd_embedding.npy           shape (1, 384), float32

Run once:
    pip install sentence-transformers torch
    python src/precompute/compute_embeddings.py \
        --candidates data/candidates.jsonl \
        --jd job_requirements.yaml \
        --out-dir artifacts/

Constraint: ZERO network calls during rank.py.
The model is downloaded once here; rank.py only does numpy cosine similarity.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 512   # safe for CPU RAM
MAX_SKILLS = 5


def build_candidate_text(candidate: dict) -> str:
    """Build a short embedding-friendly text string per candidate."""
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "") or ""
    summary = profile.get("summary", "") or ""
    # Truncate summary to first 200 chars to keep text short
    summary_short = summary[:200]

    skills = candidate.get("skills", [])
    top_skills = ", ".join(
        sk.get("name", "") for sk in skills[:MAX_SKILLS] if sk.get("name")
    )

    current_title = profile.get("current_title", "") or ""

    parts = [p for p in [current_title, headline, summary_short, top_skills] if p]
    return " | ".join(parts)


def build_jd_text() -> str:
    return (
        "Senior AI Engineer with production experience in embeddings-based retrieval, "
        "vector databases, hybrid search, ranking evaluation (NDCG, MRR, MAP, A/B testing). "
        "Skills: sentence-transformers, BGE, E5, Pinecone, Weaviate, Qdrant, Milvus, "
        "FAISS, OpenSearch, Elasticsearch, Python, LLM fine-tuning, LightGBM, "
        "recommendation systems, search ranking, RAG, semantic search."
    )


def compute_embeddings(candidates_path: str, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Read all candidate texts
    print("Reading candidates...")
    candidate_ids: list[str] = []
    texts: list[str] = []

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading"):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidate_ids.append(c.get("candidate_id", ""))
            texts.append(build_candidate_text(c))

    print(f"  {len(texts)} candidates loaded")

    # Encode in batches
    print("Encoding candidates (CPU — this will take a while)...")
    all_embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine sim = dot product after normalisation
        convert_to_numpy=True,
    )

    emb_path = out_path / "candidate_embeddings.npy"
    ids_path = out_path / "candidate_ids_emb.json"

    np.save(str(emb_path), all_embeddings.astype(np.float32))
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)

    print(f"Embeddings saved: {emb_path}  shape={all_embeddings.shape}")

    # Encode JD
    jd_text = build_jd_text()
    jd_emb = model.encode(
        [jd_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    jd_path = out_path / "jd_embedding.npy"
    np.save(str(jd_path), jd_emb.astype(np.float32))
    print(f"JD embedding saved: {jd_path}  shape={jd_emb.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()
    compute_embeddings(args.candidates, args.out_dir)


if __name__ == "__main__":
    main()
