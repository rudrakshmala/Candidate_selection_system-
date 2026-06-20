# Redrob "India Runs" — Candidate Ranking System

**Hackathon:** Redrob Intelligent Candidate Discovery & Ranking Challenge  
**Track:** India Runs  
**Author:** [Your registered participant ID]

---

## TL;DR — Exact Reproduce Command

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Stage B: Build taxonomy (once, ~2-3 min)
python src/precompute/build_skill_taxonomy.py \
    --candidates data/candidates.jsonl \
    --jd job_requirements.yaml \
    --out artifacts/skill_taxonomy.json

# 3. Stage B: Engineer features (once, ~5-8 min for 100k)
python src/precompute/engineer_features.py \
    --candidates data/candidates.jsonl \
    --taxonomy   artifacts/skill_taxonomy.json \
    --jd         job_requirements.yaml \
    --out        artifacts/features.parquet

# 4. Stage C: Rank (TIMED — must complete in ≤5 min)
python src/rank.py \
    --candidates data/candidates.jsonl \
    --features   artifacts/features.parquet \
    --out        submission.csv

# 5. Validate
python data/validate_submission.py submission.csv
```

### Optional Stage B steps (enhance quality)

```bash
# Compute embeddings (requires: pip install sentence-transformers torch)
python src/precompute/compute_embeddings.py \
    --candidates data/candidates.jsonl \
    --out-dir    artifacts/

# Train LightGBM re-ranker (requires: pip install lightgbm)
python src/precompute/train_reranker.py \
    --features artifacts/features.parquet \
    --out      artifacts/reranker_model.txt
```

---

## Architecture

```
Stage A (one-time, by hand)
  └── job_requirements.yaml        ← JD encoded by human, every weight defensible

Stage B (offline precompute, runs once)
  ├── build_skill_taxonomy.py      ← extracts real skill vocab from 100k candidates
  ├── engineer_features.py         ← vectorised feature matrix → features.parquet
  ├── compute_embeddings.py        ← optional: bge-small-en-v1.5 CPU embeddings
  └── train_reranker.py            ← optional: LightGBM lambdarank

Stage C (TIMED ≤5 min, CPU only, zero network)
  └── rank.py                      ← loads artifacts, scores, selects top-100, writes CSV
```

### Why rank.py never calls an LLM or makes network requests

Production recruiting systems cannot afford LLM-per-candidate at scale. All
LLM-assisted reasoning was moved to one-time offline Stage A/B work. The hot
path is a fast, deterministic, fully cached scorer — vectorised pandas/numpy
operations on a pre-built feature matrix, completing in seconds, not minutes.

---

## Scoring Formula

```
base_fit = (
    0.35 * skill_evidence_score      # Skill Verification Index
  + 0.30 * domain_fit_score          # career_history description evidence (JD's decisive signal)
  + 0.15 * trajectory_score          # YOE band, consulting flag, title-chaser
  + 0.10 * embedding_similarity      # optional tie-breaker (0 if not computed)
  + 0.10 * location_fit
)

final_score = base_fit
            * behavioral_multiplier   # 0.30–1.00, never zeroes a strong fit
            * (0 if honeypot_flagged else 1)
            * disqualifier_penalty    # 0.0 for hard disq, 0.1–0.9 for soft
```

### Why domain_fit_score gets 0.30 weight

The JD explicitly states: *"A Tier 5 candidate may not use the words 'RAG' or
'Pinecone'... but if their career history shows they built a recommendation
system at a product company, they're a fit."* The domain_fit_score reads
career_history[].description text for production evidence ("shipped",
"deployed", "at scale") combined with domain terms (search, ranking, retrieval,
embeddings). This is the mechanism for catching these candidates.

---

## Honeypot Detection

Candidates with **any** of these flags get `final_score = 0`:

| Pattern | Condition |
|---|---|
| H1 | `skill.duration_months > years_of_experience * 12 + 6` |
| H2 | `salary.min > salary.max` |
| H3 | proficiency=expert AND duration≤3mo AND endorsements=0 AND no assessment score |
| H4 | sum(career_history.duration_months) > years_of_experience × 12 × 2.0 |
| H5 | ≥3 expert skills with duration=0 |
| H6 | YOE≤0.5 but advanced/expert skills with >12mo duration |

Run tests: `python tests/test_honeypot_detection.py`

---

## Anti-Keyword-Stuffing Mechanism

The `skill_evidence_score` is **not** a simple keyword count. For each skill:

```
if skill has platform assessment score:
    score = 0.20×proficiency + 0.25×duration + 0.15×endorsements + 0.40×assessment
else:
    score = 0.35×proficiency + 0.45×duration + 0.20×endorsements
```

A claimed "expert" skill with 0 months duration, 0 endorsements, and no
assessment score contributes near-zero, even if it's the literal word "Pinecone".

---

## Constraints Compliance

| Constraint | Limit | Status |
|---|---|---|
| Runtime (rank.py only) | ≤ 5 min | ✅ (vectorised, sub-minute on features.parquet) |
| Memory | ≤ 16 GB | ✅ (parquet loads ~200 MB for 100k candidates) |
| Compute | CPU only | ✅ (no GPU dependency in rank.py) |
| Network | Zero calls | ✅ (all model/data loaded from disk) |
| Disk (intermediate) | ≤ 5 GB | ✅ (features.parquet ~50 MB, embeddings ~150 MB) |

---

## Pre-computation Metadata

| Step | Required? | Estimated Time |
|---|---|---|
| build_skill_taxonomy.py | Yes | ~2-3 min |
| engineer_features.py | Yes | ~5-8 min |
| compute_embeddings.py | Optional | ~20-30 min CPU |
| train_reranker.py | Optional | ~3-5 min |

---

## Repo Structure

```
redrob-ranker/
├── README.md
├── submission_metadata.yaml
├── job_requirements.yaml          # Stage A — hand-encoded, human-reviewed
├── requirements.txt
├── .gitignore
├── data/                          # gitignored — add candidates.jsonl here
├── artifacts/                     # Stage B outputs
│   ├── skill_taxonomy.json
│   ├── features.parquet
│   ├── candidate_embeddings.npy   # optional
│   ├── candidate_ids_emb.json     # optional
│   ├── jd_embedding.npy           # optional
│   └── reranker_model.txt         # optional
├── src/
│   ├── precompute/
│   │   ├── build_skill_taxonomy.py
│   │   ├── engineer_features.py
│   │   ├── compute_embeddings.py
│   │   └── train_reranker.py
│   ├── scoring/
│   │   ├── base_fit.py
│   │   ├── honeypot_filter.py
│   │   └── behavioral_multiplier.py
│   ├── reasoning/
│   │   └── compose_reasoning.py
│   └── rank.py                    ← THE TIMED ENTRY POINT
├── data/validate_submission.py
└── tests/
    └── test_honeypot_detection.py
```
