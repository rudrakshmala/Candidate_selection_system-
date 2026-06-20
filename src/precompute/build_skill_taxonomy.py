"""
build_skill_taxonomy.py — Stage B precompute (Step 1)

Reads the full candidates.jsonl, extracts every unique skill name,
then maps each skill to one or more JD categories using keyword matching
plus manual overrides.

Output: artifacts/skill_taxonomy.json
  {
    "skill_name_lower": ["must_have_category", ...] | ["nice_to_have_category", ...],
    ...
  }

Run once:
    python src/precompute/build_skill_taxonomy.py \
        --candidates data/candidates.jsonl \
        --jd job_requirements.yaml \
        --out artifacts/skill_taxonomy.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm


def load_jd(jd_path: str) -> dict:
    with open(jd_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_keyword_to_category(jd: dict) -> dict[str, list[str]]:
    """Build a flat keyword → [category_name, ...] lookup from JD yaml."""
    mapping: dict[str, list[str]] = {}

    for cat_name, cat_cfg in jd.get("must_have_categories", {}).items():
        for kw in cat_cfg.get("keywords", []):
            kw_lower = kw.lower().strip()
            mapping.setdefault(kw_lower, [])
            if cat_name not in mapping[kw_lower]:
                mapping[kw_lower].append(cat_name)

    for cat_name, cat_cfg in jd.get("nice_to_have_categories", {}).items():
        for kw in cat_cfg.get("keywords", []):
            kw_lower = kw.lower().strip()
            mapping.setdefault(kw_lower, [])
            if cat_name not in mapping[kw_lower]:
                mapping[kw_lower].append(cat_name)

    return mapping


def match_skill_to_categories(
    skill_lower: str, kw_map: dict[str, list[str]]
) -> list[str]:
    """
    Match a single skill name to JD categories.
    1. Exact match
    2. Contains-match (skill contains keyword or keyword contains skill)
    """
    cats: set[str] = set()

    # Exact match
    if skill_lower in kw_map:
        cats.update(kw_map[skill_lower])
        return list(cats)

    # Contains match — skill name contains a keyword
    for kw, cat_list in kw_map.items():
        # keyword inside skill name  e.g. "faiss index" contains "faiss"
        if kw in skill_lower:
            cats.update(cat_list)
        # skill inside keyword  e.g. "transformer" inside "sentence transformers"
        # Minimum length 4 to prevent "rag" matching "mean aveRAGe precision"
        elif skill_lower in kw and len(skill_lower) > 3:
            cats.update(cat_list)

    return list(cats)


def extract_all_skills(candidates_path: str) -> Counter:
    """Stream candidates.jsonl, collect all skill names with frequency."""
    counter: Counter = Counter()
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Extracting skills"):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            for skill in candidate.get("skills", []):
                name = skill.get("name", "").strip()
                if name:
                    counter[name] += 1
    return counter


def build_taxonomy(
    candidates_path: str, jd_path: str, out_path: str
) -> None:
    print(f"Loading JD from {jd_path}")
    jd = load_jd(jd_path)
    kw_map = build_keyword_to_category(jd)
    print(f"  {len(kw_map)} keywords loaded from JD")

    print(f"Scanning candidates from {candidates_path}")
    skill_counter = extract_all_skills(candidates_path)
    print(f"  {len(skill_counter)} unique skill names found")

    taxonomy: dict[str, dict] = {}
    unmapped: list[str] = []

    for skill_name, freq in skill_counter.most_common():
        skill_lower = re.sub(r"\s+", " ", skill_name.lower().strip())
        cats = match_skill_to_categories(skill_lower, kw_map)
        taxonomy[skill_name] = {
            "normalized": skill_lower,
            "categories": cats,
            "frequency": freq,
        }
        if not cats:
            unmapped.append(skill_name)

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    print(f"\nTaxonomy saved to {out_path}")
    print(f"  Total skills: {len(taxonomy)}")
    print(f"  Mapped to JD categories: {len(taxonomy) - len(unmapped)}")
    print(f"  Unmapped (generic/irrelevant): {len(unmapped)}")

    # Show top mapped skills per category
    cat_skills: dict[str, list[str]] = {}
    for skill_name, info in taxonomy.items():
        for cat in info["categories"]:
            cat_skills.setdefault(cat, []).append(
                f"{skill_name}({info['frequency']})"
            )

    print("\nTop skills per category:")
    for cat, skills in sorted(cat_skills.items()):
        print(f"  {cat}: {', '.join(skills[:10])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--jd", default="job_requirements.yaml")
    parser.add_argument("--out", default="artifacts/skill_taxonomy.json")
    args = parser.parse_args()
    build_taxonomy(args.candidates, args.jd, args.out)


if __name__ == "__main__":
    main()
