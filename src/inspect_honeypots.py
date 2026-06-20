import json
import sys
from collections import defaultdict
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scoring.honeypot_filter import check_honeypot

def profile_honeypots():
    file_path = "data/candidates.jsonl"
    rule_counts = defaultdict(int)
    total_candidates = 0
    total_flagged = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total_candidates += 1
            cand = json.loads(line)
            res = check_honeypot(cand)
            is_hp = res["is_honeypot"]
            reasons = res["flags"]
            if is_hp:
                total_flagged += 1
            for reason in reasons:
                rule_name = reason.split(":")[0]  # Extracts [H1], [H2] etc
                rule_counts[rule_name] += 1

    print(f"Total Candidates: {total_candidates}")
    print(f"Total Flagged as Honeypot: {total_flagged} ({(total_flagged/total_candidates)*100:.2f}%)")
    print("Breakdown by Rule:")
    for rule, count in sorted(rule_counts.items()):
        print(f"  {rule}: {count} ({(count/total_candidates)*100:.2f}%)")

if __name__ == "__main__":
    profile_honeypots()
