import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('artifacts/skill_taxonomy.json', encoding='utf-8') as f:
    t = json.load(f)
print("All 133 skills sorted by frequency:")
for name, info in sorted(t.items(), key=lambda x: -x[1]['frequency']):
    cats = info['categories']
    freq = info['frequency']
    cat_str = str(cats) if cats else "[unmapped]"
    print(f"{freq:6d}  {cat_str:60s}  {name}")
