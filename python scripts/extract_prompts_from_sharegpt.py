import json

INPUT_FILE = "sg_52k.json"          # your downloaded dataset
OUTPUT_FILE = "ShareGPTprompts.json"          # final output

output = []
idx = 1

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "conversations" not in item:
        continue

    conv = item["conversations"]
    if not conv:
        continue

    # take only the FIRST human prompt
    if conv[0].get("from") == "human":
        prompt = conv[0].get("value", "").strip()
        if prompt:
            output.append({
                "task_id": idx,
                "prompt": prompt
            })
            idx += 1

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(output)} prompts to {OUTPUT_FILE}")
