import json

lines = []
with open('results/gemini_output.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # Keep only if response is not empty
        if obj.get('response', '').strip() != '':
            lines.append(line)

with open('results/gemini_output.jsonl', 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line + '\n')

print(f"Cleaned output. Valid responses: {len(lines)}")
