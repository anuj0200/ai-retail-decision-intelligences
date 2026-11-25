import json
import urllib.request

facts = json.load(open("facts.json"))
facts["revenue_in_lakh"] = round(facts["revenue"] / 100000, 1)
facts["profit_in_lakh"] = round(facts["profit"] / 100000, 1)

prompt = f"""You are a business analyst writing a short monthly summary for the CEO of a retail chain in India.

RULES:
- Use ONLY the facts below. Do not invent any number or reason.
- Write 4 or 5 short sentences in simple English.
- Say money in lakh rupees, like Rs 23.4 lakh.
- End with one risk and one action to take.

FACTS:
{json.dumps(facts, indent=2)}
"""

body = json.dumps({
    "model": "llama3.2",
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.2},
}).encode()

req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=body,
    headers={"Content-Type": "application/json"},
)

print("Asking the AI... (the first time can take a minute)")
with urllib.request.urlopen(req, timeout=300) as r:
    answer = json.loads(r.read())["response"].strip()

print("\n--- AI SUMMARY ---")
print(answer)
open("ceo_summary.txt", "w").write(answer)
