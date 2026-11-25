import json
import re
import urllib.request
import pandas as pd

facts = json.load(open("facts.json"))
month = pd.Period(facts["month"])
this_m = month.strftime("%B %Y")
prev_m = (month - 1).strftime("%B")
last_y = (month - 12).strftime("%B %Y")

rev_lakh = round(facts["revenue"] / 100000, 1)
prof_lakh = round(facts["profit"] / 100000, 1)

fact_lines = f"""- {this_m} revenue was Rs {rev_lakh} lakh. It changed {facts['revenue_vs_last_month_%']}% versus {prev_m}, and {facts['revenue_vs_same_month_last_year_%']}% versus {last_y}.
- Profit was Rs {prof_lakh} lakh ({facts['profit_vs_same_month_last_year_%']}% versus {last_y}). Profit margin was {facts['margin_%']}%.
- {facts['top_category_by_profit']} earned the most profit. {facts['fastest_growing_category_profit']} grew its profit fastest versus last year.
- {facts['lowest_margin_category']} has the lowest profit margin.
- Best store: {facts['best_store']}. Weakest store: {facts['weakest_store']}.
- Top product by units sold: {facts['top_product_by_units']}.
- {facts['below_cost_lines']} sales lines were sold below cost, a total loss of Rs {facts['below_cost_loss_inr']}."""

prompt = f"""You are a business analyst writing a short monthly summary for the CEO of a retail chain in India.

RULES:
- Use ONLY the facts below. Do not invent any number, target, or reason.
- Write exactly 5 short sentences in simple English, then one line starting with Risk: and one line starting with Action:
- Cover: revenue versus last year, profit and margin, the top category, and the best and weakest store.
- Compare with the same month last year first. A drop versus last month right after a festive season is normal.
- The Risk must be about the lowest-margin category or the below-cost sales.
- The Action must be one concrete step that fixes that Risk.
- Do not rank categories beyond what the facts say.

FACTS:
{fact_lines}
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

print("Asking the AI... (can take a minute)")
with urllib.request.urlopen(req, timeout=300) as r:
    answer = json.loads(r.read())["response"].strip()

print("\n--- AI SUMMARY ---")
print(answer)
open("ceo_summary.txt", "w").write(answer)

def numbers(text):
    found = re.findall(r"\d[\d,]*\.?\d*", text)
    return {n.replace(",", "").rstrip(".") for n in found}

allowed = numbers(fact_lines) | numbers(this_m) | numbers(last_y)
unknown = sorted(numbers(answer) - allowed)

print("\n--- NUMBER CHECK ---")
if unknown:
    print("WARNING: these numbers are NOT in the facts:", unknown)
else:
    print("OK: every number the AI wrote exists in the facts.")
