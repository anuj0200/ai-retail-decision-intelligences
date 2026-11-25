import json
import re
import urllib.request
import pandas as pd

facts = json.load(open("facts.json"))
month = pd.Period(facts["month"])
this_m = month.strftime("%B %Y")
prev_m = (month - 1).strftime("%B")
last_y = (month - 12).strftime("%B %Y")

def move(pct):
    return ("up " if pct >= 0 else "down ") + f"{abs(pct)}%"

rev_lakh = round(facts["revenue"] / 100000, 1)
prof_lakh = round(facts["profit"] / 100000, 1)

# PART 1: Python writes every sentence that contains a number
facts_text = (
    f"{this_m} revenue was Rs {rev_lakh} lakh, "
    f"{move(facts['revenue_vs_same_month_last_year_%'])} versus {last_y} "
    f"and {move(facts['revenue_vs_last_month_%'])} versus {prev_m}. "
    f"Profit was Rs {prof_lakh} lakh, {move(facts['profit_vs_same_month_last_year_%'])} versus {last_y}, "
    f"with a profit margin of {facts['margin_%']}%. "
    f"{facts['top_category_by_profit']} earned the most profit, and "
    f"{facts['fastest_growing_category_profit']} grew its profit fastest. "
    f"Our best store was {facts['best_store']} and our weakest was {facts['weakest_store']}. "
    f"{facts['below_cost_lines']} sales lines were sold below cost, a total loss of Rs {facts['below_cost_loss_inr']:,}."
)

# PART 2: the AI writes only the Risk and the Action
prompt = f"""You are a business analyst advising the CEO of a retail chain in India.

KNOWN FACTS:
- {facts['lowest_margin_category']} has the lowest profit margin of all categories.
- {facts['below_cost_lines']} sales lines were sold below cost this month.
- Earlier analysis showed that deep promo discounts, bigger than the thin margin of {facts['lowest_margin_category']}, cause these below-cost sales.

TASK: Write exactly two lines.
Line 1 starts with "Risk:" and explains the risk in one sentence.
Line 2 starts with "Action:" and gives one concrete step, such as capping promo discounts for {facts['lowest_margin_category']}.

RULES: Use only the known facts. Do not mention targets, goals, budgets or forecasts. Do not write any number.
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

print("Asking the AI for the Risk and Action... (can take a minute)")
with urllib.request.urlopen(req, timeout=300) as r:
    ai_part = json.loads(r.read())["response"].strip()

summary = facts_text + "\n\n" + ai_part
print("\n--- CEO SUMMARY ---")
print(summary)
open("ceo_summary.txt", "w").write(summary)

print("\n--- CHECKS ---")
problems = []
if re.search(r"\d", ai_part):
    problems.append("the AI wrote a number (we told it not to)")
for word in ["target", "goal", "budget", "forecast"]:
    if word in ai_part.lower():
        problems.append(f"the AI mentioned '{word}', which is not in the facts")
if "Risk:" not in ai_part or "Action:" not in ai_part:
    problems.append("the AI did not follow the Risk/Action format")
if problems:
    for p in problems:
        print("WARNING:", p)
else:
    print("OK: no numbers, no invented words, format followed. Still read it yourself!")
