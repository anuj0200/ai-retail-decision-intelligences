import json
import pandas as pd

sales = pd.read_csv("data/fact_sales.csv", parse_dates=["order_date"])
prod = pd.read_csv("data/dim_product.csv")
store = pd.read_csv("data/dim_store.csv")

sales = sales.merge(prod[["product_id", "product_name", "category"]], on="product_id")
sales = sales.merge(store[["store_id", "city"]], on="store_id")
sales["profit"] = sales["revenue"] - sales["cogs"]
sales["month"] = sales["order_date"].dt.to_period("M")

this_month = sales["month"].max()
last_month = this_month - 1
last_year = this_month - 12

def totals(m):
    d = sales[sales["month"] == m]
    return d["revenue"].sum(), d["profit"].sum()

rev, prof = totals(this_month)
rev_lm, prof_lm = totals(last_month)
rev_ly, prof_ly = totals(last_year)

def pct(a, b):
    return round(100 * (a - b) / b, 1)

cat = sales[sales["month"] == this_month].groupby("category")[["revenue", "profit"]].sum()
cat_ly = sales[sales["month"] == last_year].groupby("category")[["revenue", "profit"]].sum()
cat["profit_change_vs_last_year_%"] = ((cat["profit"] - cat_ly["profit"]) / cat_ly["profit"] * 100).round(1)
cat["margin_%"] = (cat["profit"] / cat["revenue"] * 100).round(1)

cur = sales[sales["month"] == this_month]
store_rank = cur.groupby("city")["revenue"].sum().sort_values(ascending=False)
below_cost = cur[cur["revenue"] < cur["cogs"]]

facts = {
    "month": str(this_month),
    "revenue": round(rev),
    "revenue_vs_last_month_%": pct(rev, rev_lm),
    "revenue_vs_same_month_last_year_%": pct(rev, rev_ly),
    "profit": round(prof),
    "profit_vs_same_month_last_year_%": pct(prof, prof_ly),
    "margin_%": round(100 * prof / rev, 1),
    "top_category_by_profit": cat["profit"].idxmax(),
    "fastest_growing_category_profit": cat["profit_change_vs_last_year_%"].idxmax(),
    "lowest_margin_category": cat["margin_%"].idxmin(),
    "best_store": store_rank.index[0],
    "weakest_store": store_rank.index[-1],
    "top_product_by_units": cur.groupby("product_name")["quantity"].sum().idxmax(),
    "below_cost_lines": int(len(below_cost)),
    "below_cost_loss_inr": round((below_cost["cogs"] - below_cost["revenue"]).sum()),
}

with open("facts.json", "w") as f:
    json.dump(facts, f, indent=2)

print("INSIGHT FACTS for", facts["month"])
for k, v in facts.items():
    print(f"  {k}: {v}")
