"""
Synthetic retail dataset generator (India, INR) for a retail decision-intelligence project.

Run:  python3 generate_retail_data.py
Out:  ./retail_data/  dim_store.csv, dim_product.csv, dim_customer.csv, dim_date.csv, fact_sales.csv

Hidden patterns baked in (so your analysis has real things to discover):
  - Trend growth + weekly + monthly seasonality + festival (Diwali) spikes
  - Category seasonality (beverages in summer, apparel in winter/festive)
  - Promo periods with deeper discounts
  - Stockout windows (product x store goes to zero sales for 1-3 weeks)
  - One store closure (outage) and one viral product spike (anomalies)
  - Customer churn, and repeat-purchase behaviour differences (RFM segments)
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = "retail_data"
os.makedirs(OUT, exist_ok=True)

START, END = "2024-01-01", "2025-12-31"
dates = pd.date_range(START, END, freq="D")
ND = len(dates)

# ---------------------------------------------------------------- stores
cities = [
    ("Mumbai", "West", "Large"), ("Pune", "West", "Medium"), ("Ahmedabad", "West", "Medium"),
    ("Delhi", "North", "Large"), ("Gurugram", "North", "Medium"), ("Jaipur", "North", "Small"),
    ("Lucknow", "North", "Small"), ("Chandigarh", "North", "Small"),
    ("Bengaluru", "South", "Large"), ("Chennai", "South", "Large"), ("Hyderabad", "South", "Large"),
    ("Kochi", "South", "Small"), ("Coimbatore", "South", "Small"),
    ("Kolkata", "East", "Large"), ("Bhubaneswar", "East", "Small"), ("Patna", "East", "Small"),
    ("Bhopal", "Central", "Medium"), ("Indore", "Central", "Medium"),
    ("Jabalpur", "Central", "Small"), ("Nagpur", "Central", "Medium"),
]
size_w = {"Large": 1.8, "Medium": 1.0, "Small": 0.55}
stores = pd.DataFrame({
    "store_id": range(1, len(cities) + 1),
    "city": [c[0] for c in cities],
    "region": [c[1] for c in cities],
    "store_size": [c[2] for c in cities],
})
stores["store_name"] = "Store " + stores["city"]
store_weight = stores["store_size"].map(size_w).values * rng.uniform(0.85, 1.15, len(stores))
store_weight = store_weight / store_weight.sum()
NS = len(stores)

# ---------------------------------------------------------------- products
cats = {
    "Grocery Staples":   (40, 400,   0.12, 1.6),
    "Beverages":         (20, 150,   0.30, 1.4),
    "Snacks":            (10, 200,   0.28, 1.5),
    "Personal Care":     (50, 500,   0.35, 1.0),
    "Home & Kitchen":    (150, 2500, 0.38, 0.7),
    "Electronics Acc.":  (199, 3500, 0.45, 0.5),
    "Apparel":           (299, 3000, 0.50, 0.7),
    "Stationery":        (20, 300,   0.40, 0.6),
}
rows, pid = [], 1
n_per_cat = {"Grocery Staples": 30, "Beverages": 25, "Snacks": 30, "Personal Care": 30,
             "Home & Kitchen": 25, "Electronics Acc.": 20, "Apparel": 25, "Stationery": 15}
for cat, (lo, hi, margin, pop) in cats.items():
    for i in range(n_per_cat[cat]):
        price = float(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi))), 0))
        cost = float(np.round(price * (1 - margin * rng.uniform(0.7, 1.2)), 2))
        rows.append((pid, f"{cat.split()[0]}-{i+1:02d}", cat, price, cost,
                     rng.lognormal(0, 0.9) * pop))
        pid += 1
products = pd.DataFrame(rows, columns=["product_id", "product_name", "category",
                                       "list_price", "cost_price", "popularity"])
products["brand"] = rng.choice(["BrandA", "BrandB", "BrandC", "BrandD", "Private Label"],
                               len(products), p=[0.25, 0.22, 0.2, 0.13, 0.2])
NP = len(products)
pop = products["popularity"].values
prod_cat = products["category"].values

# category x month seasonality
def cat_month_factor(cat, m):
    f = 1.0
    if cat == "Beverages" and m in (4, 5, 6): f = 1.8
    if cat == "Beverages" and m in (12, 1): f = 0.8
    if cat == "Apparel" and m in (10, 11, 12): f = 1.7
    if cat == "Apparel" and m in (5, 6): f = 0.8
    if cat == "Electronics Acc." and m in (10, 11): f = 1.6
    if cat == "Stationery" and m in (6, 7): f = 1.8  # school season
    if cat == "Snacks" and m in (10, 11): f = 1.4
    if cat == "Home & Kitchen" and m in (10, 11): f = 1.5
    return f
cmf = np.array([[cat_month_factor(c, m) for m in range(1, 13)] for c in prod_cat])  # NP x 12

# ---------------------------------------------------------------- customers
NC = 6000
signup = pd.to_datetime("2023-06-01") + pd.to_timedelta(rng.integers(0, 880, NC), unit="D")
signup = signup.where(signup <= pd.Timestamp(END) - pd.Timedelta(days=10), pd.Timestamp("2025-06-01"))
churn_mask = rng.random(NC) < 0.28
churn_after = pd.to_timedelta(rng.integers(60, 500, NC), unit="D")
churn_date = pd.Series(pd.NaT, index=range(NC), dtype="datetime64[ns]")
churn_date[churn_mask] = (signup + churn_after)[churn_mask]
# purchase propensity: heavy-tailed -> a few loyal, many occasional
propensity = rng.pareto(2.2, NC) + 0.15
home_store = rng.choice(NS, NC, p=store_weight)
customers = pd.DataFrame({
    "customer_id": range(1, NC + 1),
    "city": stores["city"].values[home_store],
    "region": stores["region"].values[home_store],
    "signup_date": signup.values,
    "gender": rng.choice(["F", "M", "Other"], NC, p=[0.48, 0.5, 0.02]),
    "age_group": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], NC,
                            p=[0.18, 0.33, 0.24, 0.15, 0.10]),
})
signup_np = signup.values
churn_np = churn_date.values  # NaT -> far future
churn_np = np.where(pd.isna(churn_np), np.datetime64("2100-01-01"), churn_np)

# ---------------------------------------------------------------- calendar
festival_peaks = [pd.Timestamp("2024-11-01"), pd.Timestamp("2025-10-20")]  # Diwali
promo_windows = [
    ("2024-01-20", "2024-01-28", 0.12), ("2024-06-28", "2024-07-07", 0.15),
    ("2024-10-18", "2024-11-05", 0.22), ("2024-12-24", "2025-01-02", 0.15),
    ("2025-01-20", "2025-01-28", 0.12), ("2025-06-27", "2025-07-06", 0.15),
    ("2025-10-08", "2025-10-28", 0.22), ("2025-12-24", "2025-12-31", 0.15),
]
promo_windows = [(pd.Timestamp(a), pd.Timestamp(b), d) for a, b, d in promo_windows]

def festival_factor(d):
    f = 1.0
    for p in festival_peaks:
        gap = abs((d - p).days)
        if gap <= 12:
            f = max(f, 1.0 + 0.75 * (1 - gap / 13))
    return f

def promo_discount(d):
    for a, b, disc in promo_windows:
        if a <= d <= b:
            return disc
    return 0.0

dow_factor = {0: 0.9, 1: 0.85, 2: 0.9, 3: 0.95, 4: 1.1, 5: 1.35, 6: 1.3}

# ---------------------------------------------------------------- stockouts / anomalies
stock_ok = np.ones((NS, NP, ND), dtype=bool)
n_stockouts = 220
for _ in range(n_stockouts):
    s = rng.integers(0, NS)
    p = rng.choice(NP, p=pop / pop.sum())   # popular products stock out more visibly
    start = rng.integers(10, ND - 25)
    length = rng.integers(7, 22)
    stock_ok[s, p, start:start + length] = False

closed_store = 5                              # Jaipur: power outage / closure
closed_start = int(np.where(dates == pd.Timestamp("2024-08-12"))[0][0])
store_open = np.ones((NS, ND), dtype=bool)
store_open[closed_store, closed_start:closed_start + 10] = False

viral_product = int(products.index[products["category"] == "Snacks"][3])
viral_start = int(np.where(dates == pd.Timestamp("2025-03-05"))[0][0])
viral_len = 9

# ---------------------------------------------------------------- generate sales
BASE_ORDERS = 42
lines = []
order_id = 100000
for di, d in enumerate(dates):
    trend = 1.0 + 0.30 * di / ND
    lam = BASE_ORDERS * trend * dow_factor[d.dayofweek] * festival_factor(d)
    n_orders = rng.poisson(lam)
    active = (signup_np <= np.datetime64(d)) & (churn_np > np.datetime64(d))
    idx = np.where(active)[0]
    if len(idx) == 0:
        continue
    pw = propensity[idx] / propensity[idx].sum()
    cust_pick = rng.choice(idx, n_orders, p=pw)
    disc_base = promo_discount(d)
    m = d.month - 1
    for oc in cust_pick:
        # customers mostly shop at home store, sometimes elsewhere
        if rng.random() < 0.8:
            s = home_store[oc]
        else:
            s = rng.choice(NS, p=store_weight)
        if not store_open[s, di]:
            continue
        order_id += 1
        n_lines = min(1 + rng.geometric(0.5) - 1 + (rng.random() < 0.5), 6)
        w = pop * cmf[:, m] * stock_ok[s, :, di]
        if viral_start <= di < viral_start + viral_len:
            w = w.copy()
            w[viral_product] *= 25
        if w.sum() == 0:
            continue
        chosen = rng.choice(NP, n_lines, replace=False, p=w / w.sum()) if n_lines <= (w > 0).sum() else [rng.choice(NP, p=w / w.sum())]
        for p in chosen:
            qty = int(1 + rng.poisson(0.6 if products.at[p, "list_price"] > 500 else 1.2))
            if disc_base > 0:
                disc = round(disc_base * rng.uniform(0.6, 1.2), 3)
            else:
                disc = round(rng.choice([0, 0, 0, 0.05, 0.1], p=[0.7, 0.1, 0.05, 0.1, 0.05]), 3)
            price = products.at[p, "list_price"]
            net_unit = price * (1 - disc)
            lines.append((order_id, d, oc + 1, p + 1, s + 1, qty, price, disc,
                          round(net_unit * qty, 2),
                          round(products.at[p, "cost_price"] * qty, 2)))

fact = pd.DataFrame(lines, columns=["order_id", "order_date", "customer_id", "product_id",
                                    "store_id", "quantity", "unit_price", "discount_pct",
                                    "revenue", "cogs"])
fact.insert(0, "line_id", range(1, len(fact) + 1))

# ---------------------------------------------------------------- dim_date
dim_date = pd.DataFrame({"date": dates})
dim_date["date_id"] = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
dim_date["year"] = dim_date["date"].dt.year
dim_date["quarter"] = dim_date["date"].dt.quarter
dim_date["month"] = dim_date["date"].dt.month
dim_date["month_name"] = dim_date["date"].dt.month_name()
dim_date["week"] = dim_date["date"].dt.isocalendar().week.astype(int)
dim_date["day_of_week"] = dim_date["date"].dt.day_name()
dim_date["is_weekend"] = (dim_date["date"].dt.dayofweek >= 5).astype(int)
dim_date["is_festival_window"] = dim_date["date"].apply(lambda d: int(festival_factor(d) > 1.0))
dim_date["is_promo_day"] = dim_date["date"].apply(lambda d: int(promo_discount(d) > 0))
dim_date = dim_date[["date_id", "date", "year", "quarter", "month", "month_name", "week",
                     "day_of_week", "is_weekend", "is_festival_window", "is_promo_day"]]

# ---------------------------------------------------------------- save
fact["order_date"] = fact["order_date"].dt.strftime("%Y-%m-%d")
dim_date["date"] = dim_date["date"].dt.strftime("%Y-%m-%d")
customers["signup_date"] = pd.to_datetime(customers["signup_date"]).dt.strftime("%Y-%m-%d")
products.drop(columns=["popularity"]).to_csv(f"{OUT}/dim_product.csv", index=False)
stores.to_csv(f"{OUT}/dim_store.csv", index=False)
customers.to_csv(f"{OUT}/dim_customer.csv", index=False)
dim_date.to_csv(f"{OUT}/dim_date.csv", index=False)
fact.to_csv(f"{OUT}/fact_sales.csv", index=False)

print(f"fact_sales rows : {len(fact):,}")
print(f"orders          : {fact['order_id'].nunique():,}")
print(f"customers       : {len(customers):,} | products: {NP} | stores: {NS}")
print(f"total revenue   : INR {fact['revenue'].sum():,.0f}")
print(f"saved to        : ./{OUT}/")
