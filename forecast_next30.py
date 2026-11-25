import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

sales = pd.read_csv("data/fact_sales.csv", parse_dates=["order_date"])
cal = pd.read_csv("data/dim_date.csv", parse_dates=["date"])

daily = sales.groupby("order_date")["revenue"].sum().reset_index()
daily.columns = ["date", "revenue"]
df = daily.merge(cal[["date", "is_festival_window", "is_promo_day"]], on="date")

def add_features(d, start):
    d["day_of_week"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["day_of_year"] = d["date"].dt.dayofyear
    d["trend"] = (d["date"] - start).dt.days
    return d

start = df["date"].min()
df = add_features(df, start)
features = ["day_of_week", "month", "day_of_year", "trend", "is_festival_window", "is_promo_day"]

model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3)
model.fit(df[features], df["revenue"])

future = pd.DataFrame({"date": pd.date_range("2026-01-01", "2026-01-31")})
future["is_festival_window"] = 0
future["is_promo_day"] = future["date"].between("2026-01-20", "2026-01-28").astype(int)
future = add_features(future, start)
future["forecast"] = model.predict(future[features]).round()

jan25 = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-01-31")]["revenue"].sum()
total = future["forecast"].sum()
print(f"Forecast for January 2026: Rs {total:,.0f}")
print(f"Actual January 2025:       Rs {jan25:,.0f}")
print(f"Change vs last January:    {100*(total-jan25)/jan25:+.1f}%")
future[["date", "forecast"]].to_csv("forecast_jan2026.csv", index=False)

tr = df[df["date"] < "2025-11-01"]
te = df[df["date"] >= "2025-11-01"].copy()
m2 = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3).fit(tr[features], tr["revenue"])
te["pred"] = m2.predict(te[features])
g = te.groupby("month")[["revenue", "pred"]].sum()
g["miss_%"] = 100 * (g["pred"] - g["revenue"]) / g["revenue"]
print("Monthly test (Nov, Dec 2025):")
print(g.round(1))

plt.figure(figsize=(10, 4))
plt.plot(df["date"].tail(60), df["revenue"].tail(60), label="Real sales (last 60 days)")
plt.plot(future["date"], future["forecast"], label="Forecast Jan 2026")
plt.title("Forecast for the next 31 days")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("forecast_next30.png")
print("Chart saved as forecast_next30.png")
