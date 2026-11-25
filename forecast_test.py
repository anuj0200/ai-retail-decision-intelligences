import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

sales = pd.read_csv("data/fact_sales.csv", parse_dates=["order_date"])
cal = pd.read_csv("data/dim_date.csv", parse_dates=["date"])

daily = sales.groupby("order_date")["revenue"].sum().reset_index()
daily.columns = ["date", "revenue"]
df = daily.merge(cal[["date", "is_festival_window", "is_promo_day"]], on="date")

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["trend"] = (df["date"] - df["date"].min()).dt.days
features = ["day_of_week", "month", "day_of_year", "trend", "is_festival_window", "is_promo_day"]

train, test = df.iloc[:-60], df.iloc[-60:]

naive = np.full(len(test), train["revenue"].tail(28).mean())

model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3)
model.fit(train[features], train["revenue"])
pred = model.predict(test[features])

def report(name, p):
    mae = np.mean(np.abs(test["revenue"] - p))
    mape = np.mean(np.abs(test["revenue"] - p) / test["revenue"]) * 100
    print(f"{name:12s} MAE = Rs {mae:,.0f}   error = {mape:.1f}%")

report("Simple guess", naive)
report("Smart model", pred)

plt.figure(figsize=(10, 4))
plt.plot(test["date"], test["revenue"], label="Real sales")
plt.plot(test["date"], pred, label="Smart model")
plt.plot(test["date"], naive, label="Simple guess", linestyle="--")
plt.title("Last 60 days: real vs predicted")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("forecast_test.png")
print("Chart saved as forecast_test.png")
