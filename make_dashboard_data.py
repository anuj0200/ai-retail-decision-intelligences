import pandas as pd

sales = pd.read_csv("data/fact_sales.csv", parse_dates=["order_date"])
prod = pd.read_csv("data/dim_product.csv")
store = pd.read_csv("data/dim_store.csv")
cal = pd.read_csv("data/dim_date.csv", parse_dates=["date"])
seg = pd.read_csv("data/rfm_segments.csv")

df = (sales
      .merge(prod[["product_id", "product_name", "category", "brand"]], on="product_id")
      .merge(store[["store_id", "city", "region", "store_size"]], on="store_id")
      .merge(cal[["date", "is_promo_day", "is_festival_window"]],
             left_on="order_date", right_on="date")
      .merge(seg[["customer_id", "segment"]], on="customer_id", how="left"))

df["segment"] = df["segment"].fillna("Unknown")
df["profit"] = (df["revenue"] - df["cogs"]).round(2)
df["below_cost"] = (df["revenue"] < df["cogs"]).astype(int)
df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["month_name"] = df["order_date"].dt.month_name()
df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")

df = df[["order_date", "year", "month", "month_name", "order_id", "customer_id", "segment",
         "city", "region", "store_size", "category", "product_name", "brand",
         "quantity", "unit_price", "discount_pct", "revenue", "cogs", "profit",
         "below_cost", "is_promo_day", "is_festival_window"]]
df.to_csv("dashboard_data.csv", index=False)

print("Rows:", len(df))
print("Total revenue:", round(df["revenue"].sum()))
print("Total profit:", round(df["profit"].sum()))
print()
print("Customers per segment:")
print(df.drop_duplicates("customer_id").groupby("segment").size().to_string())
