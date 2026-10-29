from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
data_path = ROOT / "data" / "retail_sales_raw.csv"
out = ROOT / "outputs"
out.mkdir(exist_ok=True)

df = pd.read_csv(data_path)
initial_rows = len(df)
initial_duplicates = int(df.duplicated().sum())

df = df.drop_duplicates().copy()
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
for col in ["region", "category", "customer_type"]:
    df[col] = df[col].astype("string").str.strip().str.title()
df["units"] = pd.to_numeric(df["units"], errors="coerce")
df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce")
df.loc[(df["units"] <= 0) | (df["units"].isna()), "units"] = 1
df.loc[(df["unit_price"] <= 0) | (df["unit_price"].isna()), "unit_price"] = df["unit_price"].median()
df.loc[(df["discount_pct"] < 0) | (df["discount_pct"] > 0.8), "discount_pct"] = 0
df["region"] = df["region"].fillna("Unknown")
df["category"] = df["category"].fillna("Unknown")
df["customer_type"] = df["customer_type"].fillna("Unknown")
df["discount_pct"] = df["discount_pct"].fillna(0)
df["revenue"] = df["units"] * df["unit_price"] * (1 - df["discount_pct"])
df.to_csv(out / "retail_sales_cleaned.csv", index=False)

summary = {
    "rows_before_cleaning": initial_rows,
    "duplicate_rows_removed": initial_duplicates,
    "rows_after_cleaning": len(df),
    "missing_values_after_cleaning": int(df.isna().sum().sum()),
    "total_revenue": round(float(df["revenue"].sum()), 2),
    "average_order_revenue": round(float(df["revenue"].mean()), 2),
}
pd.Series(summary, name="value").to_csv(out / "cleaning_summary.csv")
df.groupby("category", dropna=False)["revenue"].sum().sort_values(ascending=False).to_csv(out / "revenue_by_category.csv")

plt.figure(figsize=(8, 5))
df.groupby("category")["revenue"].sum().sort_values().plot(kind="barh")
plt.title("Revenue by Product Category")
plt.xlabel("Revenue")
plt.tight_layout()
plt.savefig(out / "revenue_by_category.png", dpi=150)
plt.close()

df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
plt.figure(figsize=(9, 5))
df.groupby("order_month")["revenue"].sum().sort_index().plot(marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(out / "monthly_revenue.png", dpi=150)
plt.close()

print("Task 1 complete. Outputs saved to:", out)
print(pd.Series(summary))
