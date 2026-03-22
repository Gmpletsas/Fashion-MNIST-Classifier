import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data
data = pd.read_csv("D:/Giannis/downloads/finance_liquor_sales.csv")

# data cleaning
data['date'] = pd.to_datetime(data['date'])
new_data = data[(data["date"].dt.year >= 2016) & (data["date"].dt.year <= 2019)].reset_index()

print(data.isna().sum())
new_data.info()

# calculate the best-selling item in each zip code
bottles_sold = new_data.groupby(["zip_code", "item_number"])["bottles_sold"].sum().reset_index()
bottles_sold["zip_code"] = bottles_sold["zip_code"].astype(int)

print(bottles_sold)
idx = bottles_sold.groupby("zip_code")["bottles_sold"].idxmax()
max_bottles = bottles_sold.loc[idx].reset_index(drop=True)

sorted_bottles = max_bottles.sort_values(by="bottles_sold", ascending=False).head(20)
print(sorted_bottles)

# create barplot showing the results
plt.figure(figsize=(10,6))
plt.bar(sorted_bottles["zip_code"].astype(str), sorted_bottles["bottles_sold"], color="skyblue")
plt.xlabel("Zip Code")
plt.ylabel("Bottles Sold")
plt.title("Max bottles sold per zip code")
plt.xticks(rotation=45)
plt.show()

# compute total sales
total_sales = sum(new_data["sale_dollars"])

# compute sales per store
sales_store = new_data.groupby(["store_name"])["sale_dollars"].sum()

# calculate sales percentage per store
percentage_store = (sales_store * 100 / total_sales).round(2)
print(percentage_store)

#select top 15 stores
percentage_store_sorted = percentage_store.sort_values(ascending=False).head(15)

#construct plot
p = plt.barh(percentage_store_sorted.index, percentage_store_sorted.values, height=0.7)
plt.title("%Sales per store")
plt.xlabel("%Sales", fontsize=12)
plt.bar_label(p, fmt="%.2f")
plt.xlim([0,20])
plt.show()