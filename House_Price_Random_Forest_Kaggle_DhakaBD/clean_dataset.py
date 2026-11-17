
import pandas as pd


df = pd.read_csv("house_price_bd.csv")

print("Original dataset shape:", df.shape)

df = df[["Bedrooms", "Bathrooms", "Floor_area", "Price_in_taka"]]

df = df.drop_duplicates()

df["Price_in_taka"] = (
    df["Price_in_taka"]
    .str.replace("৳", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["Price_in_taka"] = pd.to_numeric(
    df["Price_in_taka"], errors="coerce"
)

features = ["Bedrooms", "Bathrooms", "Floor_area"]

for column in features:
    df[column] = df[column].fillna(df[column].median())

df = df.dropna(subset=["Price_in_taka"])


df.to_csv("HousePrice_Cleaned.csv", index=False)


print("\nCleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nFirst 10 cleaned records:")
print(df.head(10).to_string(index=False))

print("\nCleaned dataset saved successfully!")
