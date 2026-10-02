import pandas as pd

df = pd.read_parquet("dataset/processed_dataset.parquet")

print("Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())