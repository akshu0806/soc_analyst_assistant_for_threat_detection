import pandas as pd
import os

folder = "dataset"

dfs = []

for file in os.listdir(folder):

    if file.endswith(".parquet"):

        path = os.path.join(folder, file)

        df = pd.read_parquet(path)

        dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

print("Shape:")
print(combined_df.shape)

print()

print("Label Counts:")
print(combined_df["Label"].value_counts())