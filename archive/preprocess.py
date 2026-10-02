import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

# Load all Parquet files

folder = "dataset"
dfs = []

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

for file in os.listdir(folder):
    if file.endswith(".parquet") and file != "processed_dataset.parquet":
        path = os.path.join(folder, file)
        df = pd.read_parquet(path)
        dfs.append(df)
        print(f"Loaded: {file}")

# Combine all datasets into one
combined_df = pd.concat(dfs, ignore_index=True)

print("\nDataset Loaded Successfully!")
print(f"Original Shape: {combined_df.shape}")

# Check Missing Values

print("\n" + "=" * 60)
print("CHECKING MISSING VALUES")
print("=" * 60)

missing_values = combined_df.isnull().sum().sum()

print(f"Total Missing Values: {missing_values}")

# Check Duplicate Rows

print("\n" + "=" * 60)
print("CHECKING DUPLICATE ROWS")
print("=" * 60)

duplicates = combined_df.duplicated().sum()

print(f"Duplicate Rows: {duplicates}")

combined_df = combined_df.drop_duplicates()

print(f"Shape After Removing Duplicates: {combined_df.shape}")

# Check Infinite Values

print("\n" + "=" * 60)
print("CHECKING INFINITE VALUES")
print("=" * 60)

numeric_df = combined_df.select_dtypes(include=[np.number])

infinite_values = np.isinf(numeric_df).sum().sum()

print(f"Infinite Values: {infinite_values}")

# Replace infinite values with NaN
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows containing NaN values
combined_df.dropna(inplace=True)

print(f"Shape After Cleaning: {combined_df.shape}")

# Label Distribution

print("\n" + "=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)

print(combined_df["Label"].value_counts())

# Separate Features and Labels

X = combined_df.drop("Label", axis=1)
y = combined_df["Label"]

print("\nFeatures Shape:", X.shape)
print("Labels Shape:", y.shape)

# Feature Scaling

print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature Scaling Completed!")
print("Scaled Feature Shape:", X_scaled.shape)

# Save Scaler

os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler.pkl")

print("\nScaler saved successfully!")
print("Location: models/scaler.pkl")

# Save Processed Dataset

processed_df = pd.DataFrame(X_scaled, columns=X.columns)

processed_df["Label"] = y.values

processed_df.to_parquet(
    "dataset/processed_dataset.parquet",
    index=False
)

print("\nProcessed dataset saved successfully!")
print("Location: dataset/processed_dataset.parquet")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Final Dataset Shape : {processed_df.shape}")
print(f"Number of Features : {X.shape[1]}")
print(f"Number of Classes : {y.nunique()}")

print("\nDataset is now ready for model training.")