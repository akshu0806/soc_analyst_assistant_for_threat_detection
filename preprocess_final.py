import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "dataset"

RAW_FILES = [
    "Benign-Monday-no-metadata.parquet",
    "Botnet-Friday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
    "DDoS-Friday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata.parquet"
]


# ============================================================
# STEP 1: LOAD ONLY ORIGINAL DATASET FILES
# ============================================================

print("=" * 70)
print("LOADING RAW CIC-IDS2017 DATASET")
print("=" * 70)

dfs = []

for filename in RAW_FILES:

    path = os.path.join(
        DATASET_DIR,
        filename
    )

    print("Loading:", filename)

    temp_df = pd.read_parquet(path)

    # --------------------------------------------------------
    # Convert numeric columns to float32
    # This significantly reduces memory usage
    # --------------------------------------------------------

    numeric_columns = temp_df.select_dtypes(
        include=[np.number]
    ).columns

    temp_df[numeric_columns] = temp_df[
        numeric_columns
    ].astype(np.float32)

    dfs.append(temp_df)

    print(
        "Rows loaded:",
        len(temp_df)
    )


# Combine ONLY the original files

df = pd.concat(
    dfs,
    ignore_index=True
)

print("\nRaw dataset shape:", df.shape)


# ============================================================
# STEP 2: REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("REMOVING DUPLICATES")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(
    "Duplicate rows:",
    duplicate_count
)

df = df.drop_duplicates(
    ignore_index=True
)

print(
    "Shape after duplicates:",
    df.shape
)


# ============================================================
# STEP 3: HANDLE INFINITE VALUES
# ============================================================

print("\n" + "=" * 70)
print("HANDLING INFINITE VALUES")
print("=" * 70)

# Only process numeric columns
numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

infinite_mask = np.isinf(
    df[numeric_columns].to_numpy()
)

infinite_count = infinite_mask.sum()

print(
    "Infinite values:",
    infinite_count
)

# Replace infinities only in numeric columns

df[numeric_columns] = df[
    numeric_columns
].replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# STEP 4: REMOVE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("REMOVING MISSING VALUES")
print("=" * 70)

missing_count = df.isna().sum().sum()

print(
    "Missing values:",
    missing_count
)

df = df.dropna(
    ignore_index=True
)

print(
    "Shape after cleaning:",
    df.shape
)


# ============================================================
# STEP 5: SEPARATE FEATURES AND LABEL
# ============================================================

print("\n" + "=" * 70)
print("SEPARATING FEATURES AND LABEL")
print("=" * 70)

X = df.drop(
    "Label",
    axis=1
)

y = df["Label"]

print(
    "Features:",
    X.shape
)

print(
    "Labels:",
    y.shape
)


# ============================================================
# STEP 6: TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAIN / TEST SPLIT")
print("=" * 70)

# Create indices instead of immediately copying the
# complete feature matrix.

indices = np.arange(
    len(df)
)

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "Training records:",
    len(train_indices)
)

print(
    "Testing records:",
    len(test_indices)
)


# ============================================================
# STEP 7: CREATE TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAINING DATA")
print("=" * 70)

X_train = X.iloc[
    train_indices
]

y_train = y.iloc[
    train_indices
]

print(
    "Training features:",
    X_train.shape
)


# ============================================================
# STEP 8: CREATE TEST DATA
# ============================================================

X_test = X.iloc[
    test_indices
]

y_test = y.iloc[
    test_indices
]

print(
    "Testing features:",
    X_test.shape
)


# ============================================================
# STEP 9: FIT SCALER ONLY ON TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("FITTING SCALER")
print("=" * 70)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

print(
    "Scaler fitted ONLY on training data."
)

# Transform test data using the already fitted scaler

X_test_scaled = scaler.transform(
    X_test
)

print(
    "Test data transformed using training scaler."
)


# ============================================================
# STEP 10: SAVE FEATURE ORDER
# ============================================================

feature_columns = X_train.columns.tolist()

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    feature_columns,
    "models/feature_columns.pkl"
)

print(
    "Feature order saved."
)


# ============================================================
# STEP 11: SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "models/scaler_final.pkl"
)

print(
    "Scaler saved:"
)

print(
    "models/scaler_final.pkl"
)


# ============================================================
# STEP 12: CREATE PROCESSED TRAIN DATASET
# ============================================================

train_processed = pd.DataFrame(
    X_train_scaled,
    columns=feature_columns
)

train_processed["Label"] = y_train.to_numpy()


# ============================================================
# STEP 13: CREATE PROCESSED TEST DATASET
# ============================================================

test_processed = pd.DataFrame(
    X_test_scaled,
    columns=feature_columns
)

test_processed["Label"] = y_test.to_numpy()


# ============================================================
# STEP 14: SAVE PROCESSED DATASETS
# ============================================================

train_processed.to_parquet(
    "dataset/train_processed_final.parquet",
    index=False
)

test_processed.to_parquet(
    "dataset/test_processed_final.parquet",
    index=False
)

print("\nTraining dataset saved:")
print(
    "dataset/train_processed_final.parquet"
)

print("\nTesting dataset saved:")
print(
    "dataset/test_processed_final.parquet"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREPROCESSING COMPLETED")
print("=" * 70)

print(
    "Total cleaned records:",
    len(df)
)

print(
    "Training records:",
    len(train_processed)
)

print(
    "Testing records:",
    len(test_processed)
)

print(
    "Number of features:",
    len(feature_columns)
)

print("\nIMPORTANT:")
print(
    "Only original CIC-IDS2017 files were loaded."
)

print(
    "Scaler was fitted ONLY on training data."
)

print(
    "Test data was NOT used during scaler fitting."
)

print("\nNext step:")
print(
    "python train_isolation_final.py"
)