import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

# ==========================================
# STEP 1: Load Processed Dataset
# ==========================================

print("=" * 60)
print("LOADING PROCESSED DATASET")
print("=" * 60)

df = pd.read_parquet(
    "dataset/processed_dataset.parquet"
)

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)


# ==========================================
# STEP 2: Split Dataset
# ==========================================

print("\n" + "=" * 60)
print("CREATING TRAIN / TEST SPLIT")
print("=" * 60)

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Label"]
)

print("Training Set:", train_df.shape)
print("Testing Set :", test_df.shape)


# ==========================================
# STEP 3: Select ONLY Benign Training Data
# ==========================================

print("\n" + "=" * 60)
print("SELECTING BENIGN TRAINING DATA")
print("=" * 60)

benign_train = train_df[
    train_df["Label"] == "Benign"
]

X_train = benign_train.drop(
    "Label",
    axis=1
)

print("Benign training records:", len(X_train))
print("Training features:", X_train.shape)


# ==========================================
# STEP 4: Create Isolation Forest
# ==========================================

print("\n" + "=" * 60)
print("CREATING FINAL ISOLATION FOREST")
print("=" * 60)

model = IsolationForest(
    n_estimators=100,
    contamination=0.10,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# STEP 5: Train Model
# ==========================================

print("\nTraining final model...")
print("Training data: BENIGN TRAFFIC ONLY")

model.fit(X_train)

print("Training completed successfully!")


# ==========================================
# STEP 6: Save Model
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/isolation_forest_final.pkl"
)

print("\nFinal model saved successfully!")

print(
    "Location: models/isolation_forest_final.pkl"
)


# ==========================================
# STEP 7: Save Test Dataset
# ==========================================

test_df.to_parquet(
    "dataset/test_dataset.parquet",
    index=False
)

print("\nTest dataset saved successfully!")

print(
    "Location: dataset/test_dataset.parquet"
)


# ==========================================
# SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("FINAL MODEL TRAINING COMPLETED")
print("=" * 60)

print("Algorithm           : Isolation Forest")
print("Training strategy   : Benign traffic only")
print("Contamination       : 0.10")
print("Training records    :", len(X_train))
print("Test records        :", len(test_df))
print("Features            :", X_train.shape[1])