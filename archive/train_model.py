import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest

# ==========================================
# STEP 1: Load Processed Dataset
# ==========================================

print("=" * 60)
print("LOADING PROCESSED DATASET")
print("=" * 60)

df = pd.read_parquet("dataset/processed_dataset.parquet")

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

# ==========================================
# STEP 2: Select ONLY Benign Traffic
# ==========================================

print("\n" + "=" * 60)
print("SELECTING BENIGN TRAFFIC")
print("=" * 60)

benign_df = df[df["Label"] == "Benign"]

print("Total records:", len(df))
print("Benign records:", len(benign_df))

# ==========================================
# STEP 3: Separate Features
# ==========================================

X_train = benign_df.drop("Label", axis=1)

print("\nTraining Features Shape:", X_train.shape)

# ==========================================
# STEP 4: Create Isolation Forest
# ==========================================

print("\n" + "=" * 60)
print("CREATING ISOLATION FOREST")
print("=" * 60)

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

# ==========================================
# STEP 5: Train ONLY on Benign Traffic
# ==========================================

print("\nTraining model using ONLY benign traffic...")

model.fit(X_train)

print("Model training completed!")

# ==========================================
# STEP 6: Save Model
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/isolation_forest_benign.pkl"
)

print("\nModel saved successfully!")
print("Location: models/isolation_forest_benign.pkl")

# ==========================================
# SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("BENIGN-ONLY TRAINING COMPLETED")
print("=" * 60)

print("Algorithm       : Isolation Forest")
print("Training Data   : Benign traffic only")
print("Training Rows   :", X_train.shape[0])
print("Features        :", X_train.shape[1])