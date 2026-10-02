import pandas as pd
import joblib
import os

from sklearn.ensemble import IsolationForest


# ============================================================
# STEP 1: LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("LOADING CLEANTRAINING DATA")
print("=" * 70)

df = pd.read_parquet(
    "dataset/train_processed_final.parquet"
)

print("Training dataset:", df.shape)


# ============================================================
# STEP 2: SELECT BENIGN TRAFFIC ONLY
# ============================================================

print("\n" + "=" * 70)
print("SELECTING BENIGN TRAINING TRAFFIC")
print("=" * 70)

benign_df = df[
    df["Label"] == "Benign"
]

X_train = benign_df.drop(
    "Label",
    axis=1
)

print("Benign training records:", len(X_train))

print("Features:", X_train.shape[1])


# ============================================================
# STEP 3: CREATE ISOLATION FOREST
# ============================================================

print("\n" + "=" * 70)
print("CREATING ISOLATION FOREST")
print("=" * 70)

model = IsolationForest(
    n_estimators=100,
    contamination=0.10,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# STEP 4: TRAIN MODEL
# ============================================================

print("\nTraining Isolation Forest...")
print("Training data: BENIGN ONLY")

model.fit(
    X_train
)

print("Training completed!")


# ============================================================
# STEP 5: SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/isolation_forest_final_clean.pkl"
)

print("\nModel saved:")
print("models/isolation_forest_final_clean.pkl")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ISOLATION FOREST TRAINING COMPLETED")
print("=" * 70)

print("Training strategy : Benign only")
print("Contamination     : 0.10")
print("Trees             : 100")
print("Training records  :", len(X_train))
print("Features          :", X_train.shape[1])