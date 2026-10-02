import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# STEP 1: LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

df = pd.read_parquet(
    "dataset/train_processed_final.parquet"
)

print("Training data:", df.shape)


# ============================================================
# STEP 2: REMOVE BENIGN TRAFFIC
# ============================================================

print("\n" + "=" * 70)
print("SELECTING ATTACK TRAFFIC")
print("=" * 70)

attack_df = df[
    df["Label"] != "Benign"
].copy()

print("Attack records:", len(attack_df))


# ============================================================
# STEP 3: FEATURES AND ATTACK LABEL
# ============================================================

X_train = attack_df.drop(
    "Label",
    axis=1
)

y_train = attack_df["Label"]


print("Features:", X_train.shape)
print("Attack classes:", y_train.nunique())


# ============================================================
# STEP 4: CREATE CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("CREATING ATTACK CLASSIFIER")
print("=" * 70)

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)


# ============================================================
# STEP 5: TRAIN
# ============================================================

print("\nTraining attack classifier...")

classifier.fit(
    X_train,
    y_train
)

print("Attack classifier trained successfully!")


# ============================================================
# STEP 6: SAVE
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    classifier,
    "models/attack_classifier.pkl"
)

print("\nClassifier saved:")
print("models/attack_classifier.pkl")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ATTACK CLASSIFIER TRAINING COMPLETED")
print("=" * 70)

print("Algorithm : Random Forest")
print("Training  : Known attacks only")
print("Classes   :", y_train.nunique())