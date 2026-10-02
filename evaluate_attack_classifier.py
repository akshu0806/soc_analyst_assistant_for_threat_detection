import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    accuracy_score
)


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_parquet(
    "dataset/test_processed_final.parquet"
)


# ============================================================
# SELECT ATTACKS ONLY
# ============================================================

attack_df = df[
    df["Label"] != "Benign"
].copy()


X_test = attack_df.drop(
    "Label",
    axis=1
)

y_test = attack_df["Label"]


# ============================================================
# LOAD CLASSIFIER
# ============================================================

classifier = joblib.load(
    "models/attack_classifier.pkl"
)


# ============================================================
# PREDICT ATTACK TYPE
# ============================================================

predictions = classifier.predict(
    X_test
)


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 70)
print("ATTACK CLASSIFIER PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy: {accuracy:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)