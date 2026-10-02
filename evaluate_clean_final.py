import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# STEP 1: LOAD UNSEEN TEST DATA
# ============================================================

print("=" * 70)
print("LOADING UNSEEN TEST DATA")
print("=" * 70)

df = pd.read_parquet(
    "dataset/test_processed_final.parquet"
)

X_test = df.drop(
    "Label",
    axis=1
)

y_test = df["Label"]

print("Test dataset:", df.shape)


# ============================================================
# STEP 2: LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING FINAL ISOLATION FOREST")
print("=" * 70)

model = joblib.load(
    "models/isolation_forest_final_clean.pkl"
)

print("Model loaded successfully!")


# ============================================================
# STEP 3: PREDICT
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

predictions = model.predict(
    X_test
)

print("Predictions generated!")


# ============================================================
# STEP 4: CONVERT LABELS
# ============================================================

# Benign = Normal
# Everything else = Attack

actual = y_test.apply(
    lambda label: 1
    if label == "Benign"
    else -1
)


# ============================================================
# STEP 5: METRICS
# ============================================================

accuracy = accuracy_score(
    actual,
    predictions
)

precision = precision_score(
    actual,
    predictions,
    pos_label=-1,
    zero_division=0
)

recall = recall_score(
    actual,
    predictions,
    pos_label=-1,
    zero_division=0
)

f1 = f1_score(
    actual,
    predictions,
    pos_label=-1,
    zero_division=0
)


# ============================================================
# STEP 6: DISPLAY PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEAN MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")


# ============================================================
# STEP 7: CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        actual,
        predictions,
        labels=[-1, 1],
        target_names=[
            "Attack",
            "Normal"
        ],
        zero_division=0
    )
)


# ============================================================
# STEP 8: CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    actual,
    predictions,
    labels=[-1, 1]
)

print(cm)

print("\nRows    = Actual")
print("Columns = Predicted")

print("\n              Predicted")
print("              Attack  Normal")

print(
    f"Actual Attack  {cm[0][0]:6d}  {cm[0][1]:6d}"
)

print(
    f"Actual Normal  {cm[1][0]:6d}  {cm[1][1]:6d}"
)


# ============================================================
# STEP 9: FINISH
# ============================================================

print("\n" + "=" * 70)
print("CLEAN FINAL EVALUATION COMPLETED")
print("=" * 70)