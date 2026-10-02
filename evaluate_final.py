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

# ==========================================
# STEP 1: Load unseen test dataset
# ==========================================

print("=" * 60)
print("LOADING UNSEEN TEST DATA")
print("=" * 60)

df = pd.read_parquet("dataset/test_dataset.parquet")

print("Test dataset loaded successfully!")
print("Test dataset shape:", df.shape)

# ==========================================
# STEP 2: Separate features and labels
# ==========================================

X_test = df.drop("Label", axis=1)
y_test = df["Label"]

print("\nTest features:", X_test.shape)
print("Test labels:", y_test.shape)

# ==========================================
# STEP 3: Load final model
# ==========================================

print("\n" + "=" * 60)
print("LOADING FINAL MODEL")
print("=" * 60)

model = joblib.load(
    "models/isolation_forest_final.pkl"
)

print("Final Isolation Forest loaded successfully!")

# ==========================================
# STEP 4: Generate predictions
# ==========================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

predictions = model.predict(X_test)

print("Predictions generated successfully!")

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

print("\nPrediction Distribution:")
print(pd.Series(predictions).value_counts())

# ==========================================
# STEP 5: Convert actual labels
# ==========================================

# Benign = Normal (1)
# Everything else = Attack (-1)

actual = y_test.apply(
    lambda label: 1 if label == "Benign" else -1
)

# ==========================================
# STEP 6: Calculate metrics
# ==========================================

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

# ==========================================
# STEP 7: Display performance
# ==========================================

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# ==========================================
# STEP 8: Classification Report
# ==========================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        actual,
        predictions,
        labels=[-1, 1],
        target_names=["Attack", "Normal"],
        zero_division=0
    )
)

# ==========================================
# STEP 9: Confusion Matrix
# ==========================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    actual,
    predictions,
    labels=[-1, 1]
)

print(cm)

print("\nRows = Actual")
print("Columns = Predicted")

print("\n              Predicted")
print("              Attack  Normal")
print(
    f"Actual Attack  {cm[0][0]:6d}  {cm[0][1]:6d}"
)
print(
    f"Actual Normal  {cm[1][0]:6d}  {cm[1][1]:6d}"
)

# ==========================================
# STEP 10: Final interpretation
# ==========================================

print("\n" + "=" * 60)
print("FINAL EVALUATION COMPLETED")
print("=" * 60)

print("The model was trained only on benign traffic.")
print("Performance was evaluated on unseen test data.")