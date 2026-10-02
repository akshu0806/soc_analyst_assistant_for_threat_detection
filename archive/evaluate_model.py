import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

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
# STEP 2: Separate Features and Labels
# ==========================================

X = df.drop("Label", axis=1)
y = df["Label"]

print("\nFeatures Shape:", X.shape)
print("Labels Shape:", y.shape)

# ==========================================
# STEP 3: Load Trained Isolation Forest
# ==========================================

print("\n" + "=" * 60)
print("LOADING TRAINED MODEL")
print("=" * 60)

model = joblib.load("models/isolation_forest_benign.pkl")

print("Isolation Forest loaded successfully!")

# ==========================================
# STEP 4: Generate Predictions
# ==========================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

predictions = model.predict(X)

print("Predictions generated successfully!")

# Isolation Forest output:
#  1  = Normal
# -1  = Anomaly

print("\nModel Prediction Distribution:")
print(pd.Series(predictions).value_counts())

# ==========================================
# STEP 5: Convert Actual Labels
# ==========================================

# CIC-IDS2017:
# Benign = Normal
# Any other label = Attack

actual = y.apply(
    lambda label: 1 if label == "Benign" else -1
)

# ==========================================
# STEP 6: Calculate Metrics
# ==========================================

accuracy = accuracy_score(actual, predictions)

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
# STEP 7: Display Results
# ==========================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
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
print(f"Actual Attack  {cm[0][0]:6d}  {cm[0][1]:6d}")
print(f"Actual Normal  {cm[1][0]:6d}  {cm[1][1]:6d}")

# ==========================================
# STEP 10: Completion Message
# ==========================================

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)