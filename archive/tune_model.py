import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# STEP 1: Load processed dataset
# ==========================================

print("Loading processed dataset...")

df = pd.read_parquet(
    "dataset/processed_dataset.parquet"
)

X = df.drop("Label", axis=1)
y = df["Label"]

# ==========================================
# STEP 2: Training data = Benign only
# ==========================================

benign_data = df[df["Label"] == "Benign"]

X_train = benign_data.drop("Label", axis=1)

print("Total records:", len(df))
print("Benign training records:", len(X_train))

# ==========================================
# STEP 3: Actual labels for evaluation
# ==========================================

# Benign = 1
# Attack = -1

actual = y.apply(
    lambda label: 1 if label == "Benign" else -1
)

# ==========================================
# STEP 4: Test different contamination values
# ==========================================

contamination_values = [
    0.01,
    0.03,
    0.05,
    0.08,
    0.10
]

results = []

for contamination in contamination_values:

    print("\n" + "=" * 60)
    print("Testing contamination:", contamination)
    print("=" * 60)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    # Train only on benign traffic
    model.fit(X_train)

    # Predict on complete dataset
    predictions = model.predict(X)

    # Calculate metrics
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

    cm = confusion_matrix(
        actual,
        predictions,
        labels=[-1, 1]
    )

    false_negatives = cm[0][1]
    false_positives = cm[1][0]

    print(f"Attack Precision : {precision:.4f}")
    print(f"Attack Recall    : {recall:.4f}")
    print(f"Attack F1        : {f1:.4f}")
    print(f"False Positives  : {false_positives}")
    print(f"False Negatives  : {false_negatives}")

    results.append({
        "contamination": contamination,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    })

# ==========================================
# STEP 5: Display comparison
# ==========================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(index=False)
)

# ==========================================
# STEP 6: Select best model by F1
# ==========================================

best_row = results_df.loc[
    results_df["f1_score"].idxmax()
]

best_contamination = best_row["contamination"]

print("\n" + "=" * 80)
print("BEST CONFIGURATION")
print("=" * 80)

print(
    f"Best contamination: {best_contamination}"
)

print(
    f"Precision: {best_row['precision']:.4f}"
)

print(
    f"Recall: {best_row['recall']:.4f}"
)

print(
    f"F1-Score: {best_row['f1_score']:.4f}"
)

# ==========================================
# STEP 7: Train final candidate model
# ==========================================

print("\nTraining final candidate model...")

final_model = IsolationForest(
    n_estimators=100,
    contamination=best_contamination,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train)

# ==========================================
# STEP 8: Save candidate model
# ==========================================

joblib.dump(
    final_model,
    "models/isolation_forest_tuned.pkl"
)

print("\nTuned model saved successfully!")
print(
    "Location: models/isolation_forest_tuned.pkl"
)