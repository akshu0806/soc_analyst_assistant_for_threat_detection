import pandas as pd
import requests
import joblib

# Configurations
API_URL = "http://127.0.0.1:5000/predict"
DATASETS = {
    "DDoS": "dataset/DDoS-Friday-no-metadata.parquet",
    "DoS": "dataset/DoS-Wednesday-no-metadata.parquet",
    "PortScan": "dataset/Portscan-Friday-no-metadata.parquet",
    "Bot": "dataset/Botnet-Friday-no-metadata.parquet",
    "BruteForce": "dataset/Bruteforce-Tuesday-no-metadata.parquet",
    "Infiltration": "dataset/Infiltration-Thursday-no-metadata.parquet",
    "WebAttacks": "dataset/WebAttacks-Thursday-no-metadata.parquet"
}

#Load Models
print("Loading models...")
scaler = joblib.load(
    "models/scaler_final.pkl"
)
feature_columns = joblib.load(
    "models/feature_columns.pkl"
)
isolation_forest = joblib.load(
    "models/isolation_forest_final_clean.pkl"
)
print("Models loaded successfully!")

#Testing each dataset
for dataset_name, dataset_path in DATASETS.items():
    print("\n")
    print("=" * 70)
    print(f"TESTING DATASET: {dataset_name}")
    print("=" * 70)
    #Load dataset
    try:

        df = pd.read_parquet(
            dataset_path
        )

    except Exception as e:

        print(
            f"Could not load dataset: {e}"
        )

        continue


    print(
        f"Dataset rows: {len(df)}"
    )


    # --------------------------------------------------------
    # Select actual attack records
    # --------------------------------------------------------

    attack_df = df[
        df["Label"] != "Benign"
    ].copy()


    print(
        f"Actual attack records: {len(attack_df)}"
    )


    if len(attack_df) == 0:

        print(
            "No attack records found."
        )

        continue


    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_attack = attack_df[
        feature_columns
    ]


    X_scaled = scaler.transform(
    X_attack
)

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=feature_columns
    )

    predictions = isolation_forest.predict(
        X_scaled_df
    )


    # -1 = Attack
    #  1 = Normal

    detected_positions = (
        predictions == -1
    )


    # --------------------------------------------------------
    # Check if any attacks were detected
    # --------------------------------------------------------

    detected_indices = attack_df.index[
        detected_positions
    ]


    if len(detected_indices) == 0:

        print(
            "No actual attack was detected "
            "by Isolation Forest in this dataset."
        )

        continue


    # --------------------------------------------------------
    # Select first correctly detected attack
    # --------------------------------------------------------

    original_index = detected_indices[0]


    sample_row = df.loc[
        original_index
    ]


    actual_label = sample_row[
        "Label"
    ]


    print("\nATTACK DETECTED BY ISOLATION FOREST")

    print(
        f"Record Index : {original_index}"
    )

    print(
        f"Actual Label : {actual_label}"
    )


    # --------------------------------------------------------
    # Send ONLY this record to Flask
    # --------------------------------------------------------

    sample = sample_row.drop(
        "Label"
    ).to_dict()


    try:

        response = requests.post(
            API_URL,
            json=sample,
            timeout=120
        )


    except Exception as e:

        print(
            f"Flask API error: {e}"
        )

        continue


    # --------------------------------------------------------
    # Process Flask response
    # --------------------------------------------------------

    if response.status_code != 200:

        print(
            "Flask returned an error:"
        )

        print(
            response.json()
        )

        continue


    result = response.json()


    print("\n")
    print("-" * 70)

    print(
        f"Prediction  : {result.get('prediction')}"
    )

    print(
        f"Attack Type : {result.get('attack_type')}"
    )

    print(
        f"Severity    : {result.get('severity')}"
    )

    print(
        f"MITRE       : {result.get('mitre')}"
    )

    print("-" * 70)


print("\n")
print("=" * 70)
print("MULTI-DATASET TEST COMPLETED")
print("=" * 70)