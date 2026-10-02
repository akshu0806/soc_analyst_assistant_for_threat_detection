import pandas as pd
import requests


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_parquet(
    "dataset/DDoS-Friday-no-metadata.parquet"
)


# ============================================================
# CHECK FIRST 1000 RECORDS
# ============================================================

print("Checking first 1000 records...\n")


for index in range(min(1000, len(df))):

    # Get one network flow
    sample = df.drop(
        "Label",
        axis=1
    ).iloc[index].to_dict()

    # Send to Flask API
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json=sample,
        timeout=120
    )

    data = response.json()


    # Check for API error
    if response.status_code != 200:

        print(
            f"Record {index}: API Error"
        )

        print(data)

        continue


    # Get prediction
    prediction = data.get(
        "prediction"
    )


    # ========================================================
    # ATTACK FOUND
    # ========================================================

    if prediction == "Attack":

        print("=" * 60)

        print(
            f"ATTACK FOUND AT RECORD: {index}"
        )

        print("=" * 60)

        print(
            "Actual Label:",
            df.iloc[index]["Label"]
        )

        print(
            "Prediction:",
            data.get("prediction")
        )

        print(
            "Attack Type:",
            data.get("attack_type")
        )

        print(
            "Severity:",
            data.get("severity")
        )

        print(
            "MITRE ATT&CK:",
            data.get("mitre")
        )

        print(
            "\nLLM ANALYSIS:"
        )

        print(
            data.get("llm_analysis")
        )

        print("=" * 60)

        break


else:

    print(
        "No attack detected in the first 1000 records."
    )