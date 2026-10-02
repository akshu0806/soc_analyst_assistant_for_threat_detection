
import pandas as pd
import requests

df = pd.read_parquet(
    "dataset/DDoS-Friday-no-metadata.parquet"
)

# Inspect the labels available in this dataset
print("Available labels:")
print(df["Label"].value_counts().head(10))

# Find rows labelled exactly DDoS
mask = (
    df["Label"]
    .astype(str)
    .str.strip()
    .str.upper()
    .eq("DDOS")
)

if not mask.any():
    print("\nNo exact DDoS label found.")
    print("Check the available labels printed above.")
else:
    row_index = df.index[mask][0]
    row = df.loc[row_index]

    print("\nTesting row:", row_index)
    print("Actual label:", row["Label"])

    features = row.drop(labels=["Label"]).to_dict()

    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json=features,
        timeout=180
    )

    print("HTTP status:", response.status_code)

    if response.ok:
        result = response.json()
        print("Prediction:", result.get("prediction"))
        print("Attack type:", result.get("attack_type"))
        print("Severity:", result.get("severity"))
        print("MITRE:", result.get("mitre"))
    else:
        print("API error:", response.text)
