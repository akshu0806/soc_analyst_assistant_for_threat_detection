from flask import Flask, request, jsonify
from flask_cors import CORS
from llm_analyzer import analyze_incident
import pandas as pd
import joblib
from mitre_mapping import get_mitre_mapping
from pathlib import Path
from datetime import datetime


# ============================================================
# CREATE FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# LOAD SCALER
# ============================================================

SCALER_PATH = BASE_DIR / "models" / "scaler_final.pkl"

scaler = joblib.load(SCALER_PATH)


# ============================================================
# LOAD FEATURE ORDER
# ============================================================

FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"

feature_columns = joblib.load(FEATURE_PATH)


# ============================================================
# LOAD ISOLATION FOREST
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "isolation_forest_final_clean.pkl"
)

model = joblib.load(MODEL_PATH)


# ============================================================
# LOAD ATTACK CLASSIFIER
# ============================================================

CLASSIFIER_PATH = (
    BASE_DIR
    / "models"
    / "attack_classifier.pkl"
)

attack_classifier = joblib.load(CLASSIFIER_PATH)


print("Scaler loaded successfully!")
print("Feature columns loaded successfully!")
print("Isolation Forest loaded successfully!")
print("Attack classifier loaded successfully!")


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "running",
        "message": "AI SOC Assistant API is running"
    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # GET INPUT DATA
        # ----------------------------------------------------

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "No input data provided"
            }), 400


        # ----------------------------------------------------
        # CONVERT INPUT TO DATAFRAME
        # ----------------------------------------------------

        input_df = pd.DataFrame([data])


        # ----------------------------------------------------
        # REMOVE LABEL IF PROVIDED
        # ----------------------------------------------------

        if "Label" in input_df.columns:

            input_df = input_df.drop(
                "Label",
                axis=1
            )


        # ----------------------------------------------------
        # CHECK FOR MISSING FEATURES
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in feature_columns
            if feature not in input_df.columns
        ]

        if missing_features:

            return jsonify({
                "error": "Missing features",
                "missing_features": missing_features
            }), 400


        # ----------------------------------------------------
        # ARRANGE FEATURES IN TRAINING ORDER
        # ----------------------------------------------------

        input_df = input_df[
            feature_columns
        ]


        # ----------------------------------------------------
        # SCALE RAW INPUT
        # ----------------------------------------------------

        X_scaled = scaler.transform(
            input_df
        )


        # ----------------------------------------------------
        # STAGE 1: ISOLATION FOREST
        # ----------------------------------------------------

        anomaly_prediction = model.predict(
            X_scaled
        )[0]


        # ====================================================
        # NORMAL TRAFFIC
        # ====================================================

        if anomaly_prediction == 1:

            result = "Normal"
            attack_type = "None"
            severity = "Low"
            alert_status = "NORMAL"

            description = (
                "Network traffic appears normal."
            )

            recommended_action = (
                "No immediate action required."
            )

            llm_analysis = (
                "No LLM analysis required for normal traffic."
            )

            # No MITRE mapping for normal traffic
            mitre_info = {
                "technique_id": "None",
                "technique_name": "None",
                "tactic": "None"
            }


        # ====================================================
        # ANOMALOUS TRAFFIC
        # ====================================================

        else:

            result = "Attack"

            # ------------------------------------------------
            # STAGE 2: ATTACK CLASSIFICATION
            # ------------------------------------------------

            attack_prediction = attack_classifier.predict(
                X_scaled
            )[0]

            attack_type = str(
                attack_prediction
            )


            # ------------------------------------------------
            # MITRE ATT&CK MAPPING
            # ------------------------------------------------

            mitre_info = get_mitre_mapping(
                attack_type
            )


            # ------------------------------------------------
            # ALERT INFORMATION
            # ------------------------------------------------

            severity = "High"

            alert_status = "ALERT"

            description = (
                f"Anomalous network activity detected. "
                f"The traffic resembles {attack_type}."
            )

            recommended_action = (
                "Investigate the source and destination "
                "traffic, review related logs, and "
                "validate the suspected attack."
            )


            # ------------------------------------------------
            # LOCAL LLM SOC ANALYSIS
            # ------------------------------------------------

            llm_analysis = analyze_incident(
                attack_type=attack_type,
                severity=severity,
                prediction=result,
                mitre_info=mitre_info
            )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp = datetime.now().isoformat()


        # ====================================================
        # RETURN RESPONSE
        # ====================================================

        return jsonify({

            "alert_status": alert_status,

            "prediction": result,

            "attack_type": attack_type,

            "severity": severity,

            "timestamp": timestamp,

            "description": description,

            "recommended_action": recommended_action,

            "mitre": mitre_info,

            "llm_analysis": llm_analysis

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
    host="127.0.0.1",
    port=5000,
    debug=True,
    use_reloader=False
)