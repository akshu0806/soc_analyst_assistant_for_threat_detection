import requests


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.2:3b"


# ============================================================
# AI SOC INCIDENT ANALYZER
# ============================================================

def analyze_incident(
    attack_type,
    severity,
    prediction,
    mitre_info
):

    prompt = f"""
You are an AI Security Operations Center (SOC) Analyst.

Analyze this security alert.

Prediction: {prediction}
Attack Type: {attack_type}
Severity: {severity}

MITRE ATT&CK Mapping:
Technique ID: {mitre_info["technique_id"]}
Technique Name: {mitre_info["technique_name"]}
Tactic: {mitre_info["tactic"]}

Provide the analysis using exactly these four sections:

1. Explanation
Explain what this detected attack means in simple SOC analyst language.

2. Potential Impact
Explain the possible impact of this attack on a system or network.

3. Recommended Response
Give practical defensive actions that a SOC analyst should take.

4. MITRE ATT&CK
Explain why the provided MITRE ATT&CK technique is relevant to this attack.
Do not invent a different MITRE ATT&CK technique.

Keep the response concise and suitable for a SOC dashboard.
"""


    # ========================================================
    # SEND REQUEST TO LOCAL OLLAMA
    # ========================================================

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )


    # ========================================================
    # CHECK RESPONSE
    # ========================================================

    response.raise_for_status()


    result = response.json()


    # ========================================================
    # RETURN AI ANALYSIS
    # ========================================================

    return result["response"]