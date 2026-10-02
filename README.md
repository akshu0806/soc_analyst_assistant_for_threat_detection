# AI-Powered SOC Analyst Assistant for Intelligent Threat Detection and Incident Response

An AI-powered Security Operations Center (SOC) assistant that detects anomalous network traffic, classifies potential cyberattacks, maps threats to the MITRE ATT&CK framework, and generates AI-assisted incident analysis through a local Large Language Model (LLM).

## Project Overview

Security Operations Centers process large volumes of network traffic and security alerts. Manually investigating these alerts can be time-consuming, especially when analysts must identify the attack type, understand its potential impact, and determine suitable response actions.

This project addresses this challenge through a hybrid cybersecurity and machine learning system that combines anomaly detection, supervised attack classification, threat intelligence mapping, and AI-generated incident analysis in an interactive dashboard.

The system uses the CIC-IDS2017 dataset for model development and evaluation.

## Key Features

- **Network Anomaly Detection:** Uses Isolation Forest to identify suspicious network flows.
- **Multi-Class Attack Classification:** Uses Random Forest to classify detected attacks into known categories.
- **MITRE ATT&CK Mapping:** Associates detected attack categories with relevant techniques and tactics.
- **AI-Assisted Incident Analysis:** Uses Llama 3.2 through Ollama to generate explanations, potential impacts, and recommended response actions.
- **REST API:** Uses Flask to serve predictions and integrate the machine learning pipeline.
- **Interactive SOC Dashboard:** Uses Streamlit to display predictions, alerts, incident details, and analysis results.
- **Batch Simulation:** Supports analyzing multiple dataset records.
- **Alert Monitoring:** Displays session statistics, alert history, and visual summaries.

## System Architecture

```text
       CIC-IDS2017 Dataset
                |
                v
       Data Preprocessing
       Cleaning and Scaling
                |
                v
          Flask REST API
                |
                v
          Isolation Forest
                |
         +------+------+
         |             |
         v             v
    Normal Traffic   Anomalous Traffic
         |             |
         v             v
     Normal Result  Random Forest
                        |
                        v
                Attack Classification
                        |
                        v
                 MITRE ATT&CK
                        |
                        v
                  Llama 3.2
                   (Ollama)
                        |
                        v
              SOC Alert and Analysis
                        |
                        v
               Streamlit Dashboard
```

The Random Forest classifier and LLM analysis are used for flows flagged as anomalous. Normal flows bypass the attack-classification and LLM-analysis stages.

## Detection Pipeline

### Stage 1: Data Preprocessing

The CIC-IDS2017 dataset is cleaned before model training. The preprocessing pipeline handles duplicate records, missing values, infinite values, numeric conversion, and feature scaling.

- Original records: 2,313,810
- Records after duplicate removal: 2,199,002
- Input features: 77
- Training records: 1,759,201
- Testing records: 439,801

The dataset is split into training and testing sets, and the StandardScaler is fitted only on the training data.

### Stage 2: Anomaly Detection

Isolation Forest identifies whether a network flow is normal or anomalous.

- Normal prediction: `1`
- Anomaly prediction: `-1`

### Stage 3: Attack Classification

When a flow is flagged as anomalous, the Random Forest classifier predicts its attack category. The model supports multiple categories, including DDoS, DoS variants, PortScan, FTP and SSH brute force, Bot, Infiltration, web attacks, and Heartbleed.

### Stage 4: Threat Mapping and AI Analysis

The detected attack category is mapped to a corresponding MITRE ATT&CK technique using a predefined mapping module.

The local Llama 3.2 model then generates a concise incident analysis containing:

- Explanation of the detected attack
- Potential impact
- Recommended response
- MITRE ATT&CK context

### Stage 5: Dashboard Visualization

The Streamlit dashboard presents predictions, severity, incident details, MITRE mappings, AI-generated analysis, and session-level statistics.

## Model Evaluation

### Isolation Forest

Evaluated on the held-out test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 87.62% |
| Attack Precision | 57.30% |
| Attack Recall | 74.51% |
| Attack F1-score | 64.80% |

The model identifies a substantial proportion of attack traffic, although false positives and missed attacks remain areas for improvement.

### Random Forest

| Metric | Result |
|---|---:|
| Attack-only test accuracy | 99.71% |

**Note:** The Random Forest accuracy is measured on the attack-only test subset. It should not be interpreted as the accuracy of the complete detection system on all network traffic. Performance on rare attack categories is also affected by limited test examples.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Dataset Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Anomaly Detection | Isolation Forest |
| Attack Classification | Random Forest |
| Backend API | Flask |
| Dashboard | Streamlit |
| Local LLM | Llama 3.2 (3B) |
| LLM Runtime | Ollama |
| Threat Intelligence | MITRE ATT&CK |
| Dataset | CIC-IDS2017 |
| Version Control | Git and GitHub |

## Project Structure

The following is the logical structure of the project; adjust directory names to match the files committed to your repository.

```text
AI-SOC-Assistant/
├── backend/
│   ├── app.py
│   ├── mitre_mapping.py
│   └── llm_analyzer.py
├── dashboard/
│   └── app.py
├── dataset/
│   └── CIC-IDS2017 data files
├── models/
│   ├── feature_columns.pkl
│   ├── scaler_final.pkl
│   ├── isolation_forest_final_clean.pkl
│   └── attack_classifier.pkl
├── preprocess_final.py
├── train_isolation_final.py
├── train_attack_classifier.py
├── debug_ddos.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- Ollama
- The required CIC-IDS2017 dataset files
- The trained model artifacts

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "AI SOC Assistant"
```

Replace the placeholder with your GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Ensure `requirements.txt` includes the packages used by your implementation, such as Flask, Streamlit, pandas, NumPy, scikit-learn, PyArrow, and requests.

### 4. Prepare the dataset and model files

Place the required CIC-IDS2017 Parquet files in the expected dataset directory and make the trained model artifacts available under `models/`.

If you are starting without the trained artifacts, run the preprocessing and training scripts in the appropriate order:

```bash
python preprocess_final.py
python train_isolation_final.py
python train_attack_classifier.py
```

These commands assume the dataset is in the location expected by the scripts and the training scripts save artifacts to the `models/` directory.

### 5. Set up the local LLM

Install Ollama, then download the model:

```bash
ollama pull llama3.2:3b
```

Make sure the Ollama service is running. You can verify the model with:

```bash
ollama list
```

### 6. Start the Flask backend

From the project root, open a terminal and run:

```bash
python backend/app.py
```

The API should be available at:

```text
http://127.0.0.1:5000
```

### 7. Start the Streamlit dashboard

Open a second terminal, activate the virtual environment if needed, and run:

```bash
streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit in your browser.

**Important:** The commands and paths above assume the directory structure shown in this README. Update them if your actual repository uses different filenames or folders.

## 🔌 API Usage

The Flask backend provides a prediction endpoint:

```text
POST /predict
```

It accepts a JSON object containing the network-flow features expected by the trained models.

Example request structure:

```json
{
  "feature_1": 0.0,
  "feature_2": 0.0
}
```

The example is illustrative only. A real request must include all required features with their actual names and values.

The response includes fields such as:

- `alert_status`
- `prediction`
- `attack_type`
- `severity`
- `timestamp`
- `description`
- `recommended_action`
- `mitre`
- `llm_analysis`

## 🧪 Testing

The project can be tested using selected records from the CIC-IDS2017 dataset.

Suggested validation scenarios:

1. Submit a benign network-flow record and verify that it is classified as normal.
2. Submit a known DDoS record and verify the attack prediction and MITRE mapping.
3. Test other attack categories represented in the dataset.
4. Verify the JSON response returned by the Flask API.
5. Confirm that the Streamlit dashboard displays the prediction and incident analysis.
6. Check that the local LLM is available and responding through Ollama.

## Limitations

- The current implementation is evaluated using CIC-IDS2017 data and is not a production-validated live network monitoring system.
- Isolation Forest can generate false positives and miss some attacks.
- The dataset has class imbalance, with limited examples for certain attack categories.
- MITRE ATT&CK mappings are predefined and may not fully describe every observed behavior.
- LLM-generated explanations and response recommendations require analyst verification.
- Live packet capture, automated containment, and continuous model retraining are not implemented as complete production capabilities.

## Future Enhancements

- Integrate live network traffic collection.
- Improve anomaly detection and reduce false positives.
- Address class imbalance and improve rare-attack classification.
- Expand MITRE ATT&CK mapping and threat context.
- Add retrieval-augmented generation (RAG) for security knowledge.
- Introduce persistent alert storage and incident tracking.
- Add analyst feedback, model monitoring, and scheduled retraining.
- Improve authentication, logging, and deployment security.

## Project Objective

The objective is to develop an integrated, AI-assisted SOC prototype that combines machine learning-based network threat detection with threat intelligence and natural-language incident analysis. It aims to help security analysts interpret alerts more efficiently while keeping the final investigation and response decisions under human control.

## Contributors

- Akshitha Sivakumar
- A. Srinidhi

## References

- Sharafaldin, I., Lashkari, A. H., and Ghorbani, A. A. “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.” *International Conference on Information Systems Security and Privacy (ICISSP)*, 2018.
- Liu, F. T., Ting, K. M., and Zhou, Z.-H. “Isolation Forest.” *IEEE International Conference on Data Mining (ICDM)*, 2008.
- Breiman, L. “Random Forests.” *Machine Learning*, vol. 45, pp. 5–32, 2001.
- MITRE. [MITRE ATT&CK](https://attack.mitre.org/).
- Scikit-learn. [Machine Learning Documentation](https://scikit-learn.org/stable/).
- Ollama. [Llama 3.2 Model Library](https://ollama.com/library/llama3.2).

---

*This project was developed for academic and research purposes. It is a prototype and should not be used as the sole security control in a production environment.*
