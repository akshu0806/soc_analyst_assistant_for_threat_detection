# backend/mitre_mapping.py

MITRE_MAPPING = {
    "DDoS": {
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "tactic": "Impact"
    },

    "DoS Hulk": {
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "tactic": "Impact"
    },

    "DoS GoldenEye": {
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "tactic": "Impact"
    },

    "DoS slowloris": {
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "tactic": "Impact"
    },

    "DoS Slowhttptest": {
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "tactic": "Impact"
    },

    "PortScan": {
        "technique_id": "T1046",
        "technique_name": "Network Service Scanning",
        "tactic": "Discovery"
    },

    "FTP-Patator": {
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "tactic": "Credential Access"
    },

    "SSH-Patator": {
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "tactic": "Credential Access"
    },

    "Bot": {
        "technique_id": "T1071",
        "technique_name": "Application Layer Protocol",
        "tactic": "Command and Control"
    },

    "Infiltration": {
        "technique_id": "T1078",
        "technique_name": "Valid Accounts",
        "tactic": "Initial Access"
    },

    "Web Attack Brute Force": {
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "tactic": "Credential Access"
    },

    "Web Attack XSS": {
        "technique_id": "T1189",
        "technique_name": "Drive-by Compromise",
        "tactic": "Initial Access"
    },

    "Web Attack SQL Injection": {
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "tactic": "Initial Access"
    },

    "Heartbleed": {
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "tactic": "Initial Access"
    }
}


def get_mitre_mapping(attack_type):
    """
    Return MITRE ATT&CK information for a detected attack type.
    """

    return MITRE_MAPPING.get(
        attack_type,
        {
            "technique_id": "Unknown",
            "technique_name": "Requires Further Investigation",
            "tactic": "Unknown"
        }
    )