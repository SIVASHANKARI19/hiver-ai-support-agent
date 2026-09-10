"""System configuration, taxonomies, and paths for Apple Support Agent."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CORPUS_PATH = PROCESSED_DATA_DIR / "historical_resolutions.json"
GOLDEN_SET_PATH = DATA_DIR / "golden_eval_set.json"
GOLDEN_SET_CSV = DATA_DIR / "golden_eval_set.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

INTENTS = [
    "battery_power",
    "software_update_os",
    "connectivity_network",
    "hardware_display_audio",
    "apple_id_account_security",
    "billing_subscriptions_appstore",
    "general_inquiry_feedback"
]

INTENT_DESCRIPTIONS = {
    "battery_power": "Battery draining rapidly, charging issues, overheating, or battery health degradation.",
    "software_update_os": "iOS/macOS update installations, boot loops, system crashes, keyboard glitches, or app freezing.",
    "connectivity_network": "Wi-Fi dropping, Bluetooth audio/car pairing, cellular No Service, LTE or AirDrop issues.",
    "hardware_display_audio": "Cracked screen, display touch unresponsiveness, speaker/microphone faults, or camera hardware.",
    "apple_id_account_security": "Forgotten Apple ID password, 2FA codes, locked/disabled accounts, or iCloud login security.",
    "billing_subscriptions_appstore": "Unrecognized card charges, App Store refund requests, subscriptions, or pre-order status.",
    "general_inquiry_feedback": "Customer venting, sarcasm, feature inquiries, store hours, or general brand feedback."
}

ESCALATION_REASONS = [
    "SECURITY_AND_PII",
    "HARDWARE_DAMAGE",
    "BILLING_DISPUTE",
    "HIGH_SENTIMENT_RISK",
    "COMPLEX_UNRESOLVED",
    "KNOWN_TROUBLESHOOTING_CAN_AUTOHANDLE"
]

ESCALATION_EXPLANATIONS = {
    "SECURITY_AND_PII": "Requires private authentication or Apple ID identity verification which cannot be handled in public tweets.",
    "HARDWARE_DAMAGE": "Physical damage or hardware component failure requires an in-person Genius Bar visit or mail-in repair.",
    "BILLING_DISPUTE": "Financial transaction review and refund issuance require secure billing system access.",
    "HIGH_SENTIMENT_RISK": "Severe customer dissatisfaction, churn threat, or legal escalation requiring compassionate human de-escalation.",
    "COMPLEX_UNRESOLVED": "Customer has already attempted standard frontline troubleshooting without success, requiring Tier-2 investigation.",
    "KNOWN_TROUBLESHOOTING_CAN_AUTOHANDLE": "Standard Apple diagnostic steps, settings guidance, or self-service links can resolve the query automatically."
}

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_TOP_K = 3
SIMILARITY_THRESHOLD = 0.40
MAX_REPLY_CHARS = 280
