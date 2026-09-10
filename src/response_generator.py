"""Response Generator: Drafts replies grounded in historical brand resolutions."""

from typing import List, Dict, Any, Optional
from src.escalation_engine import EscalationDecision
from src.config import MAX_REPLY_CHARS


class GroundedResponseGenerator:
    """
    Generates Apple Support brand replies grounded in historical resolution
    patterns, with safety guardrails and tone compliance.
    """

    OFFICIAL_ESCALATION_TEMPLATES = {
        "SECURITY_AND_PII": "To protect your account security and Apple ID credentials privately, please join us in a DM so we can assist: https://apple.co/dm",
        "HARDWARE_DAMAGE": "We want to make sure your device is inspected safely. Please visit an Apple Store Genius Bar or schedule a repair: https://apple.co/repair",
        "BILLING_DISPUTE": "We'd like to review this charge with you securely. Please connect with us in a DM so we can look into your account: https://apple.co/dm",
        "HIGH_SENTIMENT_RISK": "We understand your frustration and are truly sorry for this experience. We'd like to make this right—please join us in DM: https://apple.co/dm",
        "COMPLEX_UNRESOLVED": "Since those steps have not resolved it, let's investigate deeper. Please send us a DM with your iOS version and model: https://apple.co/dm",
        "KNOWN_TROUBLESHOOTING_CAN_AUTOHANDLE": "We are here to help. Have you tried restarting your device? Check Settings > General > Software Update to ensure you're current."
    }

    INTENT_TROUBLESHOOTING_PLAYBOOKS = {
        "battery_power": "We'd like to help with your battery. Check Settings > Battery > Battery Health to see maximum capacity. A restart can also clear background tasks.",
        "software_update_os": "We're here to help get this sorted. Which iOS version is installed in Settings > General > About? Have you tried a force restart?",
        "connectivity_network": "Let's get you connected. Try toggling Airplane Mode on and off, or go to Settings > General > Reset > Reset Network Settings.",
        "hardware_display_audio": "We'd like to help with your display and audio. Have you checked Settings > Sounds & Haptics, or tested after a clean restart?",
        "apple_id_account_security": "For account safety, visit https://iforgot.apple.com to reset your password or join us in a DM for assistance: https://apple.co/dm",
        "billing_subscriptions_appstore": "You can view your active subscriptions and purchase history in Settings > [Your Name] > Subscriptions, or DM us: https://apple.co/dm",
        "general_inquiry_feedback": "Thanks for reaching out to Apple Support. We're here for you—could you describe what device and iOS version you're using?"
    }

    def generate(
        self,
        customer_text: str,
        predicted_intent: str,
        escalation: EscalationDecision,
        retrieved_resolutions: List[Dict[str, Any]]
    ) -> str:
        """
        Draft a response grounded in retrieved historical resolutions and escalation policy.
        """
        # If escalating, use authoritative secure handoff template
        if escalation.should_escalate:
            reply = self.OFFICIAL_ESCALATION_TEMPLATES.get(
                escalation.reason_code,
                "We'd like to look closer into this with you. Please join us in a DM: https://apple.co/dm"
            )
            return reply[:MAX_REPLY_CHARS]

        # For auto-handle: evaluate if top retrieved historical reply provides a high-confidence direct match
        if retrieved_resolutions and retrieved_resolutions[0].get("similarity_score", 0) >= 0.55:
            top_reply = retrieved_resolutions[0]["brand_reply"]
            # Ensure retrieved reply is clean and on-brand
            if len(top_reply) > 20 and not any(w in top_reply.lower() for w in ["hacked", "stolen", "curse"]):
                return top_reply[:MAX_REPLY_CHARS]

        # Otherwise, synthesize using intent playbook
        playbook_reply = self.INTENT_TROUBLESHOOTING_PLAYBOOKS.get(
            predicted_intent,
            self.OFFICIAL_ESCALATION_TEMPLATES["KNOWN_TROUBLESHOOTING_CAN_AUTOHANDLE"]
        )
        return playbook_reply[:MAX_REPLY_CHARS]
