"""Escalation Engine: Determines auto-handle vs. human escalation with stated reasons."""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from src.config import ESCALATION_REASONS, ESCALATION_EXPLANATIONS


@dataclass
class EscalationDecision:
    should_escalate: bool
    reason_code: str
    explanation: str
    confidence: float
    triggers: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_escalate": self.should_escalate,
            "reason_code": self.reason_code,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "triggers": self.triggers
        }


class EscalationEngine:
    """
    Evaluates whether an incoming customer inquiry should be auto-handled
    by the AI agent or escalated to a human specialist.
    """

    POLICY_RULES = {
        "SECURITY_AND_PII": [
            r"\bapple\s*id\b", r"\bicloud\b", r"\bpassword\b", r"\blocked\b",
            r"\bdisabled\b", r"\bsecurity\b", r"\b2fa\b", r"\btwo-?factor\b",
            r"\bverification\s*code\b", r"\bsign\s*in\b", r"\blogin\b", r"\bhacked\b",
            r"\bactivation\s*lock\b", r"\bstolen\b", r"\blost\s*phone\b"
        ],
        "HARDWARE_DAMAGE": [
            r"\bcrack(ed)?\b", r"\bshatter(ed)?\b", r"\bwater\s*damage\b",
            r"\bswollen\b", r"\bbulging\b", r"\bdropped\s*in\b", r"\bbroken\s*glass\b",
            r"\bphysical(ly)?\s*broken\b", r"\bspark\b", r"\bbent\b", r"\brepair\s*cost\b"
        ],
        "BILLING_DISPUTE": [
            r"\brefund\b", r"\bunauthorized\b", r"\baccidental\s*purchase\b",
            r"\bcharg(ed|es)?\s*(twice|again)\b", r"\bmoney\s*back\b", r"\bstolen\s*card\b",
            r"\bbill(ed|ing)?\b", r"\bcancel\s*subscription\b", r"\bwrong\s*amount\b"
        ],
        "COMPLEX_UNRESOLVED": [
            r"\balready\s*tried\b", r"\balready\s*restarted\b", r"\bdid\s*everything\b",
            r"\breset\s*(settings|network)\s*and\s*nothing\b", r"\bstill\s*not\s*working\b",
            r"\btried\s*that\b", r"\bdid\s*that\s*already\b", r"\bnothing\s*works\b",
            r"\bstill\s*happening\b", r"\bdone\s*that\b", r"\bthird\s*time\b"
        ],
        "HIGH_SENTIMENT_RISK": [
            r"\bworst\s*(phone|company|customer\s*service)\b", r"\bswitch(ing)?\s*to\s*(samsung|android)\b",
            r"\bgarbage\b", r"\btrash\b", r"\buseless\b", r"\bunacceptable\b", r"\bfurious\b",
            r"\bscam\b", r"\blawsuit\b", r"\blawyer\b", r"\bdone\s*with\s*apple\b", r"\bhate\s*this\s*phone\b"
        ]
    }

    def evaluate(self, text: str, predicted_intent: str) -> EscalationDecision:
        lower = text.lower()
        matched_triggers = []

        # Priority 1: High Sentiment Risk / Churn / Legal threat
        sentiment_triggers = [p for p in self.POLICY_RULES["HIGH_SENTIMENT_RISK"] if re.search(p, lower)]
        if sentiment_triggers:
            return EscalationDecision(
                should_escalate=True,
                reason_code="HIGH_SENTIMENT_RISK",
                explanation=ESCALATION_EXPLANATIONS["HIGH_SENTIMENT_RISK"],
                confidence=0.95,
                triggers=sentiment_triggers
            )

        # Priority 2: Security and PII (Apple ID / Passwords / 2FA)
        sec_triggers = [p for p in self.POLICY_RULES["SECURITY_AND_PII"] if re.search(p, lower)]
        if sec_triggers or predicted_intent == "apple_id_account_security":
            return EscalationDecision(
                should_escalate=True,
                reason_code="SECURITY_AND_PII",
                explanation=ESCALATION_EXPLANATIONS["SECURITY_AND_PII"],
                confidence=0.98,
                triggers=sec_triggers or ["intent:apple_id_account_security"]
            )

        # Priority 3: Hardware Physical Damage (requires repair / Genius Bar)
        hw_triggers = [p for p in self.POLICY_RULES["HARDWARE_DAMAGE"] if re.search(p, lower)]
        if hw_triggers:
            return EscalationDecision(
                should_escalate=True,
                reason_code="HARDWARE_DAMAGE",
                explanation=ESCALATION_EXPLANATIONS["HARDWARE_DAMAGE"],
                confidence=0.96,
                triggers=hw_triggers
            )

        # Priority 4: Financial & Billing Disputes
        bill_triggers = [p for p in self.POLICY_RULES["BILLING_DISPUTE"] if re.search(p, lower)]
        if bill_triggers or (predicted_intent == "billing_subscriptions_appstore" and any(k in lower for k in ["refund", "charge", "card", "pay", "order"])):
            return EscalationDecision(
                should_escalate=True,
                reason_code="BILLING_DISPUTE",
                explanation=ESCALATION_EXPLANATIONS["BILLING_DISPUTE"],
                confidence=0.94,
                triggers=bill_triggers or ["intent:billing_subscriptions_appstore"]
            )

        # Priority 5: Complex / Repeat Unresolved Failures
        unres_triggers = [p for p in self.POLICY_RULES["COMPLEX_UNRESOLVED"] if re.search(p, lower)]
        if unres_triggers:
            return EscalationDecision(
                should_escalate=True,
                reason_code="COMPLEX_UNRESOLVED",
                explanation=ESCALATION_EXPLANATIONS["COMPLEX_UNRESOLVED"],
                confidence=0.90,
                triggers=unres_triggers
            )

        # Priority 6: Safe to Auto-Handle with Grounded Troubleshooting
        return EscalationDecision(
            should_escalate=False,
            reason_code="KNOWN_TROUBLESHOOTING_CAN_AUTOHANDLE",
            explanation=ESCALATION_EXPLANATIONS["KNOWN_TROUBLESHOOTING_CAN_AUTOHANDLE"],
            confidence=0.88,
            triggers=["troubleshooting_workflow_available"]
        )
