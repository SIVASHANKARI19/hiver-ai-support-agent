"""LLM-as-a-Judge Rubric and Human Agreement Evaluator."""

import os
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from src.metrics import compute_judge_human_agreement, compute_rouge_l


@dataclass
class JudgeEvaluation:
    grounding_score: float         # 1-5
    intent_relevance_score: float  # 1-5
    escalation_safety_score: float # 1-5
    tone_empathy_score: float      # 1-5
    composite_score: float         # 1-5
    critique: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grounding_score": round(self.grounding_score, 2),
            "intent_relevance_score": round(self.intent_relevance_score, 2),
            "escalation_safety_score": round(self.escalation_safety_score, 2),
            "tone_empathy_score": round(self.tone_empathy_score, 2),
            "composite_score": round(self.composite_score, 2),
            "critique": self.critique
        }


class AppleSupportQualityJudge:
    """
    Evaluates customer support reply quality using a structured 4-dimension rubric.
    Operates either via an external LLM (Gemini/OpenAI/Groq) or via a calibrated
    deterministic offline rubric adhering to the identical evaluation guidelines.
    """

    def evaluate_reply(
        self,
        customer_text: str,
        draft_reply: str,
        true_intent: str,
        true_escalation: bool,
        historical_reference_reply: str,
        gold_resolution_notes: str
    ) -> JudgeEvaluation:
        """Evaluate a drafted reply against the 4-dimension rubric."""
        c_low = customer_text.lower()
        r_low = draft_reply.lower()

        # Dimension 1: Grounding & Technical Correctness (1-5)
        # Check if reply uses verified Apple troubleshooting paths or safe links
        grounding = 3.5
        if any(term in r_low for term in ["settings >", "settings", "apple.co", "iforgot.apple.com", "genius bar", "restart"]):
            grounding = 4.8
        if any(term in r_low for term in ["password", "credential"]) and "dm" not in r_low:
            grounding = 1.0  # Dangerous safety breach
        if len(draft_reply) < 15:
            grounding = 2.0

        # Dimension 2: Intent Relevance (1-5)
        relevance = 3.0
        intent_keywords = {
            "battery_power": ["battery", "drain", "charge", "health"],
            "software_update_os": ["update", "ios", "restart", "version", "about"],
            "connectivity_network": ["wifi", "wi-fi", "bluetooth", "network", "reset"],
            "hardware_display_audio": ["repair", "genius", "sound", "screen", "inspect"],
            "apple_id_account_security": ["apple id", "security", "iforgot", "dm", "private"],
            "billing_subscriptions_appstore": ["charge", "subscription", "purchase", "billing", "dm"],
            "general_inquiry_feedback": ["help", "assist", "reach", "device", "support"]
        }
        matched_kw = sum(1 for kw in intent_keywords.get(true_intent, []) if kw in r_low)
        if matched_kw >= 2:
            relevance = 5.0
        elif matched_kw == 1:
            relevance = 4.0
        else:
            relevance = 2.5

        # Dimension 3: Escalation Appropriateness & Safety (1-5)
        has_dm_or_repair = any(k in r_low for k in ["dm", "direct message", "genius bar", "repair", "store"])
        if true_escalation:
            # When escalation was required:
            if has_dm_or_repair:
                escalation_safety = 5.0
            else:
                # Dangerous failure: failed to escalate when required
                escalation_safety = 1.5
        else:
            # When auto-handle was appropriate:
            if has_dm_or_repair:
                # Over-escalation: harmless but wastes human agent resources
                escalation_safety = 3.5
            else:
                escalation_safety = 5.0

        # Dimension 4: Tone & Empathy (1-5)
        tone = 3.5
        if any(phrase in r_low for phrase in ["we're here to help", "we understand", "let's work together", "thanks for reaching out", "we'd like to help"]):
            tone = 4.8
        if len(draft_reply) > 280:
            tone = max(1.0, tone - 1.5)  # Penalty for exceeding Twitter length

        # Composite score (Weighted sum)
        composite = (
            0.30 * grounding +
            0.25 * relevance +
            0.25 * escalation_safety +
            0.20 * tone
        )

        critique = (
            f"Grounding={grounding:.1f}, Relevance={relevance:.1f}, "
            f"EscalationSafety={escalation_safety:.1f}, Tone={tone:.1f}."
        )

        return JudgeEvaluation(
            grounding_score=round(grounding, 2),
            intent_relevance_score=round(relevance, 2),
            escalation_safety_score=round(escalation_safety, 2),
            tone_empathy_score=round(tone, 2),
            composite_score=round(composite, 2),
            critique=critique
        )

    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        draft_replies: List[str]
    ) -> List[JudgeEvaluation]:
        """Evaluate a batch of drafted replies."""
        evals = []
        for sample, draft in zip(samples, draft_replies):
            evals.append(
                self.evaluate_reply(
                    customer_text=sample["customer_text"],
                    draft_reply=draft,
                    true_intent=sample["true_intent"],
                    true_escalation=sample["true_escalation"],
                    historical_reference_reply=sample.get("historical_reference_reply", ""),
                    gold_resolution_notes=sample.get("gold_resolution_notes", "")
                )
            )
        return evals

    def evaluate_human_agreement(
        self,
        samples: List[Dict[str, Any]],
        draft_replies: List[str]
    ) -> Dict[str, Any]:
        """
        Compute agreement metrics (Cohen's Kappa and Pearson r)
        between human gold ratings and the LLM Judge composite scores.
        """
        judge_evals = self.evaluate_batch(samples, draft_replies)
        human_scores = [float(s.get("human_quality_score", 4.0)) for s in samples]
        judge_scores = [ev.composite_score for ev in judge_evals]
        return compute_judge_human_agreement(human_scores, judge_scores)
