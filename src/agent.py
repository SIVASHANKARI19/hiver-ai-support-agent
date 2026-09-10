"""Unified Apple Support AI Agent Pipeline."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from src.intent_classifier import HybridSemanticClassifier, BaseIntentClassifier
from src.retriever import HistoricalResolutionRetriever
from src.escalation_engine import EscalationEngine, EscalationDecision
from src.response_generator import GroundedResponseGenerator


@dataclass
class AgentResponse:
    customer_text: str
    intent: str
    intent_confidence: float
    should_escalate: bool
    escalation_reason: str
    escalation_explanation: str
    draft_reply: str
    grounded_resolutions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_text": self.customer_text,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "should_escalate": self.should_escalate,
            "escalation_reason": self.escalation_reason,
            "escalation_explanation": self.escalation_explanation,
            "draft_reply": self.draft_reply,
            "grounded_resolutions": self.grounded_resolutions
        }


class AppleSupportAgent:
    """
    End-to-end AI Customer Support Agent for Apple Support.
    Executes intent classification, historical resolution retrieval,
    escalation triage, and grounded reply generation.
    """

    def __init__(
        self,
        classifier: Optional[BaseIntentClassifier] = None,
        retriever: Optional[HistoricalResolutionRetriever] = None,
        escalation_engine: Optional[EscalationEngine] = None,
        generator: Optional[GroundedResponseGenerator] = None
    ):
        self.classifier = classifier or HybridSemanticClassifier()
        self.retriever = retriever or HistoricalResolutionRetriever()
        self.escalation_engine = escalation_engine or EscalationEngine()
        self.generator = generator or GroundedResponseGenerator()

    def process_message(self, customer_text: str) -> AgentResponse:
        """Process a single incoming customer support tweet."""
        # 1. Classify Intent
        if hasattr(self.classifier, "predict_with_confidence"):
            intent, conf = self.classifier.predict_with_confidence(customer_text)
        else:
            intent = self.classifier.predict(customer_text)
            conf = 0.85

        # 2. Retrieve Historical Grounding
        retrieved = self.retriever.retrieve(customer_text, top_k=3)

        # 3. Escalation Decision & Reasoning
        esc_decision = self.escalation_engine.evaluate(customer_text, intent)

        # 4. Draft Grounded Reply
        reply = self.generator.generate(
            customer_text=customer_text,
            predicted_intent=intent,
            escalation=esc_decision,
            retrieved_resolutions=retrieved
        )

        return AgentResponse(
            customer_text=customer_text,
            intent=intent,
            intent_confidence=conf,
            should_escalate=esc_decision.should_escalate,
            escalation_reason=esc_decision.reason_code,
            escalation_explanation=esc_decision.explanation,
            draft_reply=reply,
            grounded_resolutions=retrieved
        )

    def process_batch(self, customer_texts: List[str]) -> List[AgentResponse]:
        """Process multiple customer inquiries."""
        return [self.process_message(text) for text in customer_texts]
