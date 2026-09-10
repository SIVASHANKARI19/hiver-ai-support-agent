"""Intent classification models: Trivial Baseline, TF-IDF Baseline, and Hybrid Semantic Classifier."""

import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.config import INTENTS, INTENT_DESCRIPTIONS


class BaseIntentClassifier:
    """Abstract base class for intent classifiers."""

    def predict(self, text: str) -> str:
        raise NotImplementedError

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.predict(t) for t in texts]


class TrivialMajorityClassifier(BaseIntentClassifier):
    """Baseline 1: Trivial baseline that always predicts the majority intent class."""

    def __init__(self, majority_intent: str = "general_inquiry_feedback"):
        self.majority_intent = majority_intent

    def predict(self, text: str) -> str:
        return self.majority_intent


class TfidfIntentClassifier(BaseIntentClassifier):
    """Baseline 2: Simple statistical classifier using TF-IDF and Logistic Regression."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
        self.classifier = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> "TfidfIntentClassifier":
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_fitted = True
        return self

    def predict(self, text: str) -> str:
        if not self.is_fitted:
            return "general_inquiry_feedback"
        X = self.vectorizer.transform([text])
        return str(self.classifier.predict(X)[0])


class HybridSemanticClassifier(BaseIntentClassifier):
    """
    Proposed System: Production-grade Hybrid Intent Classifier.
    Combines domain-specific high-precision lexical feature matching with
    semantic similarity scoring across canonical Apple Support intent prototypes.
    """

    PATTERNS = {
        "battery_power": [
            r"\bbattery\b", r"\bdrain(ing)?\b", r"\bcharg(e|ing|er)?\b", r"\boverheat(ing)?\b",
            r"\bpower\s*down\b", r"\bpercentage\b", r"\bdies\s*(fast|quickly)\b", r"\bbattery\s*health\b",
            r"\blife\s*of\s*my\s*phone\b", r"\bmAh\b", r"\blower\s*power\s*mode\b"
        ],
        "software_update_os": [
            r"\bios\b", r"\bupdate(d|s)?\b", r"\binstall(ing)?\b", r"\bglitch(es|y)?\b",
            r"\bfreez(e|ing|es)?\b", r"\bboot\s*loop\b", r"\brestart(ing|ed)?\b", r"\bkeyboard\b",
            r"\bautocorrect\b", r"\bbug(s)?\b", r"\bios\s*11\b", r"\bcrash(es|ing|ed)?\b",
            r"\blag(gy|ging)?\b", r"\bsoftware\b", r"\bapple\s*logo\b", r"\bslow\s*phone\b"
        ],
        "connectivity_network": [
            r"\bwi-?fi\b", r"\bbluetooth\b", r"\bcellular\b", r"\bairdrop\b", r"\bsignal\b",
            r"\blte\b", r"\bno\s*service\b", r"\bdata\b", r"\bconnect(ion|ing|ed)?\b",
            r"\bhotspot\b", r"\bairplane\s*mode\b", r"\bnetwork\b", r"\bcarplay\b", r"\bpairing\b"
        ],
        "hardware_display_audio": [
            r"\bscreen\b", r"\bcrack(ed)?\b", r"\bshatter(ed)?\b", r"\btouch(screen)?\b",
            r"\bdisplay\b", r"\bspeaker\b", r"\bmic(rophone)?\b", r"\bsound\b", r"\bvolume\b",
            r"\bcamera\b", r"\bhome\s*button\b", r"\bheadphone(s)?\b", r"\bjack\b",
            r"\bwater\s*damage\b", r"\bdead\s*pixel\b", r"\bgorilla\s*glass\b", r"\bstatic\b"
        ],
        "apple_id_account_security": [
            r"\bapple\s*id\b", r"\bicloud\b", r"\bpassword\b", r"\blocked\b", r"\bdisabled\b",
            r"\bsecurity\b", r"\b2fa\b", r"\btwo-?factor\b", r"\bverification\s*code\b",
            r"\bsign\s*in\b", r"\blogin\b", r"\baccount\b", r"\bhacked\b", r"\bactivation\s*lock\b"
        ],
        "billing_subscriptions_appstore": [
            r"\bbill(ing)?\b", r"\bcharge(d|s)?\b", r"\brefund\b", r"\bsubscription(s)?\b",
            r"\bapp\s*store\b", r"\bitunes\b", r"\bpay(ment)?\b", r"\bpurchase(d)?\b",
            r"\bcard\b", r"\bcredit\b", r"\breceipt\b", r"\border\b", r"\breserv(e|ation)\b",
            r"\brenew(al)?\b", r"\bapple\s*pay\b"
        ]
    }

    INTENT_PROTOTYPES = {
        "battery_power": "my battery is draining extremely fast after charging phone overheats dies",
        "software_update_os": "the new ios update crashed my phone stuck in boot loop screen froze bug glitch",
        "connectivity_network": "wifi keeps disconnecting bluetooth won't connect no cellular network data signal",
        "hardware_display_audio": "cracked glass screen touch unresponsive speaker distorted microphone camera broken",
        "apple_id_account_security": "forgot my apple id password account is locked or disabled 2fa verification code",
        "billing_subscriptions_appstore": "unexpected charge on my credit card refund request app store subscription purchase",
        "general_inquiry_feedback": "terrible customer service why does apple do this store appointment question feedback"
    }

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        proto_texts = [self.INTENT_PROTOTYPES[k] for k in INTENTS]
        self.proto_matrix = self.vectorizer.fit_transform(proto_texts)

    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        lower = text.lower()

        # Step 1: Lexical domain rule matching with score accumulation
        rule_scores = {intent: 0.0 for intent in INTENTS}
        for intent, patterns in self.PATTERNS.items():
            for pat in patterns:
                matches = len(re.findall(pat, lower))
                if matches > 0:
                    rule_scores[intent] += matches * 1.5

        top_rule_intent = max(rule_scores, key=rule_scores.get)
        top_rule_score = rule_scores[top_rule_intent]

        # Step 2: Semantic prototype similarity
        query_vec = self.vectorizer.transform([lower])
        sims = (self.proto_matrix @ query_vec.T).toarray().ravel()
        proto_scores = {intent: float(sims[idx]) for idx, intent in enumerate(INTENTS)}
        top_proto_intent = max(proto_scores, key=proto_scores.get)
        top_proto_score = proto_scores[top_proto_intent]

        # Step 3: Decision Fusion
        # If strong rule match, prioritize domain pattern
        if top_rule_score >= 1.5:
            conf = min(0.65 + (top_rule_score * 0.1), 0.98)
            return top_rule_intent, round(conf, 3)

        # If semantic similarity is distinct, use prototype match
        if top_proto_score > 0.15:
            conf = min(0.50 + top_proto_score, 0.92)
            return top_proto_intent, round(conf, 3)

        # Default fallback to general feedback
        return "general_inquiry_feedback", 0.45

    def predict(self, text: str) -> str:
        intent, _ = self.predict_with_confidence(text)
        return intent
