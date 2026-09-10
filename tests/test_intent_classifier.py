"""Unit tests for Intent Classifiers."""

import pytest
from src.intent_classifier import (
    TrivialMajorityClassifier,
    TfidfIntentClassifier,
    HybridSemanticClassifier
)
from src.config import INTENTS


def test_trivial_classifier():
    clf = TrivialMajorityClassifier("general_inquiry_feedback")
    assert clf.predict("My battery is dying fast") == "general_inquiry_feedback"
    assert clf.predict("iOS update crashed my phone") == "general_inquiry_feedback"


def test_tfidf_classifier():
    clf = TfidfIntentClassifier()
    texts = [
        "battery draining fast",
        "update ios crashed phone",
        "wifi bluetooth connection dropped",
        "cracked screen display repair"
    ]
    labels = ["battery_power", "software_update_os", "connectivity_network", "hardware_display_audio"]
    clf.fit(texts, labels)

    pred = clf.predict("my battery is dead")
    assert pred in INTENTS


def test_hybrid_semantic_classifier_predictions():
    clf = HybridSemanticClassifier()

    # Battery
    intent, conf = clf.predict_with_confidence("My battery drains in 30 minutes after charging")
    assert intent == "battery_power"
    assert conf >= 0.60

    # Software update
    intent, conf = clf.predict_with_confidence("After the iOS 11 update my phone is stuck in boot loop")
    assert intent == "software_update_os"
    assert conf >= 0.60

    # Connectivity
    intent, conf = clf.predict_with_confidence("Wi-Fi won't connect and Bluetooth keeps disconnecting")
    assert intent == "connectivity_network"
    assert conf >= 0.60

    # Apple ID
    intent, conf = clf.predict_with_confidence("I forgot my Apple ID password and my account is locked")
    assert intent == "apple_id_account_security"
    assert conf >= 0.60

    # Hardware
    intent, conf = clf.predict_with_confidence("My screen is completely shattered and cracked")
    assert intent == "hardware_display_audio"
    assert conf >= 0.60

    # Billing
    intent, conf = clf.predict_with_confidence("I was charged twice for my App Store subscription refund please")
    assert intent == "billing_subscriptions_appstore"
    assert conf >= 0.60
