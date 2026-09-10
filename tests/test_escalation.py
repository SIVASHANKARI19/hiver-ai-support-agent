"""Unit tests for Escalation Engine."""

import pytest
from src.escalation_engine import EscalationEngine


@pytest.fixture
def engine():
    return EscalationEngine()


def test_escalate_security_and_pii(engine):
    # Apple ID credentials
    dec = engine.evaluate("I forgot my Apple ID password and can't get verification code", "apple_id_account_security")
    assert dec.should_escalate is True
    assert dec.reason_code == "SECURITY_AND_PII"
    assert "DM" in dec.explanation or "private" in dec.explanation


def test_escalate_hardware_damage(engine):
    # Physical broken screen
    dec = engine.evaluate("I dropped my phone on concrete and the glass is cracked", "hardware_display_audio")
    assert dec.should_escalate is True
    assert dec.reason_code == "HARDWARE_DAMAGE"


def test_escalate_billing_dispute(engine):
    # Charge dispute
    dec = engine.evaluate("I need an immediate refund for an unauthorized App Store charge", "billing_subscriptions_appstore")
    assert dec.should_escalate is True
    assert dec.reason_code == "BILLING_DISPUTE"


def test_escalate_complex_unresolved(engine):
    # Customer already tried restart
    dec = engine.evaluate("I already restarted and reset settings and nothing works", "software_update_os")
    assert dec.should_escalate is True
    assert dec.reason_code == "COMPLEX_UNRESOLVED"


def test_escalate_high_sentiment_risk(engine):
    # Churn / rage
    dec = engine.evaluate("Worst customer service ever I am switching to Samsung garbage company", "general_inquiry_feedback")
    assert dec.should_escalate is True
    assert dec.reason_code == "HIGH_SENTIMENT_RISK"


def test_auto_handle_standard_troubleshooting(engine):
    # Standard settings question
    dec = engine.evaluate("How can I check my battery health in settings?", "battery_power")
    assert dec.should_escalate is False
    assert dec.reason_code == "KNOWN_TROUBLESHOOTING_CAN_AUTOHANDLE"
