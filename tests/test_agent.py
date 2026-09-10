"""Unit tests for Unified AppleSupportAgent."""

import pytest
from src.agent import AppleSupportAgent


@pytest.fixture
def agent():
    return AppleSupportAgent()


def test_agent_process_autohandle(agent):
    res = agent.process_message("How do I update my iPhone to the latest software?")
    assert res.intent in ["software_update_os", "general_inquiry_feedback"]
    assert res.should_escalate is False
    assert res.escalation_reason == "KNOWN_TROUBLESHOOTING_CAN_AUTOHANDLE"
    assert len(res.draft_reply) <= 280
    assert len(res.draft_reply) > 20


def test_agent_process_escalate(agent):
    res = agent.process_message("My Apple ID is locked and I cannot sign in to my account!")
    assert res.intent == "apple_id_account_security"
    assert res.should_escalate is True
    assert res.escalation_reason == "SECURITY_AND_PII"
    assert "DM" in res.draft_reply or "apple.co" in res.draft_reply or "iforgot" in res.draft_reply


def test_agent_batch_process(agent):
    queries = [
        "My battery drains fast",
        "My screen broke and cracked",
        "Worst customer service ever"
    ]
    responses = agent.process_batch(queries)
    assert len(responses) == 3
    assert responses[1].should_escalate is True
    assert responses[2].should_escalate is True
