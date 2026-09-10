"""Unit tests for Historical Resolution Retriever."""

import pytest
from src.retriever import HistoricalResolutionRetriever


@pytest.fixture
def mock_corpus():
    return [
        {
            "pair_id": 1,
            "customer_text": "My battery is draining so fast after updating to iOS 11",
            "brand_reply": "We'd like to help. Check Settings > Battery > Battery Health to inspect apps."
        },
        {
            "pair_id": 2,
            "customer_text": "Wi-Fi disconnected and Bluetooth is not finding any devices",
            "brand_reply": "Try resetting network settings under Settings > General > Reset > Reset Network Settings."
        },
        {
            "pair_id": 3,
            "customer_text": "My phone is frozen on the Apple logo",
            "brand_reply": "Try a force restart by pressing and holding the side button and volume down button."
        }
    ]


def test_retriever_initialization_and_search(mock_corpus):
    retriever = HistoricalResolutionRetriever(mock_corpus)
    results = retriever.retrieve("my battery drains quickly", top_k=2, min_similarity=0.0)

    assert len(results) >= 1
    assert "battery" in results[0]["brand_reply"].lower()
    assert results[0]["similarity_score"] >= 0.0


def test_retriever_network_query(mock_corpus):
    retriever = HistoricalResolutionRetriever(mock_corpus)
    results = retriever.retrieve("bluetooth and wifi failure", top_k=1, min_similarity=0.0)

    assert len(results) == 1
    assert "network" in results[0]["brand_reply"].lower()
