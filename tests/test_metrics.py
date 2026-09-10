"""Unit tests for Evaluation Metrics and Agreement calculations."""

import pytest
from src.metrics import (
    compute_intent_metrics,
    compute_escalation_metrics,
    compute_rouge_l,
    compute_bleu_1,
    compute_judge_human_agreement
)


def test_intent_metrics_calculation():
    y_true = ["battery_power", "software_update_os", "battery_power"]
    y_pred = ["battery_power", "software_update_os", "connectivity_network"]
    labels = ["battery_power", "software_update_os", "connectivity_network"]

    m = compute_intent_metrics(y_true, y_pred, labels)
    assert m["accuracy"] == round(2 / 3, 4)
    assert 0.0 <= m["macro_f1"] <= 1.0


def test_escalation_metrics_calculation():
    y_true = [True, True, False, False]
    y_pred = [True, False, False, True]

    m = compute_escalation_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["false_auto_handle_rate"] == 0.5


def test_rouge_and_bleu():
    cand = "We'd like to help you check your battery health in Settings"
    ref = "We'd like to help. Check Settings > Battery > Battery Health"

    rouge = compute_rouge_l(cand, ref)
    bleu = compute_bleu_1(cand, ref)

    assert 0.0 < rouge <= 1.0
    assert 0.0 < bleu <= 1.0


def test_judge_human_agreement():
    h = [5.0, 4.0, 3.0, 2.0, 1.0]
    j = [4.8, 4.2, 3.1, 1.9, 1.2]

    agr = compute_judge_human_agreement(h, j)
    assert agr["pearson_r"] > 0.90
    assert agr["mean_absolute_error"] < 0.30
