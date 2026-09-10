"""Evaluation metrics calculation for Intent, Escalation, Generation, and Judge Agreement."""

import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_intent_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    """Calculate Intent Classification accuracy, macro/micro F1, and per-class stats."""
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(per_p[idx]), 3),
            "recall": round(float(per_r[idx]), 3),
            "f1": round(float(per_f1[idx]), 3),
            "support": int(per_sup[idx])
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels
    }


def compute_escalation_metrics(
    y_true: List[bool],
    y_pred: List[bool],
    reasons_true: Optional[List[str]] = None,
    reasons_pred: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculate Escalation Precision, Recall, False Auto-handle Rate, and Reason Match."""
    total = len(y_true)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t is True and p is True)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t is False and p is True)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t is True and p is False)  # Dangerous False Auto-handle
    tn = sum(1 for t, p in zip(y_true, y_pred) if t is False and p is False)

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)  # Escalation Recall
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)

    # Risk metrics
    total_escalations_needed = max(tp + fn, 1)
    false_auto_handle_rate = fn / total_escalations_needed  # Missed escalations
    false_escalation_rate = fp / max(tn + fp, 1)            # Wasted human bandwidth

    # Cost-weighted penalty: False Auto-Handle is 3x more costly than False Escalation
    cost_penalty = (fn * 3.0 + fp * 1.0) / max(total, 1)

    reason_acc = None
    if reasons_true and reasons_pred:
        matches = sum(1 for rt, rp in zip(reasons_true, reasons_pred) if rt == rp)
        reason_acc = round(matches / max(total, 1), 4)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "false_auto_handle_rate": round(float(false_auto_handle_rate), 4),
        "false_escalation_rate": round(float(false_escalation_rate), 4),
        "cost_weighted_penalty": round(float(cost_penalty), 4),
        "reason_accuracy": reason_acc,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in text.split() if len(w) > 0]


def compute_rouge_l(candidate: str, reference: str) -> float:
    """Calculate ROUGE-L F1 score based on Longest Common Subsequence."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    m, n = len(cand_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0

    # LCS dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if cand_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    prec = lcs_len / m
    rec = lcs_len / n
    if prec + rec == 0:
        return 0.0
    return round((2 * prec * rec) / (prec + rec), 4)


def compute_bleu_1(candidate: str, reference: str) -> float:
    """Calculate unigram BLEU precision with brevity penalty."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0

    cand_counts = Counter(cand_tokens)
    ref_counts = Counter(ref_tokens)

    clipped_matches = sum(min(count, ref_counts.get(word, 0)) for word, count in cand_counts.items())
    prec = clipped_matches / len(cand_tokens)

    # Brevity penalty
    bp = math.exp(1 - len(ref_tokens) / len(cand_tokens)) if len(cand_tokens) < len(ref_tokens) else 1.0
    return round(bp * prec, 4)


def compute_generation_metrics(candidates: List[str], references: List[str]) -> Dict[str, Any]:
    """Calculate aggregate BLEU, ROUGE-L, and Semantic Similarity against historical brand replies."""
    rouge_scores = [compute_rouge_l(c, r) for c, r in zip(candidates, references)]
    bleu_scores = [compute_bleu_1(c, r) for c, r in zip(candidates, references)]

    # Semantic similarity using TF-IDF cosine similarity
    tfidf = TfidfVectorizer().fit(candidates + references)
    cand_vecs = tfidf.transform(candidates)
    ref_vecs = tfidf.transform(references)
    sims = [float(cosine_similarity(cand_vecs[i], ref_vecs[i])[0][0]) for i in range(len(candidates))]

    # Policy compliance: <= 280 chars
    char_compliant = sum(1 for c in candidates if len(c) <= 280) / max(len(candidates), 1)

    return {
        "mean_rouge_l": round(float(np.mean(rouge_scores)), 4),
        "mean_bleu_1": round(float(np.mean(bleu_scores)), 4),
        "mean_semantic_similarity": round(float(np.mean(sims)), 4),
        "character_limit_compliance": round(float(char_compliant), 4)
    }


def compute_judge_human_agreement(human_scores: List[float], judge_scores: List[float]) -> Dict[str, Any]:
    """
    Calculate Cohen's Kappa (quadratic weighted) and Pearson correlation
    between human ground-truth ratings and LLM Judge ratings.
    """
    # Round to discrete integer buckets (1 to 5) for Cohen's Kappa
    h_int = [int(round(s)) for s in human_scores]
    j_int = [int(round(s)) for s in judge_scores]

    kappa = cohen_kappa_score(h_int, j_int, weights="quadratic")

    # Pearson correlation r
    if len(human_scores) > 1:
        corr_matrix = np.corrcoef(human_scores, judge_scores)
        pearson_r = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
    else:
        pearson_r = 0.0

    mae = float(np.mean(np.abs(np.array(human_scores) - np.array(judge_scores))))

    return {
        "cohens_kappa_quadratic": round(float(kappa), 4),
        "pearson_r": round(float(pearson_r), 4),
        "mean_absolute_error": round(float(mae), 4),
        "sample_count": len(human_scores)
    }
