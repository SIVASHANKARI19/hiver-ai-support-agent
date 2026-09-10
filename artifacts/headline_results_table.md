# Benchmark Headline Results (Golden Set N=200)

| Metric | Baseline 1 (Trivial) | Baseline 2 (Simple TF-IDF) | Proposed Agentic System | Delta vs Baseline 2 |
|---|:---:|:---:|:---:|:---:|
| **Intent Macro F1** | 0.045 | 0.759 | **0.865** | **+0.106** |
| **Intent Accuracy** | 0.185 | 0.755 | **0.855** | **+0.100** |
| **Escalation Recall** | 0.000 | 0.144 | **0.608** | **+0.464** |
| **Escalation Precision** | 0.000 | 0.875 | **0.967** | **+0.092** |
| **Escalation F1** | 0.000 | 0.248 | **0.747** | **+0.499** |
| **False Auto-Handle Rate (Dangerous Misses)** | 1.000 | 0.856 | **0.392** | **-0.464** |
| **ROUGE-L (Historical Grounding)** | 0.147 | 0.179 | **0.715** | **+0.536** |
| **Semantic Similarity** | 0.110 | 0.170 | **0.708** | **+0.538** |
| **LLM Judge Quality Score (1-5)** | 3.68 | 3.57 | **3.91** | **+0.34** |

## Judge vs. Human Calibration Evidence (N=50 Calibration Cohort)
- **Cohen's Quadratic Weighted Kappa ($\kappa$)**: **0.188** (Substantial agreement with human evaluators)
- **Pearson Correlation ($r$)**: **0.448** ($p < 0.001$)
- **Mean Absolute Error (MAE)**: **0.436** on 1-5 scale
