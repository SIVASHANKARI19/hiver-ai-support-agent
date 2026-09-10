# Citations & Attribution Guide

In compliance with the assignment rules (*"Cite anything you borrowed. Borrowing is fine; not knowing what you borrowed is not."*), this document details all datasets, algorithms, libraries, and design patterns utilized in this repository.

---

## 1. Datasets & Corpora
- **Customer Support on Twitter**:
  - **Source**: Kaggle (`thoughtvector/customer-support-on-twitter`), public mirror by Sunidhi Sriram (`SunidhiSriram/twcs` on Hugging Face).
  - **License**: CC BY-SA 4.0.
  - **Usage in Repo**: Subsampled and extracted 3,720 paired inbound customer queries and outbound `@AppleSupport` resolutions into `data/processed/historical_resolutions.json`.
  - **Golden Evaluation Set**: Curated and hand-annotated 200 stratified test examples across 7 intents and 6 escalation categories into `data/golden_eval_set.json` and `data/golden_eval_set.csv`.

---

## 2. Algorithmic Formulations & Metric Implementations
- **ROUGE-L (Longest Common Subsequence F-measure)**:
  - **Citation**: Chin-Yew Lin. 2004. *ROUGE: A Package for Automatic Evaluation of Summaries*. In Proceedings of the Workshop on Text Summarization Branches Out (WAS 2004), pages 74–81, Barcelona, Spain.
  - **Implementation**: Custom dynamic programming LCS implementation in `src/metrics.py:compute_rouge_l`.
- **BLEU Precision with Brevity Penalty**:
  - **Citation**: Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. *BLEU: a Method for Automatic Evaluation of Machine Translation*. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL '02), pages 311–318.
  - **Implementation**: Unigram modified precision with exponential brevity penalty in `src/metrics.py:compute_bleu_1`.
- **Inter-Annotator Agreement (Quadratic Weighted Kappa)**:
  - **Citation**: Jacob Cohen. 1968. *Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit*. Psychological Bulletin, 70(4):213–220.
  - **Implementation**: Utilized `sklearn.metrics.cohen_kappa_score(..., weights="quadratic")`.
- **LLM-as-a-Judge Evaluation Paradigm**:
  - **Citation**: Lianmin Zheng, Wei-Lin Chiang, Hao Zhang, et al. 2023. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. Advances in Neural Information Processing Systems (NeurIPS 2023).
  - **Implementation**: Adapted multi-criteria decoupled rubric (Grounding 30%, Relevance 25%, Escalation Safety 25%, Tone 20%) in `src/judge.py`.

---

## 3. Libraries & Frameworks
- **scikit-learn** (v1.2+): Feature extraction (`TfidfVectorizer`), linear classification (`LogisticRegression`), and cross-validation utilities (`cross_val_predict`).
  - *Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.*
- **NumPy & Pandas**: Matrix computation, cosine similarity transformations, and tabular evaluation parsing.
  - *Harris et al., Nature 585, 357–362, 2020.*
- **pytest**: Automated unit testing harness for regression and verification.
