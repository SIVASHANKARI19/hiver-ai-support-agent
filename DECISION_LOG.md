# Decision Log: 12 Non-Obvious Engineering Decisions

This document details the architectural, algorithmic, and product trade-offs made while building the AI Customer Support Agent for **@AppleSupport**.

---

### 1. Selected `@AppleSupport` Over Multi-Lingual E-Commerce Brands (e.g. `@AmazonHelp`)
- **Decision**: Focused solely on `@AppleSupport` from the Twitter Customer Support dataset (`twcs.csv`).
- **Rationale**: While `@AmazonHelp` has higher raw tweet volume, over 40% of its tweets are in Japanese, German, or Spanish, and the vast majority of English replies are generic boilerplate directing users to phone/chat links. In contrast, `@AppleSupport` conversations are 99%+ English and contain rich, technical troubleshooting dialogues (iOS settings, hardware checks, battery diagnostics, recovery modes), presenting a much higher-signal testbed for grounded generation and safety triage.

### 2. Pre-Packaged Local Corpus for Instant (<2-Minute) Offline Reproducibility
- **Decision**: Pre-processed and packaged 3,720 cleaned historical Apple Support dialogues into `data/processed/historical_resolutions.json` and 200 hand-labelled test cases into `data/golden_eval_set.json`.
- **Rationale**: The assignment states: *"README must let us reproduce your headline results in under 15 minutes."* Expecting reviewers to download the 600MB Kaggle CSV, install heavy PyTorch GPU drivers, or configure paid OpenAI/Gemini API keys creates high friction and potential failure points. Packaging the curated corpus and providing an offline-capable evaluation harness ensures the entire test suite and benchmark run in under 2 seconds.

### 3. Evaluated Baseline 2 with 5-Fold Cross-Validation to Avoid Train-on-Test Leakage
- **Decision**: Implemented honest 5-fold cross-validation when evaluating Baseline 2 (TF-IDF + Logistic Regression) on the golden set.
- **Rationale**: In naive implementations, developers often fit the TF-IDF vectorizer and classifier on the entire evaluation set, artificially inflating Baseline 2 to 100% accuracy. By enforcing 5-fold cross-validation, Baseline 2 achieves an honest out-of-fold Macro F1 of **0.759**, proving that our Proposed System's **0.865** Macro F1 is a genuine structural improvement rather than a data leakage illusion.

### 4. Enforced Self-Match Exclusion During Retrieval Benchmarking
- **Decision**: During nearest-neighbor retrieval in Baseline 2, the exact matching reference tweet was excluded from candidate answers.
- **Rationale**: If the evaluation set is drawn from the historical corpus, naive vector retrieval will find the exact query and return its own reference reply, yielding an artificial ROUGE-L of 0.98. By filtering self-matches, we measured true retrieval generalization, showing that verbatim nearest-neighbor replies yield only 0.179 ROUGE-L because distinct customer problems require personalized adaptation.

### 5. Asymmetric Escalation Cost-Weighting ($3 \times \text{FN} + 1 \times \text{FP}$)
- **Decision**: Created an asymmetric penalty metric where False Auto-Handles (failing to escalate a critical issue to a human) are penalized 3x more heavily than False Escalations (unnecessarily routing a safe question to a human).
- **Rationale**: In customer support, sending an angry customer threatening legal action or a user with a compromised Apple ID to an automated troubleshooting bot causes catastrophic brand damage and security vulnerabilities. Conversely, an over-escalation only incurs minor human agent labor cost.

### 6. Prioritized Escalation Recall Over Pure Precision
- **Decision**: Tuned the escalation triage threshold to maximize Escalation Recall (**0.608** vs. **0.144** for Baseline 2), reducing the False Auto-Handle Rate from **85.6%** down to **39.2%**.
- **Rationale**: In customer service automation, a system that boasts 95% precision by only escalating the most obvious keywords misses the vast majority of nuanced failures. Driving recall up is the only way to build customer trust.

### 7. Consolidated Intent Taxonomy into 7 Mutually Exclusive Operational Classes
- **Decision**: Pruned hundreds of fine-grained Twitter topics into 7 operational categories: `battery_power`, `software_update_os`, `connectivity_network`, `hardware_display_audio`, `apple_id_account_security`, `billing_subscriptions_appstore`, and `general_inquiry_feedback`.
- **Rationale**: In customer operations, intent classification exists to route tickets to specific backend workflows or specialized agent queues. Overly granular intents (e.g. "Bluetooth car audio" vs. "Bluetooth headphones") suffer from low inter-annotator agreement and do not alter the resolution playbook (Settings > Bluetooth > Forget Device).

### 8. Structured Escalation Reason Enums with Human Justifications
- **Decision**: Required the agent to output a structured reason code (`SECURITY_AND_PII`, `HARDWARE_DAMAGE`, `BILLING_DISPUTE`, `HIGH_SENTIMENT_RISK`, `COMPLEX_UNRESOLVED`) alongside an explanation.
- **Rationale**: An opaque boolean flag (`escalate: true`) leaves human agents blind when taking over the conversation. A structured reason allows intelligent CRM routing (e.g., routing `BILLING_DISPUTE` directly to the finance tier, and `HARDWARE_DAMAGE` to the Genius Bar scheduling queue).

### 9. Hybrid Decision Fusion for Intent Prediction
- **Decision**: Built a hybrid classifier combining high-precision regex matching for Apple-specific terminology (`boot loop`, `ios 11.1`, `2fa`, `dfu mode`, `genius bar`) with semantic prototype similarity.
- **Rationale**: Pure neural/embedding models frequently confuse technical jargon when queries are short or sarcastic. Pure rule-based models fail on conversational paraphrasing. The hybrid fusion provides robust accuracy across both clean and noisy queries.

### 10. Strict 280-Character Boundary Enforcement on Generated Replies
- **Decision**: Hard-truncated and validated that all drafted replies strictly comply with Twitter's 280-character post limit.
- **Rationale**: LLM-generated customer support replies frequently output 3-4 paragraphs of verbose text. In a Twitter customer service context, an over-length response cannot be posted via the Twitter API and fails in production.

### 11. Decoupled 4-Dimension Rubric for LLM-as-a-Judge
- **Decision**: Designed the evaluation judge with 4 explicit weighted dimensions (Grounding 30%, Relevance 25%, Escalation Safety 25%, Tone 20%) rather than a single holistic 1-5 rating.
- **Rationale**: Single holistic ratings from LLM judges are notoriously subjective and prone to leniency bias. Splitting into explicit criteria forces the judge to penalize safety breaches (e.g., asking for credentials in public) regardless of how polite the tone sounds.

### 12. Deliberate Choice NOT to Build Screenshot OCR / Vision in V1
- **Decision**: Intentionally treated image-only tweets (`@AppleSupport https://t.co/xyz`) as an explicit escalation trigger rather than building an OCR pipeline.
- **Rationale**: Twitter screenshots often contain sensitive private information (phone numbers, full names, account numbers, personal photos). Automatically parsing and processing public images with vision models introduces severe privacy, hallucination, and latency risks. Routing image-heavy queries to human DM triage is the safer operational choice for V1.
