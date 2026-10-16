# ITRI625 Theory Report: Cyber Threat Intelligence Classification

## 3.1 Model scenario

A security operations centre (SOC) receives thousands of threat-intelligence reports, vendor advisories, and incident notes. Analysts must rapidly triage them: a report about ransomware should reach malware-response playbooks, while a report about an adversary group should inform attribution and hunting. This project classifies the text of each report into its primary threat category. It provides a consistent first-pass route, allowing analysts to spend their time validating and investigating rather than manually sorting repetitive reports. It is decision support, not an autonomous security control.

## 3.2 Model architecture

The implementation uses a TensorFlow/Keras BiLSTM with trainable attention:

```text
CTI report text
      │
TextVectorization (vocabulary learned from training text only)
      │
Embedding (128-dimensional learned token vectors)
      │
Bidirectional LSTM (64 units each direction; sequence output)
      │
AttentionLayer (learns the most informative tokens)
      │
Dropout (0.40) → Dense(64, ReLU) → Dropout(0.30)
      │
Softmax (one probability per threat class)
```

The embedding layer maps sparse word identifiers to dense semantic representations. The forward LSTM retains preceding context and the backward LSTM supplies succeeding context: this is useful where an entity’s meaning depends on language on either side, such as “APT28 used … malware.” Attention then computes a learned weighted summary of all BiLSTM states, allowing tokens such as a malware family, actor identifier, or attack technique to contribute more than filler prose. ReLU supplies a compact non-linear decision layer; dropout regularises the network. Softmax yields mutually exclusive class probabilities, and sparse categorical cross-entropy is appropriate because labels are integer encoded.

The model is compiled with Adam, accuracy, precision, recall, a custom sparse macro-F1 metric, and multi-label AUC. Macro F1 deliberately gives every category equal weight despite unequal frequencies.

## 3.3 Result analysis

`results/test_metrics.json` is the authoritative final summary after `python -m src.train` followed by `python -m src.evaluate`. Accuracy is the share of correctly triaged reports, but alone can hide failure on rare categories. Macro precision measures how often predicted classes are correct; macro recall measures how much of each actual class is found; macro F1 balances both. One-vs-rest macro AUC measures ranking quality across all decision thresholds. The classification report and confusion matrix show class-specific precision, recall, and F1, while ROC and precision-recall curves expose threshold behaviour.

Likely difficult labels are semantically overlapping categories such as `software`, `attack-pattern`, and `malware`: an advisory often mentions all three. Short text and documents with multiple entities also make a single first-entity-derived label imperfect. The pipeline records split distributions, rejects overlapping IDs to prevent leakage, and uses macro metrics to make imbalance visible. If the report shows poor minority-class recall, practical next steps are class weighting, more labelled examples, focal loss, and a multi-label reformulation where the problem definition permits it.

The model is effective when its held-out per-class recall and precision meet the SOC’s agreed operating threshold, especially for high-cost missed classes. It must still be reviewed by an analyst, particularly when the desktop app flags a confidence below 60%.

## 3.4 Real-world use

Machine learning can support intrusion detection, threat-text classification, anomaly detection, alert prioritisation, phishing detection, and risk assessment. Its strengths are scale, consistent routing, and the ability to discover patterns that are difficult to encode as rules. In this project, probability output and SHAP token explanations make the recommendation inspectable.

The limitations are material. False positives waste analyst capacity and can cause alert fatigue; false negatives leave attacks unseen. Cybersecurity is not a stable closed world: adversaries intentionally change vocabulary, techniques, and infrastructure, producing concept drift. Poor annotations, stale training examples, duplicate reports, and biased label distributions can therefore degrade a superficially high score. A production workflow needs feedback from analysts, monitoring of distributions and calibration, periodic re-training, strict separation of train/test data, access control, and incident-response playbooks that never rely solely on a prediction.

### References

1. Buczak, A. L., & Guven, E. (2016). A survey of data mining and machine learning methods for cybersecurity intrusion detection. *IEEE Communications Surveys & Tutorials, 18*(2), 1153–1176.
2. Sommer, R., & Paxson, V. (2010). Outside the closed world: On using machine learning for network intrusion detection. *IEEE Symposium on Security and Privacy*, 305–316.
3. Apruzzese, G., Laskov, P., Montes de Oca, E., Mallouli, W., Brdalo Rapa, L., Grammatopoulos, A. V., & Di Franco, F. (2022). The role of machine learning in cybersecurity. *Digital Threats: Research and Practice, 4*(1), 1–38.
4. Ramoliya, D., Patel, P., & Patel, H. (2023). SEAM: Deep learning-based secure message exchange. *International Journal of Information Security and Privacy*.
5. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP*, 108–116.
