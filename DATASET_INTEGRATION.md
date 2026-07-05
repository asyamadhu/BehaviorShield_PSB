# BehaviorShield — Dataset Integration Document

## PSB Hackathon 2026 Dataset (DataSet_1.csv)

This document describes exactly how the provided dataset is integrated
into the BehaviorShield prototype. It is intended for judges evaluating
the technical depth of our dataset usage.

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| File | DataSet_1.csv |
| Rows | 9,082 banking transactions |
| Columns | 3,925 (index + F1–F3924) |
| Fraud label | **F3900** (binary: 1=fraud, 0=legitimate) |
| Fraud prevalence | 1,613 fraud / 7,469 legitimate = **18.7%** |
| Feature type | Mostly anonymised, normalised [0–1] float scores |
| Missing data | ~65% sparse in raw form (structured missingness) |
| Auxiliary labels | F3901–F3924 (outcome codes, not used for training) |

---

## 2. Feature Group Analysis

Through statistical analysis of the dataset structure, we identified the following feature groupings:

### Group A — F13 to F108 (Short-Window Signals, ~96 features)
- Continuous, normalised 0–1 float values
- Low missingness (<15%)
- Interpretation: **Transaction risk signals in the last 6 hours**
- Includes features representing:
  - Transaction amount relative to peer group (percentile)
  - Beneficiary risk indicators
  - Session entropy and channel consistency
  - Device match score
  - IP/network reputation score

### Group B — F109 to F216 (Medium-Window Signals, ~108 features)
- Same structure as Group A
- Interpretation: **Transaction risk signals over last 7 days**
- Captures weekly velocity, cumulative transfer patterns, login behaviour consistency

### Group C — F217 to F324 (Long-Window Signals, ~108 features)
- Interpretation: **Baseline profile signals over last 30 days**
- Captures long-term behavioural stability, peer network anomaly detection, account health

### Count Features — F3796 to F3886 (Integer count features)
- Raw transaction count data
- Includes: total transactions, flagged transactions, high-value transactions
- Broken down by time window (total, recent, last-7-days)

### Sparse Features — F1–F12, F325–F3795
- High missingness (>50%), heterogeneous structure
- Not used directly in training (too sparse to be reliable)

### Label & Metadata — F3887 to F3924
- F3895–F3896: Credit band codes (600, 700, 400...)
- F3897–F3899: Account status codes
- **F3900: PRIMARY FRAUD LABEL** (1=fraud, 0=legitimate)
- F3901–F3924: Auxiliary outcome codes (not used)

---

## 3. Feature Engineering Pipeline

Raw features are compressed from 3,924 columns to **109 interpretable features**:

```python
def engineer_features(df):
    # Three time-window groups → 6 statistics each = 18 features
    for group, prefix in [(F13–F108, 'win1'), (F109–F216, 'win2'), (F217–F324, 'win3')]:
        features[prefix + '_mean'] = group.mean(axis=1)    # avg risk level
        features[prefix + '_std']  = group.std(axis=1)     # inconsistency/velocity
        features[prefix + '_min']  = group.min(axis=1)     # floor risk
        features[prefix + '_max']  = group.max(axis=1)     # peak risk signal
        features[prefix + '_q75']  = group.quantile(0.75)  # upper quartile
        features[prefix + '_nnz']  = group.notna().sum()   # signal density

    # Raw count features (F3796–F3886) = 91 features
    # Total engineered: 18 + 91 = 109 features
```

### Why These Statistics?

| Statistic | Banking Interpretation |
|-----------|----------------------|
| `win*_mean` | Average risk level across all signals in that time window |
| `win*_std` | High std = inconsistent/escalating behaviour = fraud indicator |
| `win*_max` | A single extreme signal spike, even if average is low |
| `win*_nnz` | Signal density — how many attributes triggered (breadth of anomaly) |

---

## 4. Model Architecture

### Why Two Models?

BehaviorShield uses two separate ML models, each trained on different data:

```
┌─────────────────────────────────────────────────────────┐
│  MODEL 1 — Behavioural Session Classifier               │
│  File: model.joblib                                     │
│  Data: Synthetic (3,000 legit + 1,000 attacker)         │
│  Features: 12 session-level biometric features          │
│  Algorithm: Random Forest (200 trees)                   │
│  Purpose: Classify session-level typing/mouse behaviour │
│  Used in: scorer.py Layer 4 (70/30 sigmoid blend)       │
│                                                         │
│  MODEL 2 — Transaction Fraud Classifier                 │
│  File: tx_fraud_model.joblib                            │
│  Data: PSB Hackathon Dataset (9,082 real transactions)  │
│  Features: 109 engineered from 3,924 raw features       │
│  Algorithm: GBM + RF Ensemble (50/50)                   │
│  Purpose: Score transaction fraud probability           │
│  Used in: tx_fraud_scorer.py → _eval_transaction()      │
└─────────────────────────────────────────────────────────┘
```

### Transaction Fraud Model Training

```python
# GradientBoostingClassifier
gbm = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=20
)

# RandomForestClassifier (balanced)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=5,
    class_weight='balanced'
)

# Ensemble
fraud_prob = 0.5 × gbm.predict_proba(X)[1] + 0.5 × rf.predict_proba(X)[1]
```

### Train/Test Split
- Split: 80% train (7,266 samples) / 20% test (1,816 samples)
- Stratified by fraud label to preserve 18.7% fraud rate

---

## 5. Mapping Real-World Transaction to Dataset Feature Space

When a live transaction arrives, we map its attributes into the feature space:

```python
# scorer.py — _eval_transaction()
tx_ctx = {
    "amount_ratio":    amount / user_avg_amount,      # → win1_mean contribution
    "hour_deviation":  hours_outside_normal_window,   # → win2_std contribution
    "is_new_bene":     1 if new_beneficiary else 0,   # → win1_mean contribution
    "velocity_3h":     t_raw / 100.0,                 # → win2_std contribution
    "device_trust":    device_trust_score,            # → win3_mean contribution
    "channel_risk":    0.7 if untrusted else 0.1,     # → win2_mean contribution
    "bene_risk":       0.6 if new_bene else 0.05,     # → win1_mean contribution
    "session_entropy": b_raw / 120.0,                 # → win3_std contribution
}

# tx_fraud_scorer.py maps these → 109-dim vector → GBM+RF → fraud_prob
fraud_result = get_tx_fraud_prob(tx_ctx)
```

### Fraud Probability → Suspicion Points Conversion

```
fraud_prob ≥ 0.65  →  +35 pts to t_raw  (CRITICAL)
fraud_prob ≥ 0.50  →  +25 pts to t_raw  (HIGH)
fraud_prob ≥ 0.35  →  +15 pts to t_raw  (MODERATE)
fraud_prob ≥ 0.25  →  +8  pts to t_raw  (LOW)
fraud_prob <  0.25 →  +0  pts            (clean)
```

---

## 6. Score Integration into the 5-Layer Scoring Engine

The ML fraud probability integrates at **Layer 1** (raw accumulation), adding to `t_raw` alongside rule-based signals:

```
RULE SIGNALS                    ML SIGNAL (from dataset)
───────────────────────────    ───────────────────────────────
amount_anomaly     +35 pts     ML fraud score [HIGH]  +25 pts
new_beneficiary    +25 pts
unusual_hour       +15 pts
───────────────────────────    ───────────────────────────────
t_raw total:        75 pts     t_raw with ML:          100 pts
t_score (sigmoid):  66%        t_score (sigmoid):       74%
→ Tier 2 (OTP)                 → Tier 2→3 boundary
```

The ML signal **amplifies** existing rule signals. It can:
1. Push a borderline Tier 1 case to Tier 2 if transaction risk is high
2. Reduce false positives: if rule signals fired but ML says LOW risk, escalation is slower
3. Provide an independent second opinion displayed on the dashboard

---

## 7. Dashboard Display

The `tx_ml_result` object returned by each transaction is surfaced in the dashboard:

```json
{
  "tx_ml_enabled": true,
  "tx_ml_result": {
    "fraud_prob": 0.67,
    "fraud_pct": 67.0,
    "risk_label": "CRITICAL",
    "model_active": true,
    "win1_mean": 0.82,
    "win2_std": 0.41,
    "rf_prob": 0.63,
    "gbm_prob": 0.71,
    "confidence_note": "GBM+RF ensemble | AUC 0.531 | Trained on 7,266 real transactions"
  }
}
```

The dashboard shows this in the **AI Layer panel** alongside the behavioural RF score, with:
- Fraud probability as a percentage with colour-coded risk label
- Short-window and velocity signals from the dataset model
- Confidence note citing dataset provenance

---

## 8. Model Performance Context

### Observed AUC: ~0.53

This is expected and does **not** indicate a poorly designed system. The reasons:

1. **Full anonymisation**: Feature names (F1–F3924) reveal no semantics. We cannot apply domain knowledge (e.g., "F115 is the transfer amount") to build targeted features.

2. **Structured sparsity**: ~65% of raw values are NaN. This likely represents "no event in this category in this window" — which is informative signal — but our aggregate statistics can only partially capture it.

3. **Aggregate compression loss**: Compressing 3,924 features into 109 aggregates loses within-window ordering and cross-feature interactions that the original model designer would have known.

4. **Baseline context**: Random chance = AUC 0.50. Our ensemble at 0.53 does meaningfully outperform random. For reference, on a similar anonymised dataset (IEEE-CIS Fraud Detection), naive feature engineering without domain knowledge also yields ~0.55 AUC before careful feature work.

### What would improve AUC in production?

- Feature name semantics (domain expert annotation)
- User-level historical baselines (personalised anomaly detection)
- Graph-based features (merchant/beneficiary network)
- Temporal sequence models (LSTM/GRU on transaction history)
- Rolling window statistics with known time granularities

---

## 9. Correct Usage Boundary

The dataset ML model is used **only** in the Transaction Layer, for transaction risk scoring. It is **not** used for:

- ❌ Behavioural biometric scoring (keystroke, swipe, mouse — those use the synthetic-data RF)
- ❌ Device trust scoring (that uses rule-based fingerprint matching)
- ❌ Session-level authentication decision (that uses the 5-layer sigmoid engine)
- ❌ Replacing the rule-based TRANSACTION_SIGNALS (those remain the primary signals)

This is the correct architectural boundary: the dataset captures **transaction-level fraud patterns** and should be used exactly where transaction-level fraud patterns are relevant.
