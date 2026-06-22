# BehaviorShield

**AI-Driven Behavioural Authentication for Public Sector Digital Banking**

> 

---

## What It Does

BehaviorShield authenticates users **continuously throughout their session** — not just at login. Every keystroke, mouse movement, and transaction is scored in real time. A legitimate user never sees a prompt. A bot, credential-stuffed attacker, or social-engineering victim is escalated through four progressive security tiers, with funds held and the account secured before any money moves.



---

## Architecture

```
+-------------------------------------------------------------+
|  BROWSER / MOBILE APP                                       |
|                                                               |
|  Layer 0 . ThreatShield -------- phishing URL detected?      |
|            (pre-entry)           scam SMS scanned?           |
|                                   fake page signals?          |
|               |  injects suspicion pts into session          |
|               v                                              |
|  Layer 1 . Login Canvas -------- keystroke dwell/flight      |
|            (4-zone biometrics)   mouse jitter                |
|                                   swipe dynamics              |
|               |                                              |
|               v                                              |
|  Layer 2 . Behavioural RF ------ 200-tree Random Forest      |
|            (session scoring)     12 live session features    |
|                                   EWMA personal baseline      |
|               |                                              |
|               v                                              |
|  Layer 3 . TX Fraud ML --------- GBM + RF ensemble           |
|            (transaction)         trained on borrowed PSB     |
|                                   mule-detection dataset      |
|                                   (used here as a transfer-   |
|                                   risk proxy signal, not as   |
|                                   a mule classifier)          |
|               |                                              |
|               v                                              |
|  Layer 4 . Escalation Engine --- Tier 0: full access         |
|            (4-tier response)     Tier 1: silent monitor      |
|                                   Tier 2: OTP + hold          |
|                                   Tier 3: bank review         |
|                                                                |
|  + Failed-Login Hardening ------ independent of score:       |
|    (PS-3 Goal 4)                 attempt 1 -> security phrase |
|                                   attempt 2 -> OTP required   |
|                                   attempt 3+ -> automated call|
|                                   + session freeze            |
+-------------------------------------------------------------+
```

---

## Key Design Decisions

**No false positives for legitimate users.** A legitimate user with normal keystroke timing, on a trusted device, sending a routine transfer to a known beneficiary, scores Tier 0 throughout their session. They see nothing unusual. The system is invisible unless something is actually wrong.

**Compound evidence drives escalation.** No single signal hard-blocks a session. A new beneficiary alone is a flag. Bot-speed typing alone is a flag. Both together with an unusual hour and a large amount anomaly — that's a freeze. The scoring engine requires corroboration.

**ThreatShield pre-conditions the session.** A user who arrives via a phishing link or pastes a suspicious SMS is already carrying elevated suspicion *before* they type a character. When they then attempt a large transfer, the pre-existing signal tips the combined score into Tier 3 immediately.

**The RF model refines, not overrides.** The behavioural Random Forest only influences scores in the ambiguous band (combined < 66). Once hard transactional and ThreatShield evidence has pushed the score into HIGH RISK or CRITICAL territory, the RF cannot veto it — it only saw keystroke timing, not the full picture.

**Failed-login hardening is score-independent (PS-3 Goal 4).** A correct password on the first attempt and a correct password on the fourth attempt are not treated the same. Each *blocked or rejected* login attempt increments a dedicated counter — completely separate from the behavioural suspicion score — that escalates the session through security phrase → mandatory OTP → automated verification call + freeze. This means an attacker who is behaviourally "clean" (e.g. a slow, careful manual attacker, not a bot) but keeps failing the login gate still gets hardened. This counter is cleared **only** by a backend-driven `reset()` (manual review) — never by any frontend action — so an attacker cannot erase their own failed-attempt history by refreshing the page or clicking a UI button.

---

## Scoring Engine

```
combined_score = sigmoid(b_raw, b_k, b_mid) x b_presence
               + sigmoid(t_raw, t_k, t_mid) x t_presence
               + synergy_term
               [+ RF_blend if combined < 66]

Tier 0  (0-45)   Full access. User notices nothing.
Tier 1  (46-65)  Security phrase activated. Silent monitoring.
Tier 2  (66-85)  OTP mandatory. Transaction held. Read-only.
Tier 3  (86-100) Account secured. Bank compliance review. Funds not moved.
```

Escalation is **dwell-gated** (mandatory time at each tier before advancing — cannot skip Tier 0->3). De-escalation is **streak-gated** (10 consecutive clean events before stepping down one tier). Tier 3 is **permanent** until manual review via `POST /reset/{profile}`.

**Failed-login ladder** (independent of the above, fires on `failed_login` events from `doLogin()`):

```
Attempt 1   +25 pts   -> security phrase tier activated
Attempt 2   +40 pts   -> OTP required on next attempt
Attempt 3+  +60 pts   -> automated verification call + session frozen
```

A visible badge on the login page ("Login attempts blocked: N — session hardened") surfaces this in real time for demo purposes.

---

## Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
# Server starts at http://localhost:8000
# API docs at http://localhost:8000/docs
```

```bash
# Frontend
cd frontend
python3 -m http.server 3000
# Then open http://localhost:3000/index.html
```

No database, no cloud dependency, no API keys. Runs entirely offline.

---

## Demo Scenarios

| Scenario | How to run | Expected result |
|---|---|---|
| Legitimate user | Select Arjun profile, type normally, transfer ₹8,000 to known payee | Tier 0 throughout, instant approval |
| Phishing chain | Paste a known-malicious test URL in ThreatShield box -> transfer ₹4.5L to new payee at 2am | Tier 3, frozen, bank review |
| Bot attack | Select Attacker profile (pre-set fast dwell), attempt large transfer | Escalates to Tier 3 via keystroke anomaly |
| Structuring | Make 4x ₹6,000 transfers to different new payees | Tier 0 -> 1 -> 2 -> 3 via velocity stacking |
| New device | Toggle device trust OFF in demo panel | Probationary mode, ₹10,000 transaction limit, lifts after 5 trusted sessions |
| Demo hour | Set transaction hour to 02:00 in transfer form | `unusual_hour` signal fires (+15 pts) |
| Failed login hardening | Trigger the security-phrase gate or a blocked login 3 times in a row | Badge shows escalating count; 3rd attempt -> automated call + freeze, independent of behavioural score |

---

## Project Structure

```
BehaviorShield/
├── backend/
│   ├── main.py              FastAPI server, WebSocket, all endpoints
│   ├── scorer.py            Core scoring engine (SuspicionScorer)
│   ├── threat_shield.py     Layer 0: URL / scam message / fake page detection
│   ├── tx_fraud_scorer.py   Transaction ML scoring (proxy signal, see Dataset section)
│   ├── profiles.py          Demo user profiles (Arjun, Attacker, New Device)
│   ├── train_model.py       Model training pipeline
│   ├── model.joblib         Behavioural RF (pre-trained, 200 trees)
│   ├── tx_fraud_model.joblib  Transaction GBM+RF ensemble (pre-trained)
│   ├── test_scorer.py       Test suite — 37 tests, all passing
│   ├── KNOWN_ANOMALIES.md   Documented edge cases and tradeoffs
│   └── requirements.txt
└── frontend/
    ├── index.html           Login page + 4-zone biometric canvas + failed-login badge
    ├── transfer.html        Transaction form + live risk display
    ├── dashboard.html       Analyst monitor — live signal feed
    ├── threatshield.html    ThreatShield standalone checker
    └── behaviorsignal.js    WebSocket SDK + ThreatShieldScanner
```

---

## API Reference

```
WS   /ws/{profile}                  WebSocket — live session scoring
POST /transaction/{profile}         Score a transaction attempt
POST /reset/{profile}               Reset session (manual review complete) —
                                     the ONLY way to clear failed_login_count
GET  /health                        Server health check
GET  /model/status                  Model info and AUC

POST /threat/check-url              Phishing URL analysis
POST /threat/check-message          Scam SMS/email/WhatsApp scan
POST /threat/check-page             Fake page signal evaluation
POST /threat/check-all              Combined threat check
GET  /threat/status                 ThreatShield module status

POST /demo/device-mode              Toggle device trust (demo control)
GET  /demo/device-mode              Current device mode
```

---

## Test Suite

```bash
cd backend
python3 test_scorer.py

# 37 passed, 0 failed
```

Tests cover: false-positive prevention, regression checks, amount sweep monotonicity, bot-speed typing detection, synergy scoring, fallback/tier consistency, de-escalation, Tier 3 permanence, ThreatShield injection, the full phishing-chain end-to-end scenario, automated-call escalation, per-beneficiary retry isolation, and the PS-3 Goal 4 failed-login hardening ladder (security phrase -> OTP -> automated call & freeze, reset-only clearing, false-positive immunity for clean sessions).

---

## Dataset

A real PSB transaction dataset (`DataSet__1_.csv`, ~9,082 records, 3,924 anonymised features) was provided for the **mule-account-detection** problem statement. BehaviorShield uses this dataset as a **borrowed proxy signal source for the transaction-risk layer** — it is not used here to build or claim a mule-account classifier, since that is a separate problem statement from the one this project addresses (continuous behavioural authentication / ATO defence). The resulting model's modest AUC reflects the limits of applying a mule-detection feature set to a differently-scoped transaction-risk task, not an attempt at solving mule detection itself. See `DATASET_INTEGRATION.md` for the full feature mapping and the dataset's actual label column.

---

 · IIIT Senapati


