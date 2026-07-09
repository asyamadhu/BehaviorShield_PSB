# =============================================================
# scorer.py — BehaviorShield  (Fresh Build)
#
# SCORING ARCHITECTURE — FIVE CLEAN LAYERS
#
# ┌─────────────────────────────────────────────────────────┐
# │  LAYER 1 — RAW ACCUMULATION                             │
# │  b_raw and t_raw accumulate signal points uncapped.     │
# │  Uncapped so sigmoid has full range to work with.       │
# ├─────────────────────────────────────────────────────────┤
# │  LAYER 2 — ASYMMETRIC SIGMOID COMPRESSION               │
# │  Each layer compressed independently before combining.  │
# │  b: k=0.055, mid=85  → forgiving for 1–2 stray signals  │
# │  t: k=0.08,  mid=50  → strict for stacked tx anomalies  │
# │  Verified:                                               │
# │    Legit (1–2 signals) → combined <20  → Tier 0 ✓      │
# │    Attacker 5+ signals → combined 86+  → Tier 3 ✓      │
# ├─────────────────────────────────────────────────────────┤
# │  LAYER 3 — PRESENCE-WEIGHTED COMBINE WITH SYNERGY       │
# │  Additive but neither layer dominates unfairly.         │
# │  Compound penalty fires only when both layers elevated.  │
# ├─────────────────────────────────────────────────────────┤
# │  LAYER 4 — RANDOM FOREST SECOND OPINION (70/30)         │
# │  12-feature session-level RF from train_model.py.       │
# │  Final = 0.70 × sigmoid + 0.30 × RF×100                 │
# │  RF only activates after 5+ keystrokes (cold-start).    │
# │  Degrades to sigmoid-only if model.joblib absent.       │
# ├─────────────────────────────────────────────────────────┤
# │  LAYER 5 — STAGED ESCALATION WITH MANDATORY DWELLS      │
# │  Score can spike but DISPLAYED tier advances slowly:    │
# │    Tier 0→1: instant  (score just crossed 46)           │
# │    Tier 1→2: 8 sec    (security phrase must be shown)   │
# │    Tier 2→3: 5 sec    (OTP must fire first)             │
# │  Progressive hardening is NEVER skipped.                │
# └─────────────────────────────────────────────────────────┘
#
# DEVICE TRUST ENGINE
#   trusted      → no penalty, full access
#   probationary → +20 pts, tx_limit applied, harden forced
#   unknown      → +20 pts, tx_limit applied, harden forced
#
# RBI-COMPLIANT FRAMING
#   All responses say "bank verification required" not "blocked".
#   Bank raises the flag — bank makes the decision.
# =============================================================

import os
import re
import math
import time
import random
import hashlib
import secrets
from datetime import datetime
from typing import Optional

import numpy as np


# ── LOAD TRANSACTION FRAUD MODEL (PSB HACKATHON DATASET) ─────
# tx_fraud_scorer provides a real-data-backed fraud probability
# trained on the PSB Hackathon 2026 dataset (9,082 labelled
# banking transactions). It augments the rule-based TRANSACTION_
# SIGNALS with a learned ML signal, contributing to t_raw.
try:
    from tx_fraud_scorer import get_tx_fraud_prob, score_to_points as _tx_pts, model_summary as _tx_summary
    _tx_fraud_enabled = True
    _tx_meta = _tx_summary()
    print(f"[BehaviorShield] TX fraud model: "
          f"ensemble AUC={_tx_meta.get('ensemble_auc',0):.3f} | "
          f"{_tx_meta.get('n_train',0):,} training samples")
except Exception as _e:
    _tx_fraud_enabled = False
    print(f"[BehaviorShield] TX fraud scorer unavailable ({_e})")
    def get_tx_fraud_prob(_ctx):
        return {"fraud_prob": 0.0, "fraud_pct": 0.0, "risk_label": "UNKNOWN",
                "model_active": False, "win1_mean": 0.0, "win2_std": 0.0,
                "confidence_note": ""}
    def _tx_pts(_p): return 0


# ── LOAD BEHAVIOURAL SESSION MODEL ───────────────────────────
_rf_model   = None
_rf_enabled = False

try:
    import joblib
    _path   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")
    _bundle = joblib.load(_path)
    _rf_model   = _bundle["model"]
    _rf_enabled = True
    print(f"[BehaviorShield] RF model loaded — {_rf_model.n_estimators} trees")
except Exception as e:
    print(f"[BehaviorShield] RF model unavailable ({e}) — sigmoid-only mode")


# ── SIGMOID PARAMETERS ───────────────────────────────────────
_B_K   = 0.055   # DO NOT CHANGE — tuned so 1–2 stray signals stay Tier 0
_B_MID = 85      # DO NOT CHANGE

_T_K   = 0.08
_T_MID = 50

# Fusion weights — sigmoid backbone, RF second opinion
_W_SIG = 0.70
_W_RF  = 0.30

# Mandatory seconds at each tier before advancing
DWELL = {0: 0, 1: 8, 2: 5, 3: 0}

# De-escalation tuning (fix #4).
# A session steps DOWN one tier when raw_tier has been at least
# DEESCALATION_MARGIN tiers below _shown_tier for DEESCALATION_STREAK
# consecutive process_event() calls. This lets a session that was
# briefly elevated (e.g. one large transaction, since decayed via the
# 0.97/0.98 per-event decay) return to normal status over time, while
# Tier 3 (_frozen) remains permanent until reset() (manual review).
DEESCALATION_MARGIN  = 1   # raw_tier must be <= shown_tier - this value
DEESCALATION_STREAK  = 10  # consecutive qualifying events required

# ── KBA (Knowledge-Based Authentication) TUNING ──────────────────────
# Layered on top of the existing Tier-2 OTP gate as a second, independent
# factor. Deliberately SCORE-INDEPENDENT escalation (mirrors failed_login
# above) — verified against the live sigmoid that 2 wrong answers alone
# stay within the Tier-2 band (~79, well under the 86 Tier-3 line), so
# the automated-call trigger and the freeze-on-call-failure are explicit
# flags, not something we lean on point values to force via the curve.
KBA_FAIL_PTS       = (20, 35)   # attempt 1, attempt 2+ (b_raw points)
KBA_CALL_FAIL_PTS  = 60         # automated call verification failed -> freeze
MIN_KBA_QUESTIONS  = 5          # registration requirement (enforced by caller)

# Probation (post-call-pass) transaction cap, as a fraction of the
# account's normal average transfer — capped further by the existing
# new-device limit if that happens to be lower.
PROBATION_CAP_FRACTION = 0.5

# Consecutive CLEAN sessions (never exceeded Tier 1) required before
# probation lifts. Lives in main.py's ACCOUNT_STATE (cross-session),
# not here — this constant documents the contract for that logic.
CLEAN_SESSIONS_TO_CLEAR = 3

# ── PROGRESSIVE TRANSACTION RESPONSE (score-independent) ───────
# A flagged transaction's response depends on ATTEMPT COUNT for that
# specific beneficiary, not on combined_score. Score-based thresholds
# (cs >= 86 etc.) decide whether a transaction is flagged AT ALL; once
# flagged, these constants decide the RESPONSE SEVERITY based on how
# many times this exact beneficiary has previously been flagged.
KYC_IMMEDIATE_AMOUNT_RATIO = 25.0
# A first-ever attempt whose amount is >= 25x the user's average
# transfer amount skips the OTP step entirely and goes straight to
# compliance review/KYC — too large to risk a recoverable OTP retry,
# regardless of attempt count. (avg=8000 -> 200,000 threshold, matching
# the team's originally-specified ₹2,00,000 cutoff for the default profile.)


# ── MATH HELPERS ─────────────────────────────────────────────

def _normalize_kba_answer(raw: str) -> str:
    """Normalise a security-question answer before hashing/comparison:
    lowercase, strip punctuation, collapse whitespace. So 'Mumbai',
    ' mumbai ', 'MUMBAI!' all match the same stored hash. Trade-off:
    this tolerates formatting differences but NOT typos — a hashed
    answer can't be fuzzy-matched, since hashing destroys distance
    information. Acceptable for a hackathon POC; production would add
    tolerance by storing a few accepted normalized variants at
    registration rather than fuzzy-matching against a single hash.
    """
    s = (raw or "").strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def hash_kba_answer(raw: str, salt: str = None) -> str:
    """Hash a security-question answer for storage in a profile.
    Returns 'salt:hexhash'. Call once at registration time; never store
    the plaintext answer. (Used to generate profiles.py's demo hashes —
    regenerate with this function if the demo answers ever change.)
    """
    if salt is None:
        salt = secrets.token_hex(8)
    digest = hashlib.sha256((salt + _normalize_kba_answer(raw)).encode("utf-8")).hexdigest()
    return f"{salt}:{digest}"


def _check_kba_answer(raw: str, stored: str) -> bool:
    """Verify a submitted answer against a stored 'salt:hexhash' string."""
    if not stored or ":" not in stored:
        return False
    salt, digest = stored.split(":", 1)
    candidate = hashlib.sha256((salt + _normalize_kba_answer(raw)).encode("utf-8")).hexdigest()
    return secrets.compare_digest(candidate, digest)


def _sigmoid(x: float, k: float, mid: float) -> float:
    """Compress raw [0..∞] → [0..100], anchored so sigmoid(0) = 0."""
    raw      = 100.0 / (1.0 + math.exp(-k * (x - mid)))
    baseline = 100.0 / (1.0 + math.exp(-k * (0.0 - mid)))
    scaled   = (raw - baseline) / (1.0 - baseline / 100.0)
    return max(0.0, min(100.0, scaled))


def _combine(b_raw: float, t_raw: float, amount_ratio: float = 1.0) -> float:
    """
    Presence-weighted combination with synergy boost.

    Parameters
    ----------
    b_raw, t_raw : raw accumulated suspicion points (behavioural / transaction)
    amount_ratio : transaction_amount / profile.avg_transfer_amount
                   (1.0 if no transaction in this event — i.e. neutral/unknown,
                   does not relax the gate below)

    Guarantees
    ──────────
    • Legit small tx on trusted device, zero biometrics
        → combined stays low, Tier 0 (the original ₹100/new-bene fix).
    • Large/anomalous tx with zero biometrics (e.g. a script hitting the
      API directly, bypassing the frontend) → NOT capped. combined_score
      reflects t_score directly and CAN reach Tier 2/3. This is the
      "account-takeover via direct API" pattern and must escalate hard.
    • Both layers elevated simultaneously → Tier 3 fast via synergy.

    ── Tier-3 gate (amount-aware) ─────────────────────────────────────
    The original gate capped combined at 65.9 (top of Tier 1) whenever
    b_p < 0.05 (i.e. b_raw < 1.5 — no behavioural/biometric telemetry at
    all). That fixed the ₹100/new-beneficiary/trusted-device false
    positive, but also silently capped a ₹4,50,000/new-beneficiary/2am
    transaction sent with zero behavioural telemetry — arguably the most
    realistic ATO pattern (a script that skips the frontend's biometric
    capture entirely).

    The gate now only applies when BOTH:
      (a) b_p < 0.05  (no behavioural signal), AND
      (b) amount_ratio <= 1.0  (transaction amount is at/below the user's
          normal average — i.e. nothing about the amount itself is
          anomalous)

    When (a) holds but (b) does not — i.e. no behavioural data AND a
    large/anomalous amount — the cap does NOT apply, and combined_score
    is allowed to reach Tier 2/3 purely on t_score. This is intentional:
    a large anomalous transfer with literally zero biometric corroboration
    is not "less suspicious" than one with some biometric noise — if
    anything it is MORE consistent with a scripted attack that bypassed
    the browser's signal capture entirely.

    Net effect of the gate, by case:
      ₹100,  new bene, trusted device, b_raw≈0   → capped, Tier 0   ✅ (orig. fix preserved)
      ₹450k, new bene, 2am,            b_raw≈0   → NOT capped, can reach Tier 2/3 ✅ (regression fixed)
      4x ₹3,000 structuring + velocity, b_raw≈0  → NOT capped (see below)         ✅
      any amount + behavioural signal present (b_p>=0.05) → gate never applies (full scoring)

    ── Structuring/velocity exemption ──────────────────────────────────
    A structuring attack (many transfers, each individually <= avg
    amount, but with repeated new-beneficiary/velocity signals stacking
    t_raw across the session) would otherwise satisfy BOTH gate
    conditions on every individual transfer (amount_ratio <= 1.0 each
    time) and stay capped at Tier 1 forever — even as t_raw climbs past
    100+ from accumulated signals.

    The gate is therefore ALSO bypassed when t_raw >= T_RAW_GATE_BYPASS
    (70 — the point where t_c alone is already ~83, solidly HIGH-RISK).
    At that level, the transaction layer has accumulated enough
    INDEPENDENT signal repetitions (not just one large amount) that
    "no behavioural corroboration" is no longer a reason to suppress
    escalation — repeated structuring IS the corroborating pattern.
    """
    T_RAW_GATE_BYPASS = 70.0

    b_c = _sigmoid(b_raw, _B_K, _B_MID)
    t_c = _sigmoid(t_raw, _T_K, _T_MID)

    b_p = min(1.0, b_raw / 30.0)   # presence: how much behavioural signal exists
    t_p = min(1.0, t_raw / 25.0)

    b_norm = max(0.0, (b_c - 20.0) / 60.0)   # elevation above baseline
    t_norm = max(0.0, (t_c - 20.0) / 60.0)

    synergy = b_p * t_p * b_norm * t_norm * 30.0   # fires only when BOTH elevated

    combined = (
        b_p * b_c * (1.0 - t_p * 0.3)
        + t_p * t_c * (1.0 - b_p * 0.3)
        + synergy
    )

    # ── Amount-aware Tier-3 gate (with structuring/velocity exemption) ──
    # Cap to Tier-1 ceiling ONLY when there's no behavioural signal AND
    # the transaction amount is unremarkable AND t_raw hasn't otherwise
    # accumulated enough independent evidence on its own.
    if b_p < 0.05 and amount_ratio <= 1.0 and t_raw < T_RAW_GATE_BYPASS:
        combined = min(combined, 65.9)

    return min(100.0, combined)


# ── SIGNAL TABLES ────────────────────────────────────────────

BEHAVIOURAL_SIGNALS = {
    "paste_password":          {"pts": 40, "label": "Password / username field pasted"},
    "new_device_location":     {"pts": 30, "label": "New device + new location (combined)"},
    "typing_speed_anomaly":    {"pts": 25, "label": "Typing speed >2σ above personal baseline"},
    "mouse_jitter":            {"pts": 20, "label": "Mouse jitter — RAT / remote control pattern"},
    "swipe_mismatch":          {"pts": 15, "label": "Swipe pattern mismatch"},
    "nav_anomaly":             {"pts": 10, "label": "Navigation pattern anomaly"},
    "unusual_session_time":    {"pts":  8, "label": "Unusual session timing"},
    "vpn_detected":            {"pts": 15, "label": "VPN / proxy environment detected"},
    "concurrent_session":      {"pts": 25, "label": "Simultaneous session on another device"},
    "otp_device_mismatch":     {"pts": 25, "label": "OTP-device inconsistency detected"},
    "remote_access_tool":      {"pts": 40, "label": "Remote-access tool indicators (RAT)"},
    "no_idle_movement":        {"pts":  8, "label": "No idle micro-movement — bot pattern"},
    "keystroke_dwell_anomaly": {"pts": 20, "label": "Keystroke dwell far below user baseline"},
}

TRANSACTION_SIGNALS = {
    "amount_anomaly":          {"pts": 35, "label": "Amount >3σ above personal baseline",              "response": "Bank verification required — step-up KYC before transaction proceeds"},
    "new_beneficiary":         {"pts": 25, "label": "New / unrecognised beneficiary",                  "response": "Bank needs to verify beneficiary details — 10-min cooling window applied"},
    "beneficiary_then_xfer":   {"pts": 40, "label": "Beneficiary added then immediately transferred",  "response": "Bank flagged for compliance review — immediate transfer after adding beneficiary"},
    "velocity_spike":          {"pts": 30, "label": "Velocity spike — >3 transfers in 5 minutes",      "response": "Bank has temporarily held this transaction for velocity review"},
    "unusual_hour":            {"pts": 15, "label": "Transaction outside user's usual hours",           "response": "Silent monitoring — bank will verify if combined with other signals"},
    "beneficiary_type_shift":  {"pts": 20, "label": "Beneficiary type shift (domestic → international)","response": "Bank requires additional KYC for international transfer"},
    "multiple_small_transfers":{"pts": 25, "label": "Multiple small transfers below threshold limits",  "response": "Bank flagged for structuring pattern review"},
    "large_new_beneficiary":   {"pts": 55, "label": "Large transfer to new beneficiary (compound)",    "response": "Bank has placed this transaction under mandatory compliance review"},
}

TIERS = {
    0: {"label": "NORMAL",    "color": "#22c55e", "band": "0 – 45",
        "action":  "Full access. Profile updated via EWMA. User notices nothing.",
        "response": "full_access"},
    1: {"label": "ELEVATED",  "color": "#eab308", "band": "46 – 65",
        "action":  "Silent monitoring. Sampling doubles. Security phrase activated. Retry delay +2 sec.",
        "response": "silent_monitor"},
    2: {"label": "HIGH RISK", "color": "#f97316", "band": "66 – 85",
        "action":  "OTP mandatory. Score must drop below 35 before access restored. Read-only.",
        "response": "otp_readonly"},
    3: {"label": "CRITICAL",  "color": "#ef4444", "band": "86 – 100",
        "action":  "Session placed under bank security review. Fraud team notified. 30-min hold.",
        "response": "bank_review"},
}


# ═══════════════════════════════════════════════════════════════
# DEVICE TRUST ENGINE
# ═══════════════════════════════════════════════════════════════

class DeviceTrustEngine:
    """
    Device trust evaluation.

    DEMO MODE (default on):  every fingerprint is treated as trusted.
        Set DEMO_ALL_TRUSTED = False (or toggle via /demo/device-mode)
        to restore real new-device probationary logic for presentations.

    Why demo mode:
        In a live demo we don't want the fingerprint of the judge's laptop
        to accidentally trigger probationary mode and confuse the narrative.
        Flip the toggle to show new-device behaviour deliberately.
    """

    # ── Global demo toggle ─────────────────────────────────────
    # True  → every device is trusted (presentation default)
    # False → real fingerprint matching (new device = probationary)
    DEMO_ALL_TRUSTED: bool = True

    def __init__(self, profile: dict):
        self.devices   = profile.get("trusted_devices", [])
        self.threshold = profile.get("device_trust_threshold", 0.6)
        self.tx_limit  = profile.get("new_device_tx_limit", 10000)

    def evaluate(self, fingerprint: str) -> dict:
        # ── Demo shortcut: all devices trusted ────────────────
        if DeviceTrustEngine.DEMO_ALL_TRUSTED:
            return {
                "status":          "trusted",
                "label":           "Demo mode — all devices trusted",
                "sessions":        99,
                "trust_score_raw": 1.0,
                "suspicion_pts":   0,
                "tx_limit":        None,
                "force_harden":    False,
                "message":         "Demo mode: device trust bypassed. Toggle /demo/device-mode to enable real checks.",
                "demo_mode":       True,
            }

        # ── Real fingerprint matching ──────────────────────────
        match = next((d for d in self.devices if d["fingerprint"] == fingerprint), None)

        if match and match["trust_score"] >= self.threshold:
            return {
                "status":          "trusted",
                "label":           match.get("label", "Known device"),
                "sessions":        match.get("sessions", 0),
                "trust_score_raw": match["trust_score"],
                "suspicion_pts":   0,
                "tx_limit":        None,
                "force_harden":    False,
                "message":         f"Trusted device — {match.get('sessions', 0)} sessions",
                "demo_mode":       False,
            }
        elif match:
            return {
                "status":          "low_trust",
                "label":           match.get("label", "Low-trust device"),
                "sessions":        match.get("sessions", 0),
                "trust_score_raw": match["trust_score"],
                "suspicion_pts":   20,
                "tx_limit":        self.tx_limit,
                "force_harden":    True,
                "message":         "Device has low trust score — probationary monitoring",
                "demo_mode":       False,
            }
        else:
            return {
                "status":          "new",
                "label":           "Unrecognised device",
                "sessions":        0,
                "trust_score_raw": 0.0,
                "suspicion_pts":   20,
                "tx_limit":        self.tx_limit,
                "force_harden":    True,
                "message":         "New unrecognised device — probationary state activated",
                "demo_mode":       False,
            }


# ═══════════════════════════════════════════════════════════════
# SESSION FEATURES  (feeds the Random Forest)
# ═══════════════════════════════════════════════════════════════

class SessionFeatures:
    MIN_KEYSTROKES = 5

    def __init__(self, profile: dict):
        self.profile        = profile
        self._dwells        = []
        self._flights       = []
        self._wpms          = []
        self._jitters       = []
        self.paste_events   = 0
        self.concurrent     = 0
        self.nav_score      = 0.0
        self.time_score     = 0.0
        self.amount_ratio   = 1.0
        self.new_bene       = 0
        self.hour_deviation = 0.0
        self.device_trust   = 1.0

    @property
    def ready(self):
        return len(self._dwells) >= self.MIN_KEYSTROKES

    def record_keystroke(self, dwell_ms, flight_ms):
        if dwell_ms  and dwell_ms  > 0: self._dwells.append(float(dwell_ms))
        if flight_ms and flight_ms > 0: self._flights.append(float(flight_ms))

    def record_wpm(self, wpm):
        if wpm > 0: self._wpms.append(float(wpm))

    def record_jitter(self, j):
        self._jitters.append(float(j))

    def record_paste(self):
        self.paste_events += 1

    def record_device(self, ds: dict):
        self.device_trust = ds.get("trust_score_raw", 1.0)

    def record_transaction(self, amount, is_new_bene, hour, profile):
        avg = profile.get("avg_transfer_amount", 8000)
        self.amount_ratio   = amount / avg if avg > 0 else 1.0
        self.new_bene       = 1 if is_new_bene else 0
        h0 = profile.get("usual_hour_start", 8)
        h1 = profile.get("usual_hour_end",   22)
        self.hour_deviation = float(max(0, h0 - hour, hour - h1))

    def to_vector(self):
        return np.array([[
            float(np.mean(self._dwells))   if self._dwells   else 95.0,
            float(np.mean(self._flights))  if self._flights  else 140.0,
            float(np.mean(self._wpms))     if self._wpms     else 52.0,
            float(np.mean(self._jitters))  if self._jitters  else 1.8,
            float(self.paste_events),
            self.amount_ratio,
            float(self.new_bene),
            self.hour_deviation,
            self.device_trust,
            float(self.concurrent),
            self.nav_score,
            self.time_score,
        ]])

    def to_dict(self):
        v = self.to_vector()[0]
        names = ["avg_dwell_ms","avg_flight_ms","wpm","mouse_jitter_score",
                 "paste_events","amount_ratio","is_new_beneficiary",
                 "tx_hour_deviation","device_trust_score","concurrent_sessions",
                 "nav_anomaly_score","session_time_score"]
        return {n: round(float(v[i]), 3) for i, n in enumerate(names)}


# ═══════════════════════════════════════════════════════════════
# SUSPICION SCORER — MAIN ENGINE
# ═══════════════════════════════════════════════════════════════

class SuspicionScorer:

    def __init__(self, profile: dict):
        self.profile = profile

        # Raw accumulators — uncapped (sigmoid needs full range)
        self.b_raw = 0.0
        self.t_raw = 0.0

        # Signal logs (separate objects — no aliasing)
        self.signal_log   = []
        self.b_signal_log = []
        self.t_signal_log = []

        self.pending_tx = None

        # ── New-beneficiary attempt tracking (independent of score) ────
        # Maps beneficiary identifier -> number of times a transaction to
        # that beneficiary has been flagged as suspicious (is_new beneficiary
        # OR amount_anomaly fired). This is DELIBERATELY separate from
        # combined_score: score is a blended, decaying, multi-signal number
        # and is the wrong source of truth for "is this attempt #1 or #2 to
        # this specific account" — a session reset, score decay, or an
        # unrelated signal firing must never reset or corrupt this count.
        # Survives for the lifetime of this SuspicionScorer instance and is
        # only cleared by reset() (manual review), never by UI actions.
        self.flagged_attempts = {}
        self.failed_login_count = 0  # PS-3 Goal 4: counts blocked login attempts

        # Most recent transaction's amount_ratio (amount / avg_transfer_amount).
        # Defaults to 0.0 — meaning "no transaction in this session yet",
        # which is treated as <= 1.0 by the Tier-3 gate (i.e. gate CAN apply
        # if there's also no behavioural signal — matches "nothing anomalous
        # has happened" for a pure-login session with no transaction).
        self._last_amount_ratio = 0.0

        # TX fraud ML result (populated after each transaction_attempt)
        self._tx_ml_result = None

        # Sub-engines
        self.device_engine  = DeviceTrustEngine(profile)
        self.device_state   = None
        self._tx_limit      = None

        self.features       = SessionFeatures(profile)
        self._ai_prob       = 0.0
        self._ai_active     = False

        # Staged escalation
        self._shown_tier    = 0
        self._tier_times    = {}   # {tier: timestamp when entered}

        # De-escalation tracking (fix #4)
        # Counts consecutive process_event() calls where raw_tier is at
        # least DEESCALATION_MARGIN tiers below _shown_tier. Once this
        # reaches DEESCALATION_STREAK, _shown_tier steps down by ONE
        # (never skipping, mirroring the escalation rule in reverse).
        # Tier 3 (_frozen) NEVER de-escalates automatically — that
        # requires the manual reset() path per the design doc.
        self._deescalation_streak = 0

        # Progressive hardening flags
        self._harden        = False   # security phrase shown
        self._otp           = False
        self._call          = False   # automated verification call initiated
        self._frozen        = False
        self._force_harden  = False   # set by device engine

        # Freezes b_raw (no decay, no new points) the entire time an
        # unresolved hardening gate is being shown to the user — the
        # security phrase, a KBA question, or the automated call.
        # Without this, the frontend's 3s mouse_idle heartbeat (see
        # behaviorsignal.js _idleMonitor) keeps calling process_event
        # purely as a keep-alive, and its unconditional decay line was
        # silently moving combined_score up and down by 10-20+ points
        # while someone simply sat there reading a question — observed
        # live drifting 68.6 -> 65.0 -> 57.5 -> 60.7 -> 57.0 over ~20s
        # of zero interaction. That could flip tier_label out of HIGH
        # RISK mid-read (hiding an unanswered KBA box), then an
        # idle-streak penalty could flip it back with a DIFFERENT
        # random question, and it silently discarded in-progress
        # phrase-field typing the same way. Set True the moment any
        # hardening tier is first reached (see state()); cleared only
        # by an explicit user response for whichever gate is active
        # (security_phrase submit, failed_login retry, or a KBA/call
        # verify) — see process_event()'s early-return guard and
        # verify_kba_answer()'s unlock at the end.
        self._gate_locked = False

        # ── KBA (security question) state ───────────────────────
        self.kba_failed_count = 0     # wrong answers this episode (question A/B only)
        self._kba_verified    = False # a Tier-2 question (A or B) was answered correctly
        self._call_pending    = False # 2 KBA fails -> awaiting the automated call's 3rd question
        self._asked_kba_ids   = []    # question ids already used this episode — never repeat one

        # ── Probation (post-call-pass) state ────────────────────
        # Sticky: NOT cleared by _on_tier_down / score decay. Only
        # cleared by main.py's cross-session ACCOUNT_STATE tracker once
        # CLEAN_SESSIONS_TO_CLEAR consecutive calm sessions have passed,
        # or by reset() (manual review).
        self._probation         = False
        self._probation_tx_limit = None
        # WHY probation was granted -- the tx-cap mechanism itself is
        # identical regardless of cause (same field, same clearing
        # logic), but the user-facing explanation must be honest about
        # which thing actually happened. "call_verification_passed" is
        # the original/default reason; "phishing_referrer_detected" is
        # granted immediately on connect (see grant_phishing_referrer_
        # probation below), with no KBA/call ever involved.
        self._probation_reason  = None

        # Cross-session count of how many times THIS account has been
        # flagged via a phishing-pattern referrer/entry link. Persisted
        # in main.py's ACCOUNT_STATE (mirrors kba_failed_count's
        # pattern) so a second occurrence within a short window is
        # treated as meaningfully more suspicious than the first --
        # one phished click is unlucky, two in close succession
        # suggests an active campaign specifically targeting this
        # account, not a one-off.
        self.phishing_referrer_flag_count = 0

        # Highest _shown_tier reached this session — main.py reads this
        # at disconnect to decide whether the session counted as "clean"
        # for probation-clearing purposes.
        self.max_shown_tier = 0

    # ── COMPUTED PROPERTIES ───────────────────────────────────

    @property
    def b_score(self):
        return _sigmoid(self.b_raw, _B_K, _B_MID)

    @property
    def t_score(self):
        return _sigmoid(self.t_raw, _T_K, _T_MID)

    @property
    def sigmoid_score(self):
        return _combine(self.b_raw, self.t_raw, self._last_amount_ratio)

    @property
    def combined_score(self):
        """
        Blend sigmoid_score with the behavioural RF model's fraud
        probability — but ONLY as a refinement within the
        NORMAL/ELEVATED band (sigmoid_score < 66).

        Rationale: the RF model is trained on keystroke/mouse/device
        features only — it has no visibility into ThreatShield flags or
        transaction-layer signals. Once sigmoid_score itself has already
        reached HIGH RISK/CRITICAL (>=66) based on that hard evidence
        (e.g. a CRITICAL phishing-URL flag + a ₹4.5L transfer to a new
        beneficiary at 2am, sigmoid_score=93.75), the RF's opinion about
        "this keystroke pattern looks like normal typing" must not be
        allowed to silently veto that — the old 70/30 blend dragged a
        93.75 down to 65.9 (CRITICAL -> ELEVATED) for exactly this
        reason (KNOWN_ANOMALIES.md #5).

        Below 66, the score is still ambiguous/low-signal and the RF
        blend continues to provide useful fine-tuning as before.
        """
        s = self.sigmoid_score
        if _rf_enabled and self._ai_active and s < 66.0:
            return min(100.0, _W_SIG * s + _W_RF * self._ai_prob * 100.0)
        return s

    # ── TIER LOGIC ────────────────────────────────────────────

    def _raw_tier(self):
        c = self.combined_score
        if c < 46: return 0
        if c < 66: return 1
        if c < 86: return 2
        return 3

    def _advance_tier(self):
        """
        Walk tiers one at a time (escalation), then check de-escalation.
        Only advances when mandatory dwell elapsed.
        Cannot skip — progressive hardening always fires.

        De-escalation (fix #4): if raw_tier stays at least
        DEESCALATION_MARGIN below _shown_tier for DEESCALATION_STREAK
        consecutive calls, step _shown_tier down by ONE (one-step-at-a-
        time, mirroring escalation). Tier 3 (_frozen) never de-escalates
        automatically — manual reset() is required, per the design doc's
        "skip-proof escalation, frozen sessions require manual review"
        guarantee.
        """
        if self._frozen:
            return 3

        raw = self._raw_tier()
        now = time.time()

        # ── Escalation (existing behaviour, unchanged) ─────────────
        while self._shown_tier < raw:
            cur = self._shown_tier
            if cur not in self._tier_times:
                self._tier_times[cur] = now
            elapsed  = now - self._tier_times[cur]
            required = DWELL.get(cur, 0)
            if elapsed >= required:
                self._shown_tier += 1
                self._tier_times[self._shown_tier] = now
                self._on_tier(self._shown_tier)
                self._deescalation_streak = 0   # reset streak — we just moved
            else:
                break

        # ── De-escalation (fix #4) ──────────────────────────────────
        if not self._frozen and self._shown_tier > 0:
            if raw <= self._shown_tier - DEESCALATION_MARGIN:
                self._deescalation_streak += 1
                if self._deescalation_streak >= DEESCALATION_STREAK:
                    self._shown_tier -= 1
                    self._tier_times[self._shown_tier] = now
                    self._deescalation_streak = 0
                    self._on_tier_down(self._shown_tier)
            else:
                self._deescalation_streak = 0

        return self._shown_tier

    def _on_tier(self, tier):
        self.max_shown_tier = max(self.max_shown_tier, tier)
        if tier == 1: self._harden = True
        if tier == 2:
            self._otp = True
            # Automated call applies to the UPPER portion of HIGH RISK
            # (cs >= 78), independent of whether a transaction is even
            # in progress — this lets a sufficiently suspicious LOGIN
            # alone (e.g. ThreatShield pre-flagged + bot-speed typing,
            # no transaction attempted yet) also trigger the call, not
            # just a flagged transaction. Mirrors the cs>=78 threshold
            # used in _fallback() for the transaction-specific path.
            if self.combined_score >= 78:
                self._call = True
        if tier == 3: self._frozen = True

    def _on_tier_down(self, new_tier):
        """
        Called when _shown_tier steps DOWN by one (fix #4).
        Clears progressive-hardening flags once the displayed tier drops
        below the level that originally required them.

        _force_harden (set by DeviceTrustEngine for new/low-trust devices)
        is intentionally NOT cleared here — that flag reflects an ongoing
        device-trust condition, not a transient score elevation, and is
        only cleared by a fresh device_check with a trusted result.
        """
        if new_tier < 1:
            self._harden = False
        if new_tier < 2:
            self._otp  = False
            self._call = False
        # new_tier can never be >= 3 here (Tier 3 never de-escalates),
        # so _frozen is never cleared by this path.

    def _time_to_next(self):
        if self._shown_tier >= 3:
            return 0.0
        entered  = self._tier_times.get(self._shown_tier, time.time())
        required = DWELL.get(self._shown_tier, 0)
        return round(max(0.0, required - (time.time() - entered)), 1)

    # ── EVENT PROCESSOR ───────────────────────────────────────

    @property
    def _kba_episode_active(self) -> bool:
        """True from the moment a KBA question has been served
        (next_kba_question) until it resolves — a correct answer
        (_kba_verified), or the automated call's explicit pass/fail
        (_probation or _frozen gets set by verify_kba_answer). Single
        source of truth for every place that needs to know "is there
        an unanswered challenge on screen right now" — used to hold
        both scoring (process_event) and the CRITICAL auto-freeze
        (state()) still until the episode's own explicit outcome
        decides what happens next."""
        return (
            len(self._asked_kba_ids) > 0
            and not self._kba_verified
            and not self._frozen
            and not self._probation
        )

    def process_event(self, event: dict) -> dict:
        t = event.get("type", "")

        # While a KBA/call episode is unresolved, hold the score
        # PERFECTLY still against ambient telemetry — neither eroding
        # (decay) nor accruing (new points) — until the user's actual
        # response (verify_kba_answer, a separate code path from this
        # one) resolves it. Earlier iterations of this fix let genuinely
        # "new" signals keep escalating the score during an open
        # episode, reasoning that continuous behavioural telemetry
        # shouldn't pause for a pending challenge — but that meant the
        # number on screen could still visibly climb while someone was
        # simply reading/typing an answer, which is exactly the
        # instability this whole fix exists to eliminate. `device_check`
        # is exempt (identity, not suspicion-scoring) since switching
        # devices mid-challenge is a legitimate thing to detect
        # regardless. This cannot be exploited to dodge scoring
        # indefinitely: the frontend blocks login until the active
        # question is answered, and the episode always terminates in an
        # explicit verified/frozen/probation outcome.
        if self._kba_episode_active and t != "device_check":
            return self.state()


        # Device fingerprint check
        if t == "device_check":
            fp = event.get("device_fingerprint", "unknown")
            ds = self.device_engine.evaluate(fp)
            self.device_state = ds
            self.features.record_device(ds)
            if ds["suspicion_pts"] > 0:
                self._add_b(ds["suspicion_pts"], ds["message"], layer="device")
            if ds["force_harden"]:
                self._force_harden = True
                self._harden       = True
                if 0 not in self._tier_times:
                    self._tier_times[0] = time.time()
            if ds["tx_limit"] is not None:
                # Probation cap is a floor while active — a device_check
                # result must not loosen it back up.
                if self._probation and self._probation_tx_limit is not None:
                    self._tx_limit = min(ds["tx_limit"], self._probation_tx_limit)
                else:
                    self._tx_limit = ds["tx_limit"]

        # Named behavioural signals
        elif t in BEHAVIOURAL_SIGNALS:
            sig = BEHAVIOURAL_SIGNALS[t]
            self._add_b(sig["pts"], sig["label"])
            # Feed RF features
            if t == "paste_password":      self.features.record_paste()
            if t == "concurrent_session":  self.features.concurrent = 1
            if t == "nav_anomaly":         self.features.nav_score  = min(1.0, self.features.nav_score + 0.2)
            if t == "unusual_session_time":self.features.time_score = min(1.0, self.features.time_score + 0.25)
            if t in ("mouse_jitter",):     self.features.record_jitter(0.2)

        # Live keystroke dynamics
        elif t == "keystroke":
            d = event.get("dwell_ms", 0)
            f = event.get("flight_ms")
            self.features.record_keystroke(d, f)
            base = self.profile.get("avg_dwell_ms", 95)
            if base > 0 and d > 0:
                # Only flag keystrokes that are FASTER than baseline — that is
                # the bot/credential-stuffing signal (e.g. 9ms vs 95ms baseline).
                # Keystrokes SLOWER than baseline (d > base) are normal human
                # behaviour — typing carefully, reading while typing, hesitating.
                # Flagging slow typing (ratio > 0.7 when d > base) was causing
                # false positives for every normal user who typed deliberately.
                if d < base:
                    ratio = (base - d) / base   # how far BELOW baseline
                    if   ratio > 0.85: self._add_b(25, f"Keystroke dwell {d:.0f}ms — {ratio:.1f}× below {base}ms baseline — bot-speed typing")
                    elif ratio > 0.60: self._add_b(12, f"Keystroke dwell {d:.0f}ms — fast typing anomaly vs {base}ms baseline")

        # Typing speed
        elif t == "typing_speed":
            wpm  = event.get("wpm", 0)
            base = self.profile.get("avg_wpm", 52)
            self.features.record_wpm(wpm)
            if   wpm > base * 2.5: self._add_b(25, f"Typing speed {wpm}wpm — {wpm/base:.1f}× above {base}wpm baseline")
            elif wpm > base * 1.8: self._add_b(12, f"Typing speed elevated — {wpm}wpm vs {base}wpm baseline")

        # Idle movement (bot detection)
        elif t == "mouse_idle":
            mv = event.get("micro_movement_px", 0)
            # behaviorsignal.js sends this every 3 seconds.
            # A human user who pauses to read or think naturally has mv=0
            # between polls — this is NOT suspicious on its own.
            # Only flag after 5 CONSECUTIVE zero-movement intervals (~15s
            # of complete stillness), then only every 10 more intervals
            # (~30s) so it doesn't spam-accumulate on a still user.
            #
            # Suppressed entirely while _gate_locked: this heartbeat
            # fires every 3s purely to keep the connection alive, even
            # while someone is reading an unresolved security phrase /
            # KBA question / call prompt. Without this guard, sitting
            # still to READ the challenge was itself penalized as
            # "possible automated session" after ~15s — the opposite of
            # what should happen while someone is deliberately being
            # careful about a security answer. Ordinary idle detection
            # (this same mv<2 tracking) still runs normally at all
            # other times.
            if not self._gate_locked:
                if mv < 2:
                    self._idle_streak = getattr(self, '_idle_streak', 0) + 1
                    if self._idle_streak == 5:
                        self._add_b(8, "No mouse movement for ~15s — possible automated session")
                    elif self._idle_streak > 5 and (self._idle_streak - 5) % 10 == 0:
                        self._add_b(8, "Sustained no mouse movement — possible automated session")
                else:
                    self._idle_streak = 0  # any real movement resets streak

        # Mouse jitter — feed RF and flag robotic movement
        elif t == "mouse_jitter":
            self.features.record_jitter(event.get("jitter_score", 0.2))
            # The behaviorsignal.js already gates this to 3 consecutive
            # low-jitter windows before sending. Here we add a session-level
            # cap: flag once (+20) on first detection, then only again after
            # 5 more events — prevents repeated firing from spamming b_raw
            # even if the JS sends multiple events in succession.
            self._jitter_flag_count = getattr(self, '_jitter_flag_count', 0) + 1
            if self._jitter_flag_count == 1:
                self._add_b(BEHAVIOURAL_SIGNALS["mouse_jitter"]["pts"],
                            BEHAVIOURAL_SIGNALS["mouse_jitter"]["label"])
            elif (self._jitter_flag_count - 1) % 5 == 0:
                self._add_b(BEHAVIOURAL_SIGNALS["mouse_jitter"]["pts"],
                            BEHAVIOURAL_SIGNALS["mouse_jitter"]["label"])

        # Named transaction signals
        elif t in TRANSACTION_SIGNALS:
            sig = TRANSACTION_SIGNALS[t]
            self._add_t(sig["pts"], sig["label"], sig["response"])

        # Full transaction evaluation
        elif t == "transaction_attempt":
            self._eval_transaction(event)

        # Mouse move — captures jitter for RF, flags bot-flat movement
        elif t == "mouse_move":
            jitter = event.get("jitter", event.get("jitter_score", 1.8))
            self.features.record_jitter(float(jitter))
            # Near-zero jitter = robotic/scripted movement — but only flag
            # after consecutive detections to avoid false positives from
            # a user moving their mouse in a brief straight line.
            if jitter < 0.3:
                self._move_jitter_streak = getattr(self, '_move_jitter_streak', 0) + 1
                if self._move_jitter_streak == 5:
                    self._add_b(20, f"Mouse movement robotically straight — RAT/scripted pattern detected")
            else:
                self._move_jitter_streak = 0

        # Paste event — password or username pasted
        elif t == "paste_event":
            field = event.get("field", "")
            self.features.record_paste()
            pts = 40 if field in ("password", "pass", "") else 20
            self._add_b(pts, f"Field '{field}' pasted — credential auto-fill / paste attack indicator")

        # ThreatShield (Layer 0) session-point injection.
        #
        # main.py's _inject_threat_pts() sends this event after
        # /threat/check-url, /threat/check-message, /threat/check-page,
        # or /threat/check-all detects a phishing URL, scam message, or
        # fake page and computes suspicion_pts_for_session (or
        # session_suspicion_pts for check-all). Before this branch
        # existed, the event fell through every elif with no match and
        # was silently dropped (decay still ran, but no points were
        # added) — ThreatShield's session-injection was a complete no-op.
        #
        # Points are added to b_raw via _add_b with layer="threat_shield"
        # (a distinct layer tag, alongside the existing "device" and
        # "behavioural" conventions) so the signal log / dashboard can
        # distinguish "this score increase came from a pre-login threat
        # detection" from ordinary keystroke/mouse biometrics. The
        # dashboard's signal-log currently routes any layer != "transaction"
        # to the behavioural ('b') column — see KNOWN_ANOMALIES.md for the
        # "threat_shield" layer's dashboard styling status.
        #
        # Using b_raw (not t_raw) is deliberate: a ThreatShield flag is
        # evidence about THIS SESSION/USER's risk context (arrived via a
        # phishing link, on a flagged page), analogous to other
        # behavioural-layer signals — not evidence about a specific
        # transaction. This also means a sufficiently large injection
        # (>=1.5 pts, i.e. b_p>=0.05) disables the Tier-3
        # "no behavioural corroboration" gate in _combine() for any
        # subsequent transaction in the same session — see point 5 of
        # the fix description / KNOWN_ANOMALIES.md.
        elif t == "threat_shield_signal":
            pts = event.get("pts", 0)
            if pts > 0:
                self._add_b(pts, event.get("label", "ThreatShield signal"),
                             layer="threat_shield")

        elif t == "failed_login":
            # PS-3 Goal 4: "Harden the login process after unsuccessful
            # login attempts." Fired by doLogin() when the attempt is
            # blocked (CRITICAL) or the security-phrase gate fires.
            # Each failed attempt injects direct b_raw points —
            # INDEPENDENT of combined_score — so the session hardens
            # progressively even when the password itself is correct:
            #   attempt 1: +25 pts -> security phrase tier
            #   attempt 2: +40 pts -> OTP tier
            #   attempt 3+: +60 pts -> automated call / freeze
            self.failed_login_count += 1
            n = self.failed_login_count
            # Retuned (was 25/40/60): the old values only ever pushed
            # combined_score to ~2 / ~22 / ~86 — attempt 2 never actually
            # crossed into the HIGH RISK band (66-86) on combined_score,
            # only the internal _otp flag flipped. Since the KBA gate on
            # the frontend keys off tier_label === 'HIGH RISK' (not the
            # _otp flag), attempt 2 never visibly showed a KBA challenge —
            # the UI jumped straight from NORMAL to CRITICAL on attempt 3.
            # These values are chosen (empirically, via _combine) so that:
            #   attempt 1 (+50) -> combined ~13,  stays NORMAL/ELEVATED
            #   attempt 2 (+60) -> combined ~75,  lands in HIGH RISK ->
            #                      tier_label flips, KBA gate now shows
            #   attempt 3 (+80) -> combined >=100, CRITICAL -> freeze
            pts = 50 if n == 1 else 60 if n == 2 else 80
            self._add_b(pts, f"Failed/blocked login attempt {n} — session hardened per PS-3 Goal 4")
            self._harden = True
            if n >= 2: self._otp = True
            if n >= 3: self._call = True

        # Natural decay — legit users drift to 0, attackers keep adding.
        # By this point, the early-return at the top of this function
        # already handled the KBA-episode case for every event type
        # except device_check (which is exempt from the freeze —
        # identity, not suspicion-scoring — but should still hold the
        # score steady like everything else during an episode). The
        # remaining case this line covers is Tier 1's phrase gate
        # (`_gate_locked` but NO active KBA episode yet): the idle
        # heartbeat must never move the score there either, but genuine
        # typing must still decay normally — that's the intentional
        # de-escalation path (test_deescalation_from_tier1_after_
        # sustained_clean_events), and collapsing it into a blanket
        # "any gate locked" suspension was tried and broke it.
        if not ((self._gate_locked and t == "mouse_idle") or self._kba_episode_active):
            self.b_raw = max(0.0, self.b_raw * 0.97)
            self.t_raw = max(0.0, self.t_raw * 0.98)

        # Update RF
        self._update_rf()

        return self.state()

    def _eval_transaction(self, ev: dict):
        amount  = ev.get("amount", 0)
        bene    = ev.get("beneficiary", "unknown")
        hour    = ev.get("hour", datetime.now().hour)
        is_new  = bene not in self.profile.get("known_beneficiaries", [])
        avg_amt = self.profile.get("avg_transfer_amount", 8000)
        h0      = self.profile.get("usual_hour_start", 8)
        h1      = self.profile.get("usual_hour_end", 22)

        self.features.record_transaction(amount, is_new, hour, self.profile)

        # Record amount_ratio for the Tier-3 gate in _combine().
        # amount_ratio > 1.0 means the transaction amount itself is at/above
        # the user's normal average — this disables the Tier-3 gate even
        # when b_raw == 0, so a large anomalous transfer with zero
        # behavioural telemetry can still reach Tier 2/3.
        self._last_amount_ratio = (amount / avg_amt) if avg_amt > 0 else 1.0

        # Device tx_limit check
        if self._tx_limit is not None and amount > self._tx_limit:
            self._add_t(35,
                f"₹{amount:,} exceeds new-device limit of ₹{self._tx_limit:,}",
                "Bank requires device verification before this transaction can proceed")

        fired = []
        if avg_amt > 0 and amount > avg_amt * 3:
            sig = TRANSACTION_SIGNALS["amount_anomaly"]
            self._add_t(sig["pts"], f"₹{amount:,} — {int(amount//avg_amt)}× above ₹{avg_amt:,} baseline", sig["response"])
            fired.append("amount")

        if is_new:
            sig = TRANSACTION_SIGNALS["new_beneficiary"]
            amount_ratio = (amount / avg_amt) if avg_amt > 0 else 1.0
            scale = min(1.0, max(0.15, amount_ratio ** 0.5))
            # Hour context: new beneficiary during normal banking hours
            # (8am–10pm) is more likely a legitimate first transfer.
            # Reduce to 40% weight during normal hours.
            # unusual_hour (+15) and compound (+55) still fire when both
            # conditions are met so genuine off-hours risk still escalates.
            hour_in_window = (h0 <= hour <= h1)
            hour_scale = 0.40 if hour_in_window else 1.0
            scaled_pts = round(sig["pts"] * scale * hour_scale, 1)
            hour_note = ", normal hours" if hour_in_window else ""
            self._add_t(scaled_pts,
                        f"{sig['label']} (scaled: {scaled_pts}pts{hour_note})",
                        sig["response"])
            fired.append("bene")

        if not (h0 <= hour <= h1):
            sig = TRANSACTION_SIGNALS["unusual_hour"]
            self._add_t(sig["pts"], f"Transfer at {hour:02d}:00 — outside usual {h0}–{h1}h", sig["response"])
            fired.append("hour")
        if "amount" in fired and "bene" in fired:
            sig = TRANSACTION_SIGNALS["large_new_beneficiary"]
            self._add_t(sig["pts"], "COMPOUND: large amount + new beneficiary", sig["response"])

        # ── DATASET-TRAINED ML FRAUD SCORE ───────────────────────────
        # Map live transaction context into PSB dataset feature space
        # and obtain an ensemble (GBM + RF) fraud probability.
        is_trusted_device = (
            self.device_state is not None and
            self.device_state.get("status") == "trusted" and
            self.device_state.get("trust_score_raw", 0) >= 0.7
        )
        tx_ctx = {
            "amount_ratio":    amount / avg_amt if avg_amt > 0 else 1.0,
            "hour_deviation":  float(max(0, h0 - hour, hour - h1)),
            "is_new_bene":     1 if is_new else 0,
            "velocity_3h":     min(1.0, self.t_raw / 100.0),
            "device_trust":    self.features.device_trust,
            # Use actual device trust for channel_risk — trusted device = low risk
            "channel_risk":    0.1 if is_trusted_device else
                               0.7 if (self._force_harden or not self.device_state
                                       or self.device_state.get("status") != "trusted") else 0.3,
            "bene_risk":       0.6 if is_new else 0.05,
            "session_entropy": min(1.0, self.b_raw / 120.0),
        }
        tx_ml = get_tx_fraud_prob(tx_ctx)
        self._tx_ml_result = tx_ml
        ml_pts = _tx_pts(tx_ml["fraud_prob"])

        # ── FIX: Cap ML signal on trusted device + small amount ───────
        # When device is fully trusted AND amount is well below avg,
        # the ML score is inflated by bene_risk alone (the model doesn't
        # know the amount is tiny because bene_risk=0.6 dominates win1_mean).
        # Apply a dampener so ML can still flag moderate risk but cannot
        # push a trivial transfer into Tier 2/3 on its own.
        # Dampener is 1.0 (no effect) when amount ≥ avg_amt.
        # Dampener reaches minimum 0.3 when amount is < 5% of avg_amt.
        if is_trusted_device and avg_amt > 0:
            amount_ratio_raw = amount / avg_amt
            ml_dampener = min(1.0, max(0.3, amount_ratio_raw ** 0.4))
        else:
            ml_dampener = 1.0   # no dampening for untrusted devices

        ml_pts_damped = round(ml_pts * ml_dampener)

        if ml_pts_damped > 0:
            dampener_note = (f", dampened {ml_dampener:.2f}× for trusted+small-amount"
                             if ml_dampener < 1.0 else "")
            self._add_t(
                ml_pts_damped,
                f"ML fraud score: {tx_ml['fraud_pct']:.0f}% [{tx_ml['risk_label']}] "
                f"(dataset model, win1={tx_ml['win1_mean']:.2f}{dampener_note})",
                f"Bank risk model flagged transaction — {tx_ml['risk_label'].lower()} probability"
            )
            fired.append("ml_fraud")

        self.pending_tx = {"amount": amount, "beneficiary": bene, "fired": fired,
                           "tx_ml": tx_ml}

        # ── Progressive escalation: increment beneficiary attempt count ──
        # Only count this as a flagged "attempt" if a NOVELTY/AMOUNT signal
        # fired — "bene" (new_beneficiary), "amount" (amount_anomaly), or
        # the "large_new_beneficiary" compound. Deliberately EXCLUDES
        # "ml_fraud": the dataset-trained ensemble is a continuous
        # background-calibration signal (see KNOWN_ANOMALIES — its AUC is
        # only ~0.53 on the anonymised dataset) and can nudge the score on
        # almost any transaction, including a known beneficiary at a
        # perfectly normal amount. If ml_fraud alone counted as a "flagged
        # attempt", a legitimate user repeating normal transactions to a
        # KNOWN beneficiary would incorrectly accumulate retry history and
        # eventually get escalated to compliance review for doing nothing
        # wrong — exactly the false-positive failure mode this feature
        # must avoid per the system's core objective (smooth for genuine
        # users, tough for attackers).
        novelty_signals_fired = [f for f in fired if f in ("bene", "amount", "large_new_beneficiary")]

        avg_amt_for_kyc = self.profile.get("avg_transfer_amount", 8000)
        amount_ratio_for_kyc = (amount / avg_amt_for_kyc) if avg_amt_for_kyc > 0 else 1.0
        is_very_large = amount_ratio_for_kyc >= KYC_IMMEDIATE_AMOUNT_RATIO

        if novelty_signals_fired:
            prior_attempts = self.flagged_attempts.get(bene, 0)
            self.flagged_attempts[bene] = prior_attempts + 1
        else:
            prior_attempts = 0

        # is_retry: this beneficiary has been flagged (on novelty/amount
        # grounds) before THIS attempt. force_immediate_kyc: amount alone
        # is large enough to skip OTP even on a genuine first attempt.
        self.pending_tx["is_retry"]            = bool(novelty_signals_fired) and prior_attempts > 0
        self.pending_tx["attempt_number"]      = self.flagged_attempts.get(bene, 0) if novelty_signals_fired else 0
        self.pending_tx["force_immediate_kyc"] = is_very_large

    # ── RF UPDATE ─────────────────────────────────────────────

    def _update_rf(self):
        if not _rf_enabled or not self.features.ready:
            return
        self._ai_active = True
        try:
            self._ai_prob = float(_rf_model.predict_proba(self.features.to_vector())[0][1])
        except Exception:
            pass

    # ── SCORE ADDERS ─────────────────────────────────────────

    def _add_b(self, pts: float, label: str, layer: str = "behavioural"):
        self.b_raw = min(500.0, self.b_raw + pts)
        e = {"layer": layer, "signal": label, "pts": f"+{int(pts)}",
             "b_score": round(self.b_score, 1), "combined": round(self.combined_score, 1)}
        self.signal_log.insert(0, e);   self.signal_log   = self.signal_log[:20]
        self.b_signal_log.insert(0, e); self.b_signal_log = self.b_signal_log[:10]

    def _add_t(self, pts: float, label: str, response: str):
        self.t_raw = min(500.0, self.t_raw + pts)
        e = {"layer": "transaction", "signal": label, "pts": f"+{int(pts)}",
             "response": response, "t_score": round(self.t_score, 1),
             "combined": round(self.combined_score, 1)}
        self.signal_log.insert(0, e);   self.signal_log   = self.signal_log[:20]
        self.t_signal_log.insert(0, e); self.t_signal_log = self.t_signal_log[:10]

    # ── KBA (SECURITY QUESTIONS) + AUTOMATED CALL ──────────────
    #
    #   Tier 2 entry -> question A (wrong: +20 b_raw, n=1)
    #                -> question B, different from A (wrong: +35 b_raw,
    #                   n=2, _call=True, _call_pending=True)
    #   Automated call (mocked, no telephony API) -> question C,
    #   different from A and B:
    #       pass -> _probation=True, _tx_limit capped
    #       fail -> _frozen=True, permanent (bank_review)
    #
    # Correct answers add ZERO points — deliberately. The only downward
    # force on b_raw anywhere in this system is passive decay (0.97/
    # event); a signal that could push the score both up AND down based
    # on live user input is exactly the kind of thing that caused the
    # earlier oscillation bugs. KBA follows the same one-directional
    # rule as everything else.

    def next_kba_question(self) -> Optional[dict]:
        """Return a security question not yet asked this hardening
        episode, or None if the pool is exhausted (should not happen
        with MIN_KBA_QUESTIONS=5 and at most 3 asked per episode — a
        caller seeing None should fail safe to bank_review rather than
        repeat a question)."""
        pool = self.profile.get("security_questions", [])
        unused = [q for q in pool if q["id"] not in self._asked_kba_ids]
        if not unused:
            return None
        q = random.choice(unused)
        self._asked_kba_ids.append(q["id"])
        return {"question_id": q["id"], "question_text": q["text"]}

    def verify_kba_answer(self, question_id: str, answer: str, via_call: bool = False) -> dict:
        """Verify an answer to a previously-issued question. Set
        via_call=True only for the automated call's question C."""
        pool    = {q["id"]: q for q in self.profile.get("security_questions", [])}
        q       = pool.get(question_id)
        correct = bool(q) and _check_kba_answer(answer, q.get("answer_hash", ""))

        if via_call:
            self._call_pending = False
            if correct:
                self._probation = True
                self._probation_reason = "call_verification_passed"
                cap = self.profile.get("new_device_tx_limit")
                # legit_avg_transfer_amount anchors to the real account
                # holder's spending pattern — NOT self.profile's own
                # avg_transfer_amount, which is deliberately distorted
                # on some demo personas (e.g. "attacker": ₹450k) to
                # trigger unrelated transaction-anomaly checks. Falls
                # back to avg_transfer_amount only if a profile doesn't
                # define the dedicated field.
                avg = self.profile.get("legit_avg_transfer_amount",
                                        self.profile.get("avg_transfer_amount", 8000))
                fraction_cap = avg * PROBATION_CAP_FRACTION if avg > 0 else None
                candidates = [c for c in (cap, fraction_cap) if c is not None]
                self._probation_tx_limit = min(candidates) if candidates else None
                self._tx_limit = self._probation_tx_limit
                self._add_b(0, "Automated call verification passed — access restored, transfer amount capped and monitored", layer="kba")
            else:
                self._add_b(KBA_CALL_FAIL_PTS,
                            "Automated call verification failed — identity not confirmed", layer="kba")
                self._frozen        = True
                self._otp           = True
                self._call          = True
                self._harden        = True
                self._shown_tier    = 3
                self._tier_times[3] = time.time()
                self.max_shown_tier = max(self.max_shown_tier, 3)
            self._gate_locked = False  # call outcome IS the response; re-evaluate fresh
            return self.state()

        # Ordinary Tier-2 challenge (question A or B)
        if correct:
            self._kba_verified = True
            self._add_b(0, "Security question answered correctly", layer="kba")
        else:
            self.kba_failed_count += 1
            n   = self.kba_failed_count
            pts = KBA_FAIL_PTS[0] if n == 1 else KBA_FAIL_PTS[1]
            self._add_b(pts, f"Security question answered incorrectly (attempt {n})", layer="kba")
            self._harden = True
            self._otp    = True
            if n >= 2:
                self._call         = True
                self._call_pending = True
                # An incidental freeze from ORDINARY ambient scoring
                # (unrelated to KBA) can land in the window between
                # question A and question B — continuous behavioural
                # telemetry doesn't pause just because a KBA episode is
                # in progress. The call_pending guard in state() stops
                # any NEW freeze while pending, but can't retroactively
                # undo one that already landed. From the moment the
                # call is triggered, it becomes the sole authority on
                # freeze vs. probation for this episode — so any prior
                # incidental freeze is explicitly superseded here. This
                # is a one-time, one-directional transition (not a
                # recurring un-freeze), so it doesn't reopen the
                # oscillation risk the dwell/streak system already
                # closed elsewhere.
                self._frozen = False
        self._gate_locked = False  # this answer IS the response; re-evaluate fresh
        return self.state()

    def grant_phishing_referrer_probation(self, risk_label: str, matched_pattern: str) -> dict:
        """Called immediately on connect (see main.py's referrer_check
        handling) when this session's entry link/referrer matches a
        known phishing pattern -- NOT tied to any KBA/call flow at
        all. Reuses the EXACT same probation mechanism already granted
        after a passed automated-call verification: same tx-cap
        calculation, same cross-session persistence via main.py's
        ACCOUNT_STATE, same clean-session-streak clearing. Only
        `_probation_reason` differs, so the UI can show an honest,
        specific explanation instead of implying a call/KBA happened
        when it didn't.

        Deliberately "harden, don't lock" -- the person completing
        this login is very often the legitimate account holder who
        was phished, not an attacker. Their credentials may already be
        compromised from thirty seconds ago on a page this system has
        zero visibility into, but blocking a legitimate login on a
        pattern match alone would be actively harmful. Capping
        transactions and requiring extra verification for transfers is
        the proportionate response -- contain the potential blast
        radius without punishing someone for being a phishing target
        through no fault of their own."""
        self.phishing_referrer_flag_count += 1
        repeat = self.phishing_referrer_flag_count >= 2

        self._probation = True
        self._probation_reason = "phishing_referrer_detected"
        # Same cap logic as the call-verification-passed path (see
        # verify_kba_answer above) -- anchored to the account's real
        # spending baseline, not a persona's deliberately-inflated
        # avg_transfer_amount. A second occurrence within a short
        # window (tracked cross-session by main.py) is treated as an
        # active targeted campaign, not a one-off -- halve the cap
        # rather than just repeating the same one.
        cap = self.profile.get("new_device_tx_limit")
        avg = self.profile.get("legit_avg_transfer_amount",
                                self.profile.get("avg_transfer_amount", 8000))
        fraction_cap = avg * PROBATION_CAP_FRACTION if avg > 0 else None
        candidates = [c for c in (cap, fraction_cap) if c is not None]
        base_cap = min(candidates) if candidates else None
        self._probation_tx_limit = (base_cap / 2) if (repeat and base_cap) else base_cap
        self._tx_limit = self._probation_tx_limit

        label = (f"Entry link matched a known phishing pattern ({risk_label}"
                 f"{', repeat occurrence' if repeat else ''}) — {matched_pattern}")
        self._add_b(0, label, layer="threat_shield")
        # Force at least Tier 1 (security phrase) even if the
        # behavioural score alone wouldn't otherwise reach it yet --
        # "harden immediately" per the design brief, not "wait for the
        # score to catch up".
        self._harden = True
        return self.state()

    # ── FULL STATE ────────────────────────────────────────────

    def state(self) -> dict:
        shown = self._advance_tier()
        cs    = round(self.combined_score, 1)
        raw   = self._raw_tier()
        fb    = self._fallback(cs)

        # ── Fix #3: tier/fallback consistency ──────────────────────
        # _fallback() is intentionally driven by raw combined_score (cs),
        # NOT the dwell-gated shown tier — funds-blocking decisions
        # (step_up_otp, bank_review) cannot wait out an 8-second dwell
        # timer while a transaction is in flight.
        #
        # However, this previously meant the UI could show
        # tier_label="ELEVATED" / tier_action="user notices nothing"
        # (from `shown`) at the SAME time as fallback.action="bank_review"
        # (from raw `cs`) — a contradictory message.
        #
        # Resolution (option b from the spec): the DISPLAYED tier badge
        # is escalated immediately to match whatever the fallback message
        # implies, via `display_tier = max(shown, fallback_implied_tier)`.
        # `shown` (the dwell-gated _shown_tier) still governs the
        # progressive-hardening FLAGS (_harden/_otp/_frozen state
        # transitions, via _on_tier) and the dwell countdown — but the
        # tier the user is SHOWN, and the tier_label/tier_action/
        # system_response derived from it, always match the fallback.
        #
        # This preserves "user always sees an explanatory step before a
        # freeze": the fallback.message IS that explanatory step, and it
        # now arrives in the SAME response as the matching tier badge —
        # never a tier badge that claims "nothing happening" while a
        # bank_review fallback fires underneath it.
        fallback_tier = {
            "none":          0,
            "silent_reauth": 1,
            "step_up_otp":   2,
            "bank_review":   3,
        }.get(fb["action"], 0)

        # ── Fix: raw_tier floor ──────────────────────────────────────
        # display_tier must never under-report raw_tier. Both raw_tier
        # and fallback_tier are derived from the SAME combined_score
        # (cs), but via different threshold tables:
        #   _raw_tier:  [0,46)->0  [46,66)->1  [66,86)->2        [86,100]->3
        #   _fallback:  [0,66)->none(0)  [66,86)->silent_reauth(1)
        #               or step_up_otp(2)  [86,100]->bank_review(3)
        #
        # The [66,86) band maps to raw_tier=2 (HIGH RISK) but
        # fallback_tier=1 or 2 depending on pending_tx.amount — so
        # fallback_tier alone can under-report a HIGH RISK combined_score
        # as ELEVATED (the reproduced bug: cs=84.2, raw=2, fallback=1,
        # old display=max(shown=1,fallback=1)=1).
        #
        # Including raw in the max() fixes this. Note raw=3 can never
        # occur with fallback_tier<3 (both use the same cs and agree at
        # the 86 boundary, see KNOWN_ANOMALIES.md — confirmed by direct
        # trace), so this floor cannot silently introduce a "CRITICAL
        # without freeze" case beyond what fallback_tier==3 already
        # produces — but the freeze condition below is now keyed on
        # display_tier==3 (post-floor) rather than fallback_tier==3
        # alone, to stay correct even if that invariant ever changes.
        display_tier = max(shown, fallback_tier, raw)

        # ── Probation cap ────────────────────────────────────────────
        # A passed call verification deliberately does NOT reduce b_raw
        # (see verify_kba_answer's "never subtract on success" rule —
        # same principle as OTP success never reducing score elsewhere
        # in this file). That means raw combined_score can legitimately
        # sit at/above 86 even AFTER the user has proven their identity
        # via the call. Left alone, display_tier would still read 3 and
        # the freeze block below would re-trigger bank_review on the
        # very next event — contradicting the explicit "capped +
        # monitored, not frozen" outcome verify_kba_answer just granted.
        # Cap the BADGE at HIGH RISK (2) while probation is active; the
        # underlying score/signal log still shows the true numbers for
        # audit purposes, only the displayed tier and freeze gate are
        # capped.
        if self._probation and display_tier > 2:
            display_tier = 2

        # If the displayed tier is CRITICAL, the SESSION must actually
        # freeze immediately — not just the badge. A CRITICAL badge with
        # the session still "open" would be a state/badge inconsistency.
        # _frozen=True is permanent until reset() (manual review), same
        # guarantee as reaching Tier 3 via normal dwell-gated escalation.
        #
        # Checked on display_tier (post-floor) rather than fallback_tier
        # alone: with the current threshold tables this is equivalent
        # (raw==3 implies fallback_tier==3, see above), but this form
        # remains correct if that ever changes.
        #
        # EXCEPTION — self._call_pending, and more generally any active,
        # unresolved KBA episode: a wrong answer alone can legitimately
        # push combined_score past 86, which would otherwise auto-freeze
        # the session here BEFORE the user has had their intended chances
        # — defeating the "question A -> question B -> automated call"
        # design.
        #
        # Originally this exception only covered _call_pending (i.e.
        # AFTER the 2nd wrong answer). That left a real gap: with
        # realistic organic entry into HIGH RISK (combined ~78, verified
        # via live testing) plus KBA_FAIL_PTS[0]=20 on the FIRST wrong
        # answer, combined_score reliably crossed 86 immediately —
        # freezing the account before question B was ever offered, i.e.
        # before the episode's own 2nd-chance mechanism could kick in
        # at all. `len(self._asked_kba_ids) > 0 and not self._kba_verified`
        # is true from the moment question A is served until either a
        # correct answer (_kba_verified=True) or the automated call
        # resolves (_frozen or _probation gets set explicitly by
        # verify_kba_answer) — covering the whole episode, not just its
        # tail. While a call result is pending, or any KBA question is
        # outstanding, the explicit outcome from verify_kba_answer() is
        # the sole authority on freezing, not an incidental score
        # crossing. The badge still shows CRITICAL (display_tier is
        # unaffected) — only the hard freeze-the-session action is held
        # off pending that explicit outcome. This cannot be exploited to
        # avoid freezing indefinitely: the frontend gate blocks login
        # until the active question is answered, and every path through
        # the episode terminates in an explicit frozen/probation/
        # verified outcome.
        # Mirrors the tier-3 block below: display_tier already reflects
        # raw_tier immediately via max(shown, fallback_tier, raw) — it
        # has no dwell. So the badge/KBA gate can read HIGH RISK (and
        # the frontend's merged KBA+OTP box, with its "an OTP has also
        # been sent" note, shows immediately) well before shown_tier's
        # dwell-gated walk reaches 2 and _on_tier(2) would otherwise be
        # the only thing setting _otp=True. Without this, otp_triggered
        # could read False at the exact moment the UI is already
        # telling the user an OTP was sent — force it true in lockstep
        # with what's actually on screen, same as the tier-3 case
        # already does for call_triggered/progressive_harden.
        if display_tier >= 2 and not self._frozen:
            self._otp    = True
            self._harden = True

        if display_tier == 3 and not self._frozen and not self._call_pending and not self._kba_episode_active:
            self._frozen = True
            self._otp    = True
            self._call   = True
            self._harden = True
            self._shown_tier = 3
            self._tier_times[3] = time.time()
            shown = 3

        # Lock the score the instant any hardening gate is reached, so
        # it holds steady while that gate is unresolved (see
        # process_event's guard and the _gate_locked field docstring
        # for why). Deliberately does NOT auto-clear just because
        # display_tier later reads back below 1 while locked — that
        # would reopen exactly the flicker this exists to prevent.
        # Only an explicit response (handled elsewhere) clears it.
        if display_tier >= 1 and not self._frozen and not self._probation:
            self._gate_locked = True

        td = TIERS[display_tier]

        return {
            "behavioural_score": round(self.b_score, 1),
            "transaction_score": round(self.t_score, 1),
            "sigmoid_score":     round(self.sigmoid_score, 1),
            "combined_score":    cs,
            "b_raw":             round(self.b_raw, 1),
            "t_raw":             round(self.t_raw, 1),

            "tier":              display_tier,
            "shown_tier":        shown,          # dwell-gated internal tier (drives _harden/_otp/_frozen)
            "raw_tier":          raw,
            "tier_label":        td["label"],
            "tier_color":        td["color"],
            "tier_band":         td["band"],
            "tier_action":       td["action"],
            "system_response":   td["response"],
            "tier_escalated_by_fallback": display_tier > shown,

            "progressive_harden": self._harden or self._force_harden,
            "otp_triggered":      self._otp,
            "call_triggered":     self._call,
            "failed_login_count":  self.failed_login_count,
            "frozen":             self._frozen,
            "time_to_next_tier":  self._time_to_next(),
            "deescalation_streak": self._deescalation_streak,

            "kba_verified":       self._kba_verified,
            "kba_failed_count":   self.kba_failed_count,
            "call_pending":       self._call_pending,   # awaiting the automated call's question C
            "probation":          self._probation,       # sticky — see main.py ACCOUNT_STATE
            "probation_reason":   self._probation_reason,  # "call_verification_passed" | "phishing_referrer_detected" | None
            "phishing_referrer_flag_count": self.phishing_referrer_flag_count,
            "probation_tx_limit": self._probation_tx_limit,
            # Authoritative "hold the score still" signal, for any
            # frontend surface that runs its own local timer/animation
            # loop rather than purely reacting to each message (e.g.
            # dashboard.html's chart-sampling interval). Exists so those
            # surfaces consume this flag directly instead of trying to
            # re-derive "is a gate currently locked" from tier/otp/call
            # fields themselves — a second, hand-rolled implementation
            # of this condition is exactly what silently drifted out of
            # sync with scorer.py's real rules and caused a visible
            # sawtooth (local decay fighting the backend's correctly
            # frozen value every time a real update arrived).
            "score_locked":       self._kba_episode_active,

            "ai_enabled":    _rf_enabled,
            "ai_active":     self._ai_active,
            "ai_fraud_prob": round(self._ai_prob, 3),
            "ai_fraud_pct":  round(self._ai_prob * 100.0, 1),
            "ai_features":   self.features.to_dict() if self._ai_active else {},

            "tx_ml_enabled": _tx_fraud_enabled,
            "tx_ml_result":  self._tx_ml_result,

            "device_state":  self.device_state,
            "tx_limit":      self._tx_limit,
            "fallback":      fb,
            "signals":       self.signal_log,
            "pending_tx":    self.pending_tx,
        }

    def _fallback(self, cs: float) -> dict:
        pt = self.pending_tx

        # ── Progressive transaction response (score-independent) ───────
        # Score (cs) still decides WHETHER a transaction-driven response
        # fires at all (cs >= 66 threshold below, unchanged). Once it does,
        # the RESPONSE SEVERITY is driven by pt["force_immediate_kyc"] and
        # pt["is_retry"] — both computed in _eval_transaction() from
        # self.flagged_attempts, NOT from cs. This is deliberate: score is
        # a blended, decaying, multi-signal number and was the wrong
        # source of truth for "is this attempt #1 or a retry to this
        # specific beneficiary" — see KNOWN_ANOMALIES for the prior,
        # broken frontend-only (`otpAlreadyTriggered`) implementation this
        # replaces.
        if pt and pt.get("force_immediate_kyc"):
            return {"triggered": True, "action": "bank_review",
                    "message": "Transaction amount requires mandatory KYC verification before processing. Funds have not moved. Please visit your branch with valid KYC documents."}

        if cs >= 86:
            return {"triggered": True, "action": "bank_review",
                    "message": "The bank has flagged this session for mandatory compliance review. Transaction cancelled. Account secured. Funds have not moved."}
        if cs >= 66:
            if pt and pt.get("is_retry"):
                return {"triggered": True, "action": "bank_review",
                        "message": "A previous transaction to this beneficiary already required verification. Repeated high-risk attempts require mandatory KYC. Transaction cancelled. Funds have not moved."}
            amt = pt.get("amount", 0) if pt else 0
            avg_amt = self.profile.get("avg_transfer_amount", 8000)
            # ── Automated verification call (new rung, between OTP and
            # freeze) ─────────────────────────────────────────────────
            # Sits in the UPPER portion of the HIGH RISK band (78-85),
            # i.e. evidence strong enough that a one-time OTP code alone
            # is judged insufficient, but not yet at the 86 freeze
            # threshold. Mirrors a real PSB practice: an automated
            # outbound call to the registered mobile asking the customer
            # to confirm via keypad PIN entry, rather than relying on a
            # code the customer just types back into the same
            # potentially-compromised browser session. is_retry is
            # checked above this and always wins — a SECOND flagged
            # attempt to the same beneficiary skips straight to
            # bank_review rather than getting a second call, since a
            # retry after one verification step is itself suspicious.
            if cs >= 78:
                return {"triggered": True, "action": "automated_call",
                        "message": "Bank has initiated an automated verification call to your registered mobile number. Please answer and enter your secure PIN on the keypad to confirm this transaction. Transaction held pending call confirmation."}
            if amt > avg_amt * 5:
                return {"triggered": True, "action": "step_up_otp",
                        "message": "Bank needs to verify this transaction. OTP step-up and KYC check initiated. Transaction held for up to 10 minutes."}
            return {"triggered": True, "action": "silent_reauth",
                    "message": "Bank is verifying your session details. Please re-authenticate to continue."}
        return {"triggered": False, "action": "none", "message": ""}

    def reset(self):
        self.b_raw        = 0.0
        self.t_raw        = 0.0
        self.signal_log   = []      # three separate list objects
        self.b_signal_log = []
        self.t_signal_log = []
        self.pending_tx   = None
        self.device_state = None
        self._tx_limit    = None
        self._tx_ml_result = None
        self._last_amount_ratio = 0.0
        self._idle_streak       = 0
        self._jitter_flag_count = 0
        self._move_jitter_streak = 0
        self.features     = SessionFeatures(self.profile)
        self._ai_prob     = 0.0
        self._ai_active   = False
        self._shown_tier  = 0
        self._tier_times  = {}
        self._deescalation_streak = 0
        self._harden      = False
        self._otp         = False
        self._call         = False
        self._frozen      = False
        self._force_harden= False
        self.flagged_attempts = {}
        self.failed_login_count = 0
        # ^ Deliberately cleared here, and ONLY here. reset() represents
        # backend-driven manual review / session reset — not a UI
        # "New Transfer" button click, which must NOT clear this (a UI
        # action clearing fraud-attempt history would let an attacker
        # erase their own retry record simply by clicking a button).

        self.kba_failed_count = 0
        self._kba_verified    = False
        self._call_pending    = False
        self._asked_kba_ids   = []
        self._probation          = False
        self._probation_tx_limit = None
        self._probation_reason   = None
        self.phishing_referrer_flag_count = 0
        self.max_shown_tier      = 0
        self._gate_locked        = False
