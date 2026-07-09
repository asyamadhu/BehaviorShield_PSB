# =============================================================
# profiles.py — BehaviorShield
#
# Hardcoded demo profiles for the hackathon prototype.
# In production these are learned per user over 30+ sessions
# via EWMA blending (Section 3.5 Continuous Adaptation).
#
# Each profile contains:
#   Behavioural baseline  — how this user types and moves
#   Transaction baseline  — their normal banking patterns
#   Device trust list     — devices they have used before
# =============================================================

# ── KBA (Knowledge-Based Authentication) QUESTION POOL ───────
# Belongs to the ACCOUNT (SBI-XXXX-4821), not any one session profile —
# "arjun", "arjun_new_device" and "attacker" all target the same
# account, so all three share this pool. A real attacker session would
# be challenged with these same questions and (by definition) not know
# the answers; the "attacker" profile carries the pool for that reason,
# not because the attacker registered it.
#
# answer_hash format: "salt:sha256(salt + normalized_answer)".
# Never store plaintext answers. Regenerate hashes with
# scorer.hash_kba_answer(raw_answer) if the demo answers ever change.
# Normalization (scorer._normalize_kba_answer) lowercases, strips
# punctuation, and collapses whitespace, so "Mumbai", " mumbai ",
# "MUMBAI!" all match the same stored hash.
#
# 6 questions registered (min 5, max 7 per design): a hardening episode
# consumes at most 3 (question A, question B, the call's question C),
# leaving spares so the exact same 3 aren't asked every single time a
# session escalates.
SECURITY_QUESTIONS_SBI_4821 = [
    {"id": "sq1", "text": "What was your childhood nickname?",
     "answer_hash": "demosq1:f3c7562ab591acf2197908a2d29e0671d9bcd181824a8c9d3aaa531ae2b7bb95",
     "system_generated": False},
    {"id": "sq2", "text": "What is the name of your first school?",
     "answer_hash": "demosq2:183d66b3672389d24e768dd479c0bbfbdd75f7268db0333539cc92fd23711ce5",
     "system_generated": True},
    {"id": "sq3", "text": "What did you call your first pet?",
     "answer_hash": "demosq3:4d4aca4ed94647a5b8743fe676e0af9bea3144081a464b9ab852c57028ac3a50",
     "system_generated": False},
    {"id": "sq4", "text": "In what city were you born?",
     "answer_hash": "demosq4:dfadd2c3be264ead55e04edc5adfbdd667510c1976f38510b77975ffcec09179",
     "system_generated": True},
    {"id": "sq5", "text": "What is your favourite childhood dish?",
     "answer_hash": "demosq5:1de242d1a089a7da34bac2572d1d2927cb46a1e88c1a1d3b3920005c86f1b98f",
     "system_generated": False},
    {"id": "sq6", "text": "What was the model of your first bike?",
     "answer_hash": "demosq6:c9633a121088189930239eaa32bd416dddb54d5ba12576a5d3ae9592788e0c29",
     "system_generated": False},
]


PROFILES = {

    # ── LEGITIMATE USER ──────────────────────────────────────
    "arjun": {
        "name":    "Arjun Sharma",
        "account": "SBI-XXXX-4821",

        # Behavioural biometric baseline
        "avg_dwell_ms":  95,     # normal human key-hold ~95ms
        "avg_flight_ms": 140,    # normal gap between keys ~140ms
        "avg_wpm":       52,     # normal typing ~52 WPM

        # Transaction pattern baseline
        "avg_transfer_amount": 8000,
        "max_normal_transfer": 25000,
        "transfers_per_week":  3,
        "known_beneficiaries": [
            "SBI-XXXX1234",
            "HDFC-XXXX5678",
            "ICICI-XXXX9012",
        ],
        "usual_hour_start": 8,
        "usual_hour_end":   22,

        # Device trust list
        # fingerprint format: Browser-OS-ScreenResolution-Timezone
        # All common Ubuntu/Linux resolutions included so demo
        # machine is always recognised as trusted.
        "trusted_devices": [
            {"fingerprint": "Chrome-Win11-1920x1080-Asia/Kolkata",  "label": "Arjun's Laptop (Windows)", "trust_score": 1.0, "sessions": 47},
            {"fingerprint": "Chrome-Android-390x844-Asia/Kolkata",   "label": "Arjun's Phone",           "trust_score": 0.9, "sessions": 23},
            {"fingerprint": "Chrome-Linux-1920x1080-Asia/Kolkata",   "label": "Demo Machine (1080p)",    "trust_score": 1.0, "sessions": 10},
            {"fingerprint": "Chrome-Linux-1366x768-Asia/Kolkata",    "label": "Demo Machine (768p)",     "trust_score": 1.0, "sessions": 10},
            {"fingerprint": "Chrome-Linux-1536x864-Asia/Kolkata",    "label": "Demo Machine (864p)",     "trust_score": 1.0, "sessions": 10},
            {"fingerprint": "Chrome-Linux-2560x1440-Asia/Kolkata",   "label": "Demo Machine (1440p)",    "trust_score": 1.0, "sessions": 10},
            {"fingerprint": "Chrome-Linux-1440x900-Asia/Kolkata",    "label": "Demo Machine (900p)",     "trust_score": 1.0, "sessions": 10},
            {"fingerprint": "Firefox-Linux-1920x1080-Asia/Kolkata",  "label": "Demo Firefox (1080p)",    "trust_score": 1.0, "sessions": 5},
            {"fingerprint": "Firefox-Linux-1366x768-Asia/Kolkata",   "label": "Demo Firefox (768p)",     "trust_score": 1.0, "sessions": 5},
        ],
        "device_trust_threshold":   0.6,
        "new_device_tx_limit":      10000,   # ₹10,000 max on unknown device
        "new_device_session_limit": 5,

        "security_questions": SECURITY_QUESTIONS_SBI_4821,
        # Real account-holder baseline, used specifically for the
        # post-call-verification probation cap (scorer.verify_kba_answer).
        # Kept separate from avg_transfer_amount because that field is
        # deliberately distorted on the "attacker" profile below (₹450k)
        # to trigger unrelated transaction-anomaly checks — the
        # probation cap must always anchor to Arjun's real spending
        # pattern, not whichever demo persona triggered the challenge.
        "legit_avg_transfer_amount": 8000,
    },

    # ── ATTACKER (stolen credentials) ────────────────────────
    "attacker": {
        "name":    "Unknown Session",
        "account": "SBI-XXXX-4821",

        "avg_dwell_ms":  10,     # bot-fast
        "avg_flight_ms": 6,      # bot-fast
        "avg_wpm":       220,    # impossibly fast

        "avg_transfer_amount": 450000,
        "max_normal_transfer": 450000,
        "transfers_per_week":  12,
        "known_beneficiaries": [],
        "usual_hour_start":    0,
        "usual_hour_end":      4,

        "trusted_devices":          [],
        "device_trust_threshold":   0.6,
        "new_device_tx_limit":      None,    # None = no device-based limit
        "new_device_session_limit": 5,

        # Same account, same question pool — the attacker demo persona
        # is challenged with Arjun's real questions and is not expected
        # to know the answers.
        "security_questions": SECURITY_QUESTIONS_SBI_4821,
        "legit_avg_transfer_amount": 8000,   # see note on arjun's profile above
    },

    # ── ARJUN ON NEW DEVICE ───────────────────────────────────
    # Demonstrates: new device → probationary → behaviour still matches
    # → system monitors but does NOT block outright
    "arjun_new_device": {
        "name":    "Arjun Sharma",
        "account": "SBI-XXXX-4821",

        "avg_dwell_ms":  95,
        "avg_flight_ms": 140,
        "avg_wpm":       52,

        "avg_transfer_amount": 8000,
        "max_normal_transfer": 25000,
        "transfers_per_week":  3,
        "known_beneficiaries": [
            "SBI-XXXX1234",
            "HDFC-XXXX5678",
            "ICICI-XXXX9012",
        ],
        "usual_hour_start": 8,
        "usual_hour_end":   22,

        "trusted_devices":          [],   # no known devices — forces probationary
        "device_trust_threshold":   0.6,
        "new_device_tx_limit":      10000,
        "new_device_session_limit": 5,

        "security_questions": SECURITY_QUESTIONS_SBI_4821,
        "legit_avg_transfer_amount": 8000,   # see note on arjun's profile above
    },
}
