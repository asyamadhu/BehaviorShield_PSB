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
    },
}
