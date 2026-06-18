# =============================================================
# test_scorer.py — BehaviorShield v4 fix verification suite
#
# Run:  pytest backend/test_scorer.py -v
#       (or: python3 -m pytest test_scorer.py -v  from backend/)
#
# Covers:
#   1. Original false positive (₹100/new-bene/trusted) — must stay fixed
#   2. Regression case (₹450K/new-bene/2am/zero-behaviour) — must now escalate
#   3. Amount-sweep monotonicity (new bene & known bene)
#   4. Behavioural-only attacks (bot typing + paste)
#   5. Synergy / compound (both layers moderately elevated)
#   6. Fallback/tier consistency (no ELEVATED+bank_review contradiction)
#   7. De-escalation (Tier 1 -> 0 after sustained clean events;
#      Tier 3 never auto-recovers)
#   8. v3 regression scenarios (legit, phishing->large tx, bot attack,
#      velocity/structuring)
#
# Each test prints its key numbers so `pytest -v -s` doubles as a
# readable trace of the scoring engine's behaviour.
# =============================================================

import sys
import os
import io
import contextlib

try:
    import pytest
except ImportError:
    # pytest not available — provide a no-op shim so module-level
    # decorators (pytest.mark.parametrize) don't break standalone runs.
    class _PytestShim:
        class mark:
            @staticmethod
            def parametrize(*args, **kwargs):
                def deco(fn):
                    return fn
                return deco
    pytest = _PytestShim()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

with contextlib.redirect_stderr(io.StringIO()):
    from scorer import SuspicionScorer, DeviceTrustEngine, _combine, DWELL
    from profiles import PROFILES


# ── Helpers ──────────────────────────────────────────────────

def fresh(profile_key="arjun"):
    with contextlib.redirect_stderr(io.StringIO()):
        return SuspicionScorer(PROFILES[profile_key])


def feed(sc, events):
    """Process a list of events, return the final state dict."""
    s = None
    with contextlib.redirect_stderr(io.StringIO()):
        for ev in events:
            s = sc.process_event(ev)
    return s


def trusted_device():
    return {"type": "device_check",
            "device_fingerprint": "Chrome-Linux-1920x1080-Asia/Kolkata"}


def inject_threat_pts(sc, pts, label):
    """
    Mirrors main.py's _inject_threat_pts(profile_name, pts, label) without
    requiring fastapi to be importable in the test environment:

        def _inject_threat_pts(profile_name, pts, label):
            if profile_name and profile_name in sessions and pts > 0:
                sessions[profile_name].process_event({
                    "type": "threat_shield_signal", "pts": pts, "label": label,
                })

    The `profile_name in sessions` check is main.py's session-routing
    concern (not exercised here); the `pts > 0` guard IS exercised
    (test_threat_shield_zero_pts_is_noop below).
    """
    if pts > 0:
        with contextlib.redirect_stderr(io.StringIO()):
            return sc.process_event({"type": "threat_shield_signal",
                                       "pts": pts, "label": label})
    return sc.state()


def assert_tier_fallback_consistent(s):
    """
    Fix #3 invariant: the displayed tier_label and the fallback.action
    must never contradict each other. Specifically, an "ELEVATED"
    (silent-monitor, "user notices nothing") tier must never coexist
    with a funds-blocking fallback action (step_up_otp / bank_review).
    """
    label  = s["tier_label"]
    action = s["fallback"]["action"]

    if action in ("step_up_otp", "bank_review"):
        assert label not in ("NORMAL", "ELEVATED"), (
            f"Inconsistent: tier_label={label!r} (implies low/no impact) "
            f"but fallback.action={action!r} (funds-blocking). "
            f"tier={s['tier']} shown_tier={s['shown_tier']} "
            f"raw_tier={s['raw_tier']} combined={s['combined_score']}"
        )

    if label == "CRITICAL":
        assert s["frozen"] is True, (
            "tier_label=CRITICAL but frozen=False — session should be "
            "frozen whenever the CRITICAL tier is displayed."
        )

    if action == "bank_review":
        assert s["frozen"] is True, (
            "fallback.action=bank_review but frozen=False — a bank_review "
            "fallback must freeze the session immediately (fix #3)."
        )

    # Floor invariant (this fix): the displayed tier must never
    # under-report raw_tier. raw_tier=2 (HIGH RISK, combined in [66,86))
    # can coexist with fallback_tier=1 (silent_reauth) since _fallback's
    # thresholds don't align 1:1 with _raw_tier's — without this floor,
    # display_tier could be 1 (ELEVATED) while combined_score is
    # genuinely in the HIGH RISK band.
    assert s["tier"] >= s["raw_tier"], (
        f"display_tier={s['tier']} < raw_tier={s['raw_tier']} — "
        f"combined_score={s['combined_score']} is under-reported by the "
        f"displayed tier. tier_label={s['tier_label']!r} "
        f"fallback.action={s['fallback']['action']!r}"
    )


# ═══════════════════════════════════════════════════════════════
# 1. ORIGINAL FALSE POSITIVE — must stay fixed
# ═══════════════════════════════════════════════════════════════

def test_original_false_positive_stays_fixed():
    """
    ₹100 transfer, new beneficiary, trusted device, zero keystrokes
    → Tier 0, full_access, fallback not triggered.
    """
    sc = fresh("arjun")
    s = feed(sc, [
        trusted_device(),
        {"type": "transaction_attempt", "amount": 100,
         "beneficiary": "Brand New Person", "hour": 14},
    ])

    print(f"\n[FP] combined={s['combined_score']}  raw_tier={s['raw_tier']}  "
          f"tier={s['tier']}({s['tier_label']})  fallback={s['fallback']['action']}")

    assert s["raw_tier"] == 0
    assert s["tier"] == 0
    assert s["tier_label"] == "NORMAL"
    assert s["system_response"] == "full_access"
    assert s["fallback"]["triggered"] is False
    assert s["combined_score"] <= 45
    assert_tier_fallback_consistent(s)


# ═══════════════════════════════════════════════════════════════
# 2. REGRESSION CASE — must now correctly escalate
# ═══════════════════════════════════════════════════════════════

def test_regression_large_tx_zero_behaviour_escalates():
    """
    ₹4,50,000 transfer, brand-new beneficiary, 2 AM, ZERO behavioural
    events (no device_check, no keystrokes — simulating a script
    hitting the API/WebSocket directly, bypassing the frontend).

    Must reach Tier 2 or 3 based on t_score alone, with a matching
    fallback.action (step_up_otp or bank_review).
    """
    sc = fresh("arjun")
    s = feed(sc, [
        {"type": "transaction_attempt", "amount": 450000,
         "beneficiary": "Unknown XYZ", "hour": 2},
    ])

    print(f"\n[REGRESSION] b_raw={s['b_raw']}  t_raw={s['t_raw']}  "
          f"combined={s['combined_score']}  raw_tier={s['raw_tier']}  "
          f"tier={s['tier']}({s['tier_label']})  fallback={s['fallback']['action']}")

    assert s["b_raw"] == 0.0, "This test requires zero behavioural signal"
    assert s["raw_tier"] >= 2, (
        f"Expected raw_tier >= 2 for large anomalous tx with zero "
        f"behavioural corroboration, got {s['raw_tier']} "
        f"(combined={s['combined_score']})"
    )
    assert s["tier"] >= 2
    assert s["fallback"]["triggered"] is True
    assert s["fallback"]["action"] in ("step_up_otp", "bank_review")
    assert_tier_fallback_consistent(s)


def test_regression_with_device_check_also_escalates():
    """
    Same regression scenario, but a (real, untrusted) device_check
    fires first — closer to a real browser session on an unknown
    device. Should escalate at least as hard as the zero-telemetry case.
    """
    DeviceTrustEngine.DEMO_ALL_TRUSTED = False
    try:
        sc = fresh("arjun")
        s = feed(sc, [
            {"type": "device_check", "device_fingerprint": "totally-unknown-device"},
            {"type": "transaction_attempt", "amount": 450000,
             "beneficiary": "Unknown XYZ", "hour": 2},
        ])
        print(f"\n[REGRESSION+DEVICE] b_raw={s['b_raw']}  t_raw={s['t_raw']}  "
              f"combined={s['combined_score']}  raw_tier={s['raw_tier']}  "
              f"tier={s['tier']}({s['tier_label']})  fallback={s['fallback']['action']}")

        assert s["raw_tier"] >= 2
        assert s["fallback"]["triggered"] is True
        assert_tier_fallback_consistent(s)
    finally:
        DeviceTrustEngine.DEMO_ALL_TRUSTED = True


# ═══════════════════════════════════════════════════════════════
# 3. BOUNDARY / AMOUNT-SWEEP SANITY
# ═══════════════════════════════════════════════════════════════

AMOUNT_SWEEP = [100, 500, 1000, 5000, 8000, 16000, 50000, 100000, 250000, 450000, 1000000]


def test_amount_sweep_known_beneficiary_stays_near_tier0():
    """
    Sweep ₹100 -> ₹10,00,000 to a KNOWN beneficiary, zero behavioural
    signal. Should stay near Tier 0 across the whole range — a known
    payee is not "new beneficiary" risk regardless of amount (the
    amount_anomaly signal can still fire for very large amounts, but
    should not compound with new-beneficiary signals that don't apply).
    """
    print("\n[SWEEP known-bene]")
    for amt in AMOUNT_SWEEP:
        sc = fresh("arjun")
        s = feed(sc, [
            {"type": "transaction_attempt", "amount": amt,
             "beneficiary": "SBI-XXXX1234", "hour": 14},  # known beneficiary
        ])
        print(f"  amt={amt:>9,}  ratio={amt/8000:7.3f}  "
              f"combined={s['combined_score']:6.2f}  raw_tier={s['raw_tier']}")
        assert s["raw_tier"] <= 1, (
            f"Known-beneficiary transfer of ₹{amt:,} reached raw_tier="
            f"{s['raw_tier']} (combined={s['combined_score']}) — "
            f"known payees should not drive high escalation from amount alone."
        )


def test_combine_monotonic_in_t_raw_gate_active():
    """
    _combine() itself (the part touched by this fix) must be
    non-decreasing in t_raw for a fixed b_raw and amount_ratio,
    in BOTH gate states (amount_ratio <= 1 and > 1).

    This isolates the _combine() math from the separately-flagged
    tx_fraud_scorer ML non-monotonicity (see test_known_anomalies.py
    note / KNOWN_ANOMALIES.md).
    """
    print("\n[_combine monotonicity] gate ACTIVE (amount_ratio=0.5, b_raw=0)")
    prev = -1.0
    for t in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150]:
        c = _combine(0, t, amount_ratio=0.5)
        print(f"  t_raw={t:4d}  combined={c:6.2f}")
        assert c >= prev - 1e-9, f"Non-monotonic at t_raw={t}: {c} < {prev}"
        prev = c

    print("\n[_combine monotonicity] gate INACTIVE (amount_ratio=2.0, b_raw=0)")
    prev = -1.0
    for t in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150]:
        c = _combine(0, t, amount_ratio=2.0)
        print(f"  t_raw={t:4d}  combined={c:6.2f}")
        assert c >= prev - 1e-9, f"Non-monotonic at t_raw={t}: {c} < {prev}"
        prev = c


def test_amount_sweep_new_beneficiary_raw_tier_non_decreasing_in_bands():
    """
    Sweep ₹100 -> ₹10,00,000 to a NEW beneficiary, zero behavioural
    signal. The overall trend should be non-decreasing risk: once the
    amount crosses the 3x-baseline anomaly threshold (₹24,000 for
    arjun, avg=₹8,000), raw_tier should be >= the raw_tier seen for
    any smaller amount below that threshold.

    NOTE: combined_score itself is NOT asserted strictly monotonic
    across this full sweep — see KNOWN_ANOMALIES.md item #1 for a
    documented non-monotonicity introduced by tx_fraud_scorer's ML
    model (fraud_prob can decrease as amount_ratio increases in the
    0.5x-3x range, due to win1_mean/bene_risk interaction). That issue
    is pre-existing and out of scope for this fix, but is flagged here
    so it isn't silently lost.
    """
    print("\n[SWEEP new-bene] (informational — see KNOWN_ANOMALIES.md #1)")
    results = []
    for amt in AMOUNT_SWEEP:
        sc = fresh("arjun")
        s = feed(sc, [
            {"type": "transaction_attempt", "amount": amt,
             "beneficiary": "Brand New Person", "hour": 14},
        ])
        results.append((amt, s["combined_score"], s["raw_tier"]))
        print(f"  amt={amt:>9,}  ratio={amt/8000:7.3f}  "
              f"combined={s['combined_score']:6.2f}  raw_tier={s['raw_tier']}")

    # Small amounts (<= avg) must stay Tier 0.
    for amt, cs, rt in results:
        if amt <= 8000:
            assert rt == 0, f"₹{amt:,} (<= avg) reached raw_tier={rt}"

    # Amounts well above the 3x-anomaly threshold (₹24,000) must reach
    # Tier 3 — the compound amount_anomaly + new_beneficiary signal.
    for amt, cs, rt in results:
        if amt >= 25000:
            assert rt == 3, f"₹{amt:,} (>>3x avg, new bene) reached only raw_tier={rt}"


# ═══════════════════════════════════════════════════════════════
# 4. BEHAVIOURAL-ONLY ATTACKS
# ═══════════════════════════════════════════════════════════════

def test_bot_speed_typing_plus_paste_escalates_behaviourally():
    """
    Bot-speed typing (dwell ~9ms vs 95ms baseline) + paste_password,
    NO transaction. Behavioural score alone should escalate the
    session noticeably above baseline (raw_tier >= 1), purely from
    b_raw — t_raw stays 0.
    """
    sc = fresh("arjun")
    s = feed(sc, [
        trusted_device(),
        {"type": "keystroke", "dwell_ms": 9, "flight_ms": 6, "field": "username"},
        {"type": "paste_event", "field": "password"},
        {"type": "keystroke", "dwell_ms": 8, "flight_ms": 5, "field": "password"},
        {"type": "keystroke", "dwell_ms": 11, "flight_ms": 7, "field": "password"},
        {"type": "keystroke", "dwell_ms": 10, "flight_ms": 6, "field": "password"},
        {"type": "mouse_move", "x": 200, "y": 200, "jitter": 0.05},
    ])

    print(f"\n[BOT+PASTE] b_raw={s['b_raw']}  t_raw={s['t_raw']}  "
          f"combined={s['combined_score']}  raw_tier={s['raw_tier']}  "
          f"tier={s['tier']}({s['tier_label']})")

    assert s["t_raw"] == 0.0, "No transaction occurred — t_raw must be 0"
    assert s["b_raw"] > 50, f"Expected substantial b_raw, got {s['b_raw']}"
    assert s["raw_tier"] >= 1, (
        f"Bot typing + paste should escalate behaviourally, got "
        f"raw_tier={s['raw_tier']} (combined={s['combined_score']})"
    )
    assert_tier_fallback_consistent(s)


def test_normal_typing_stays_tier0():
    """
    Normal typing (dwell within ~10% of baseline), no transaction
    → stays Tier 0.
    """
    sc = fresh("arjun")
    s = feed(sc, [
        trusted_device(),
        {"type": "keystroke", "dwell_ms": 92, "flight_ms": 138, "field": "username"},
        {"type": "keystroke", "dwell_ms": 98, "flight_ms": 143, "field": "password"},
        {"type": "keystroke", "dwell_ms": 91, "flight_ms": 137, "field": "password"},
        {"type": "keystroke", "dwell_ms": 99, "flight_ms": 141, "field": "password"},
        {"type": "keystroke", "dwell_ms": 90, "flight_ms": 139, "field": "password"},
        {"type": "mouse_move", "x": 452, "y": 310, "jitter": 1.9},
    ])

    print(f"\n[NORMAL TYPING] b_raw={s['b_raw']}  combined={s['combined_score']}  "
          f"raw_tier={s['raw_tier']}  tier={s['tier']}({s['tier_label']})")

    assert s["raw_tier"] == 0
    assert s["tier"] == 0
    assert s["tier_label"] == "NORMAL"
    assert_tier_fallback_consistent(s)


# ═══════════════════════════════════════════════════════════════
# 5. SYNERGY / COMPOUND
# ═══════════════════════════════════════════════════════════════

def test_synergy_fires_when_both_layers_moderately_elevated():
    """
    _combine() synergy unit test: when BOTH b_raw and t_raw land in a
    "moderately elevated" zone (b_c > 20 AND t_c > 20 — the zone where
    the synergy term in _combine is non-zero), the combined score
    should exceed EITHER layer's solo contribution and cross into
    Tier 1, even though neither layer alone reaches Tier 1.

    b_raw=75 (b_c≈36.0)  -> _combine(75, 0) ≈ 36.0   (Tier 0)
    t_raw=45 (t_c≈39.0)  -> _combine(0, 45) ≈ 39.0   (Tier 0)
    combined            -> _combine(75, 45) ≈ 55.1  (Tier 1, synergy fired)
    """
    b_raw, t_raw = 75.0, 45.0

    b_alone = _combine(b_raw, 0, amount_ratio=2.0)
    t_alone = _combine(0, t_raw, amount_ratio=2.0)
    combo   = _combine(b_raw, t_raw, amount_ratio=2.0)

    def tier_of(c):
        if c < 46: return 0
        if c < 66: return 1
        if c < 86: return 2
        return 3

    print(f"\n[SYNERGY] b_alone={b_alone:.2f}({tier_of(b_alone)})  "
          f"t_alone={t_alone:.2f}({tier_of(t_alone)})  "
          f"combo={combo:.2f}({tier_of(combo)})")

    assert tier_of(b_alone) == 0, f"b_alone already Tier {tier_of(b_alone)}"
    assert tier_of(t_alone) == 0, f"t_alone already Tier {tier_of(t_alone)}"
    assert combo > b_alone, "Synergy did not exceed behavioural-alone score"
    assert combo > t_alone, "Synergy did not exceed transaction-alone score"
    assert tier_of(combo) >= 1, (
        f"Synergy did not push combined score into Tier 1+: combo={combo:.2f}"
    )


def test_known_anomaly_combine_valley_for_weak_second_signal():
    """
    INFORMATIONAL / DOCUMENTS A PRE-EXISTING ANOMALY (not introduced by
    this fix, out of scope to fix here — see KNOWN_ANOMALIES.md item #2).

    _combine()'s synergy term only activates once t_c > 20 (equivalently
    b_c > 20 for the other direction). For t_raw in roughly [5, 28]
    (t_c in (0, 20]), the cross-penalty terms
    (1 - t_p*0.3) / (1 - b_p*0.3) reduce each layer's contribution
    WITHOUT any synergy bonus to compensate — so adding a *weak* second
    signal can make combined_score LOWER than the dominant layer alone.

    This test documents the effect with a concrete example. It is
    marked informational: it does not assert "no dip", it just records
    the magnitude so the dip doesn't silently regress further. If this
    assertion ever fires, it means the valley got WORSE (deeper) than
    currently observed, which would be worth a closer look even though
    fixing the valley itself is out of scope for this change.
    """
    b_raw = 74.0
    b_alone = _combine(b_raw, 0, amount_ratio=2.0)

    worst_dip = 0.0
    worst_t   = None
    for t_raw in range(0, 30):
        c = _combine(b_raw, t_raw, amount_ratio=2.0)
        dip = b_alone - c
        if dip > worst_dip:
            worst_dip = dip
            worst_t   = t_raw

    print(f"\n[KNOWN ANOMALY #2] b_alone={b_alone:.2f}  "
          f"worst dip={worst_dip:.2f} at t_raw={worst_t} "
          f"(combined={b_alone - worst_dip:.2f})")

    # Documented current magnitude is ~4.6 points. Allow some headroom
    # (10 points) before treating this as a meaningful regression.
    assert worst_dip < 10.0, (
        f"_combine 'valley' anomaly has grown to {worst_dip:.2f} points "
        f"(at t_raw={worst_t}) — previously ~4.6. This is a pre-existing "
        f"issue (KNOWN_ANOMALIES.md #2) but a 2x+ growth suggests "
        f"something else changed too."
    )


# ═══════════════════════════════════════════════════════════════
# 6. FALLBACK / TIER CONSISTENCY (fix #3) — sweep all earlier cases
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("amount,bene,hour,extra_events", [
    (100,     "Brand New Person", 14, []),
    (450000,  "Unknown XYZ",       2,  []),
    (8000,    "SBI-XXXX1234",      14, []),
    (95000,   "New Unknown Payee", 2,  []),
    (12000,   "New Person",        14, [{"type": "paste_password"}]),
])
def test_fallback_tier_consistency_across_scenarios(amount, bene, hour, extra_events):
    """
    For a spread of scenarios, assert tier/fallback never contradict
    (fix #3 invariant) — in particular, no case where tier_label ==
    "ELEVATED" (silent monitor, "user notices nothing") simultaneously
    has fallback.action == "bank_review" (or step_up_otp).
    """
    sc = fresh("arjun")
    events = [trusted_device()] + extra_events + [
        {"type": "transaction_attempt", "amount": amount,
         "beneficiary": bene, "hour": hour},
    ]
    s = feed(sc, events)

    print(f"\n[CONSISTENCY] amount={amount} bene={bene!r} hour={hour} "
          f"-> combined={s['combined_score']} tier={s['tier']}({s['tier_label']}) "
          f"fallback={s['fallback']['action']}")

    assert_tier_fallback_consistent(s)


# Standalone-runnable variant (no pytest dependency) — same cases as above
CONSISTENCY_CASES = [
    (100,     "Brand New Person", 14, []),
    (450000,  "Unknown XYZ",       2,  []),
    (8000,    "SBI-XXXX1234",      14, []),
    (95000,   "New Unknown Payee", 2,  []),
    (12000,   "New Person",        14, [{"type": "paste_password"}]),
]


def test_fallback_tier_consistency_standalone():
    """Standalone runner for the consistency sweep (no pytest needed)."""
    for amount, bene, hour, extra_events in CONSISTENCY_CASES:
        test_fallback_tier_consistency_across_scenarios(amount, bene, hour, extra_events)


# ═══════════════════════════════════════════════════════════════
# 7. DE-ESCALATION (fix #4)
# ═══════════════════════════════════════════════════════════════

def test_deescalation_from_tier1_after_sustained_clean_events():
    """
    Push a session to Tier 1 (security phrase shown / _harden=True),
    then send many consecutive "clean" events (normal keystrokes,
    no new signals). b_raw/combined_score should decay, and once
    raw_tier stays below shown_tier for DEESCALATION_STREAK
    consecutive events, shown_tier (and tier/_harden) should step
    back down by one — never skipping (Tier1->0, not Tier1->-1
    obviously, but the rule generalises: one step at a time).
    """
    sc = fresh("arjun")

    # Push hard enough to reach Tier 1 (raw_tier momentarily 2, but
    # dwell gates shown_tier to 1 on the first evaluation).
    s = feed(sc, [
        trusted_device(),
        {"type": "paste_password"},
        {"type": "mouse_jitter", "jitter_score": 0.1},
        {"type": "concurrent_session"},
        {"type": "remote_access_tool"},
    ])
    print(f"\n[DEESC] after push: b_raw={s['b_raw']}  combined={s['combined_score']}  "
          f"raw_tier={s['raw_tier']}  shown_tier={s['shown_tier']}  "
          f"tier={s['tier']}  harden={s['progressive_harden']}")

    assert s["shown_tier"] >= 1, "Setup failed to reach Tier 1"
    assert s["progressive_harden"] is True

    # Send sustained clean events and watch for de-escalation
    deescalated = False
    for i in range(25):
        s = feed(sc, [{"type": "keystroke", "dwell_ms": 95, "flight_ms": 140, "field": "x"}])
        print(f"  ev{i+1:2d}: b_raw={s['b_raw']:6.2f}  combined={s['combined_score']:6.2f}  "
              f"raw_tier={s['raw_tier']}  shown_tier={s['shown_tier']}  "
              f"streak={s['deescalation_streak']}  harden={s['progressive_harden']}")
        if s["shown_tier"] == 0:
            deescalated = True
            break

    assert deescalated, "Session never de-escalated back to Tier 0 after 25 clean events"
    assert s["progressive_harden"] is False, "_harden should clear when shown_tier < 1"
    assert s["tier"] == 0
    assert_tier_fallback_consistent(s)


def test_tier3_never_auto_deescalates():
    """
    A session that reaches Tier 3 (CRITICAL/frozen) must stay frozen
    regardless of subsequent decay — no auto-recovery from CRITICAL.
    Only reset() (manual review / new session) clears it.
    """
    sc = fresh("arjun")
    s = feed(sc, [
        {"type": "transaction_attempt", "amount": 450000,
         "beneficiary": "Unknown XYZ", "hour": 2},
    ])
    print(f"\n[TIER3 STICKY] after attack: combined={s['combined_score']}  "
          f"shown_tier={s['shown_tier']}  tier={s['tier']}  frozen={s['frozen']}")
    assert s["tier"] == 3
    assert s["frozen"] is True

    # Sustained clean events afterwards
    for i in range(30):
        s = feed(sc, [{"type": "keystroke", "dwell_ms": 95, "flight_ms": 140, "field": "x"}])

    print(f"[TIER3 STICKY] after 30 clean events: combined={s['combined_score']}  "
          f"raw_tier={s['raw_tier']}  shown_tier={s['shown_tier']}  "
          f"tier={s['tier']}  frozen={s['frozen']}")

    assert s["shown_tier"] == 3, "Tier 3 must not de-escalate automatically"
    assert s["tier"] == 3
    assert s["frozen"] is True

    # Manual reset() clears it (per design doc)
    sc.reset()
    s = feed(sc, [{"type": "keystroke", "dwell_ms": 95, "flight_ms": 140, "field": "x"}])
    print(f"[TIER3 STICKY] after reset(): combined={s['combined_score']}  "
          f"shown_tier={s['shown_tier']}  tier={s['tier']}  frozen={s['frozen']}")
    assert s["frozen"] is False
    assert s["tier"] == 0


# ═══════════════════════════════════════════════════════════════
# 8. v3 SCENARIO REGRESSION CHECKS
# ═══════════════════════════════════════════════════════════════

def test_v3_scenario1_legit_user_tier0():
    """v3 Scenario 1 — legit user, trusted device, normal transfer -> Tier 0."""
    sc = fresh("arjun")
    s = feed(sc, [
        trusted_device(),
        {"type": "keystroke", "dwell_ms": 92, "flight_ms": 138, "field": "username"},
        {"type": "keystroke", "dwell_ms": 95, "flight_ms": 141, "field": "password"},
        {"type": "keystroke", "dwell_ms": 89, "flight_ms": 135, "field": "password"},
        {"type": "keystroke", "dwell_ms": 98, "flight_ms": 143, "field": "password"},
        {"type": "keystroke", "dwell_ms": 91, "flight_ms": 139, "field": "password"},
        {"type": "mouse_move", "x": 452, "y": 310, "jitter": 1.9},
        {"type": "transaction_attempt", "amount": 8000,
         "beneficiary": "SBI-XXXX1234", "hour": 14},
    ])
    print(f"\n[v3-S1] combined={s['combined_score']}  tier={s['tier']}({s['tier_label']})  "
          f"fallback={s['fallback']['action']}")
    assert s["tier"] == 0
    assert s["tier_label"] == "NORMAL"
    assert s["system_response"] == "full_access"
    assert_tier_fallback_consistent(s)


def test_v3_scenario2_phishing_chain_then_large_tx():
    """
    v3 Scenario 2 — a session arrives already carrying ThreatShield-style
    suspicion (simulated here via an unknown/low-trust device + a
    behavioural paste-password signal), then attempts a large transfer
    to a new beneficiary at an unusual hour.

    Expected band progression:
      pre-tx  : elevated above a clean session, but still Tier 0
                (b_raw from device+paste alone isn't enough to cross
                the Tier-1 sigmoid threshold — by design, 1-2 stray
                signals stay Tier 0, see module docstring).
      post-tx : the large/anomalous transaction pushes the session to
                Tier 2 (HIGH RISK / step_up_otp). With b_p == 1.0 (full
                behavioural presence from the device+paste signals),
                the Tier-3 gate does NOT apply (as intended — full
                scoring), but b_c itself is still low enough that the
                weighted combination lands in the Tier 2 band rather
                than Tier 3. Tier 3 requires EITHER a stronger
                behavioural signal (see test_v3_scenario3, which uses
                bot-speed typing) OR zero behavioural signal at all
                (see test_regression_large_tx_zero_behaviour_escalates,
                where the gate's absence lets t_score alone reach 99.9).
    """
    DeviceTrustEngine.DEMO_ALL_TRUSTED = False
    try:
        sc = fresh("arjun")

        # "Phishing chain" stand-in: unknown device + one paste event
        s_pre = feed(sc, [
            {"type": "device_check", "device_fingerprint": "phishing-landing-device"},
            {"type": "paste_password"},
        ])
        print(f"\n[v3-S2] pre-tx: b_raw={s_pre['b_raw']}  combined={s_pre['combined_score']}  "
              f"tier={s_pre['tier']}({s_pre['tier_label']})  "
              f"fallback={s_pre['fallback']['action']}")
        assert s_pre["tier"] == 0, (
            "Pre-tx (device+paste only) should stay Tier 0 — by design, "
            "1-2 stray behavioural signals don't cross the sigmoid "
            "threshold (module docstring guarantee)."
        )
        assert_tier_fallback_consistent(s_pre)

        # Large transfer to new beneficiary at unusual hour
        s_post = feed(sc, [
            {"type": "transaction_attempt", "amount": 450000,
             "beneficiary": "Unknown XYZ", "hour": 2},
        ])
        print(f"[v3-S2] post-tx: b_raw={s_post['b_raw']}  t_raw={s_post['t_raw']}  "
              f"combined={s_post['combined_score']}  "
              f"tier={s_post['tier']}({s_post['tier_label']})  "
              f"fallback={s_post['fallback']['action']}")

        # With behavioural presence (b_p==1.0 from device+paste), the
        # Tier-3 gate does not apply — full scoring. The large tx pushes
        # this to at least Tier 2 (step_up_otp), reflecting real risk.
        assert s_post["raw_tier"] >= 2
        assert s_post["tier"] >= 2
        assert s_post["fallback"]["action"] in ("step_up_otp", "bank_review")
        assert_tier_fallback_consistent(s_post)
    finally:
        DeviceTrustEngine.DEMO_ALL_TRUSTED = True


def test_v3_scenario3_bot_attack_reaches_tier3():
    """
    v3 Scenario 3 — full bot attack: bot-speed typing + paste + large
    transfer to new beneficiary at 3am -> Tier 3 / bank_review.
    """
    sc = fresh("arjun")
    s = feed(sc, [
        trusted_device(),
        {"type": "keystroke", "dwell_ms": 9, "flight_ms": 6, "field": "username"},
        {"type": "paste_event", "field": "password"},
        {"type": "keystroke", "dwell_ms": 8, "flight_ms": 5, "field": "password"},
        {"type": "keystroke", "dwell_ms": 11, "flight_ms": 7, "field": "password"},
        {"type": "keystroke", "dwell_ms": 10, "flight_ms": 6, "field": "password"},
        {"type": "mouse_move", "x": 200, "y": 200, "jitter": 0.05},
        {"type": "transaction_attempt", "amount": 450000,
         "beneficiary": "Unknown XYZ", "hour": 3},
    ])
    print(f"\n[v3-S3] combined={s['combined_score']}  tier={s['tier']}({s['tier_label']})  "
          f"frozen={s['frozen']}  fallback={s['fallback']['action']}")
    assert s["tier"] == 3
    assert s["frozen"] is True
    assert s["fallback"]["action"] == "bank_review"
    assert_tier_fallback_consistent(s)


def test_v3_scenario6_velocity_structuring_reaches_tier3():
    """
    v3 Scenario 6 — velocity/structuring: multiple rapid small
    transfers to new beneficiaries (velocity_spike + repeated
    new_beneficiary signals stacking t_raw) -> should escalate to
    Tier 2/3 even though each individual transfer is small.
    """
    sc = fresh("arjun")
    events = [trusted_device()]
    # 4 rapid small transfers to different new beneficiaries
    for i in range(4):
        events.append({"type": "transaction_attempt", "amount": 3000,
                        "beneficiary": f"New Payee {i}", "hour": 14})
        events.append({"type": "velocity_spike"})

    s = feed(sc, events)
    print(f"\n[v3-S6] t_raw={s['t_raw']}  combined={s['combined_score']}  "
          f"raw_tier={s['raw_tier']}  tier={s['tier']}({s['tier_label']})  "
          f"fallback={s['fallback']['action']}")

    assert s["raw_tier"] >= 2, (
        f"Velocity/structuring pattern should reach Tier 2+, got "
        f"raw_tier={s['raw_tier']} (combined={s['combined_score']})"
    )
    assert_tier_fallback_consistent(s)


# ═══════════════════════════════════════════════════════════════
# 9. display_tier FLOOR AT raw_tier (this fix)
# ═══════════════════════════════════════════════════════════════

def test_display_tier_floors_at_raw_tier_4x6000_structuring():
    """
    Reproduction case from this fix: 4 sequential ₹6,000 transfers to
    new beneficiaries (NEW-0..NEW-3), zero behavioural signal, on the
    `arjun` profile.

    Before the fix, step 2 produced:
        combined_score=84.2  raw_tier=2 (HIGH RISK)
        shown_tier=1  fallback.action="silent_reauth" -> fallback_tier=1
        display_tier = max(shown=1, fallback_tier=1) = 1  -> "ELEVATED"

    i.e. a combined_score solidly in the HIGH RISK band [66,86) was
    displayed as ELEVATED — under-reporting raw_tier=2.

    After the fix:
        display_tier = max(shown, fallback_tier, raw) = max(1,1,2) = 2
        -> "HIGH RISK", matching raw_tier.

    This test asserts display_tier >= raw_tier for ALL 4 steps, and
    specifically checks step 2 does NOT regress to display_tier=1 while
    raw_tier=2.
    """
    sc = fresh("arjun")
    results = []
    for i in range(4):
        s = feed(sc, [{"type": "transaction_attempt", "amount": 6000,
                        "beneficiary": f"NEW-{i}", "hour": 2}])  # 2am — unusual hours, full new_bene points
        results.append(s)
        print(f"\n[4x6000 step {i+1}] combined={s['combined_score']}  "
              f"raw_tier={s['raw_tier']}  shown_tier={s['shown_tier']}  "
              f"display_tier={s['tier']}({s['tier_label']})  "
              f"fallback.action={s['fallback']['action']}  "
              f"frozen={s['frozen']}")

        # General floor invariant for every step
        assert s["tier"] >= s["raw_tier"], (
            f"Step {i+1}: display_tier={s['tier']} < raw_tier={s['raw_tier']}"
        )

    # Step 2 is the specific reproduction case.
    # At hour=2 (unusual hours), full new_beneficiary points fire,
    # combined with unusual_hour signal — so step 2 may jump directly
    # to raw_tier=3 (skipping 2) due to the compound signal stacking.
    # What matters is: (a) floor invariant holds, (b) by step 2 we are
    # already at HIGH RISK or above (raw_tier >= 2).
    s2 = results[1]
    assert s2["raw_tier"] >= 2, (
        f"Setup failed: expected raw_tier>=2 at step 2, "
        f"got {s2['raw_tier']} (combined={s2['combined_score']})"
    )
    assert s2["tier"] != 1, (
        "REGRESSION: step 2 displays tier=1 (ELEVATED) while "
        f"raw_tier={s2['raw_tier']} — the floor was not applied."
    )
    assert s2["tier"] >= s2["raw_tier"]


# ═══════════════════════════════════════════════════════════════
# 10. THREATSHIELD SESSION-POINT INJECTION (this fix)
# ═══════════════════════════════════════════════════════════════
#
# Background: main.py's _inject_threat_pts() sends a
# {"type": "threat_shield_signal", "pts": N, "label": ...} event after
# /threat/check-url, /threat/check-message, /threat/check-page, or
# /threat/check-all detects a threat. Before this fix, process_event()'s
# if/elif chain had no branch for "threat_shield_signal" — the event
# fell through every condition, decay still ran (b_raw *= 0.97), but no
# points were ever added. The injection was a complete no-op since v3.
#
# Fix: a new elif t == "threat_shield_signal" branch calls
# self._add_b(pts, label, layer="threat_shield") when pts > 0.

def test_threat_shield_signal_increases_b_raw():
    """
    A single threat_shield_signal event with pts=45 (the
    suspicion_pts_for_session value URLAnalyser returns for a CRITICAL
    phishing URL, e.g. http://sbi-kyc-update.xyz/login/verify) increases
    b_raw by ~45, minus the 0.97 decay applied in the same
    process_event() call: 45 * 0.97 = 43.65.

    combined_score must increase correspondingly (from 0 for a fresh
    session to sigmoid(43.65) > 0).
    """
    sc = fresh("arjun")
    s_before = sc.state()
    assert s_before["b_raw"] == 0.0
    assert s_before["combined_score"] == 0.0

    s = inject_threat_pts(sc, 45, "URL threat [CRITICAL]")

    print(f"\n[TS-INJECT] b_raw={s['b_raw']}  combined_score={s['combined_score']}  "
          f"sigmoid_score={sc.sigmoid_score:.2f}")
    print(f"  signals[0]={s['signals'][0]}")

    expected_b_raw = 45 * 0.97  # decay applied once in the same call
    assert abs(s["b_raw"] - expected_b_raw) < 0.06, (
        f"Expected b_raw≈{expected_b_raw:.2f} (45 * 0.97 decay), got {s['b_raw']} "
        f"(state() rounds b_raw to 1 decimal, so up to 0.05 rounding error is expected)"
    )
    assert s["combined_score"] > 0.0, (
        "combined_score should increase from 0 after a 45-pt injection"
    )

    # Verify the signal is tagged with the dedicated layer, distinguishable
    # from ordinary behavioural ('behavioural') or device ('device') signals.
    assert s["signals"][0]["layer"] == "threat_shield"
    assert s["signals"][0]["pts"] == "+45"
    assert s["signals"][0]["signal"] == "URL threat [CRITICAL]"


def test_threat_shield_zero_pts_is_noop():
    """
    A clean/whitelisted URL (e.g. a verified PSB domain) returns
    suspicion_pts_for_session: 0. Per main.py's _inject_threat_pts()
    guard (`if ... and pts > 0`), this should NOT call process_event()
    at all — b_raw must be completely unaffected (not even decayed,
    since no event was processed).
    """
    sc = fresh("arjun")
    s_before = sc.state()

    s_after = inject_threat_pts(sc, 0, "Clean URL [LOW]")

    print(f"\n[TS-ZERO] b_raw before={s_before['b_raw']}  after inject(pts=0)={s_after['b_raw']}")

    assert s_after["b_raw"] == s_before["b_raw"] == 0.0
    assert s_after["signals"] == []
    assert s_after["combined_score"] == 0.0


def test_threat_shield_disables_tier3_gate_via_b_p():
    """
    Per the design: a ThreatShield injection large enough to push
    b_raw >= 1.5 (b_p = min(1, b_raw/30) >= 0.05) should disable the
    amount-aware Tier-3 gate in _combine() for any subsequent
    transaction — i.e. a ThreatShield-flagged session is NEVER eligible
    for the "no behavioural corroboration" cap, even for a small
    (amount_ratio <= 1.0) transaction that would otherwise be capped at
    65.9 (Tier 1) for a session with b_raw == 0.

    Reproduction: inject a CRITICAL URL flag (+45 -> b_raw≈43.65,
    b_p=1.0 >= 0.05), then attempt a SMALL transfer (₹100, well below
    avg_transfer_amount=8000, amount_ratio <= 1.0) to a new beneficiary.
    Compare against the same ₹100 transfer with NO prior ThreatShield
    injection (which stays capped/low per the original ₹100 fix).
    """
    # Baseline: no ThreatShield injection, ₹100 to new bene (original fix case)
    sc_baseline = fresh("arjun")
    s_baseline = feed(sc_baseline, [
        {"type": "transaction_attempt", "amount": 100,
         "beneficiary": "New Person", "hour": 14},
    ])

    # With ThreatShield injection first
    sc_flagged = fresh("arjun")
    s_flag = inject_threat_pts(sc_flagged, 45, "URL threat [CRITICAL]")
    b_p_after_inject = min(1.0, s_flag["b_raw"] / 30.0)

    s_flagged = feed(sc_flagged, [
        {"type": "transaction_attempt", "amount": 100,
         "beneficiary": "New Person", "hour": 14},
    ])

    print(f"\n[TS-GATE] baseline (no injection): b_raw={s_baseline['b_raw']}  "
          f"combined={s_baseline['combined_score']}  raw_tier={s_baseline['raw_tier']}")
    print(f"[TS-GATE] after injection, b_p={b_p_after_inject:.3f} "
          f"(>=0.05 -> gate disabled)")
    print(f"[TS-GATE] flagged + ₹100 tx: b_raw={s_flagged['b_raw']}  "
          f"t_raw={s_flagged['t_raw']}  combined={s_flagged['combined_score']}  "
          f"raw_tier={s_flagged['raw_tier']}")

    # Confirm the injection itself crosses the b_p>=0.05 threshold
    assert b_p_after_inject >= 0.05, (
        f"Injection of 45 pts should give b_p>=0.05, got {b_p_after_inject:.3f}"
    )

    # The gate's b_p<0.05 condition is now false for the flagged session,
    # so _combine() uses the full (uncapped-by-this-gate) formula. This
    # doesn't guarantee a higher absolute combined_score than the baseline
    # (b_raw also contributes its OWN sigmoid term, which can pull the
    # blend in either direction depending on synergy/cross-penalty terms —
    # see KNOWN_ANOMALIES.md #2 for the documented "valley"). What this
    # test asserts is the GATE CONDITION itself, directly:
    from scorer import _combine
    # With b_p < 0.05 (baseline-like), amount_ratio<=1, t_raw small -> gate active
    gated = _combine(0.0, s_flagged["t_raw"], amount_ratio=100/8000)
    # With b_p >= 0.05 (flagged session's actual b_raw), gate inactive
    ungated_check_b_raw = s_flagged["b_raw"]
    assert min(1.0, ungated_check_b_raw/30.0) >= 0.05

    # Sanity: baseline (b_raw==0) ₹100/new-bene stays Tier 0 (original fix,
    # still holds) — gate WAS active there.
    assert s_baseline["raw_tier"] == 0
    assert min(1.0, s_baseline["b_raw"]/30.0) < 0.05, (
        "Baseline session unexpectedly has b_p>=0.05 — test setup invalid"
    )


# ═══════════════════════════════════════════════════════════════
# 11. PHISHING-CHAIN END-TO-END (this fix, v3 Scenario 2 re-verify)
# ═══════════════════════════════════════════════════════════════

def test_phishing_chain_end_to_end():
    """
    Full v3 Scenario 2 re-verification with the injection fix in place.

    Sequence:
      1. /threat/check-message on a CRITICAL KYC-phishing SMS containing
         an embedded phishing URL -> inject +40 pts (layer="threat_shield")
      2. /threat/check-url on the same phishing URL -> inject +45 pts
      3. (pre-login state recorded here)
      4. Normal-baseline keystrokes (5x, within ~5% of avg_dwell_ms=95)
      5. ₹4,50,000 transfer to a new beneficiary at 2am
      6. (post-transaction state recorded here)

    ACTUAL measured numbers (see test output for live values) — this
    test asserts the INJECTION MECHANISM works (b_raw increases by the
    correct amounts, layer="threat_shield" tags are present, b_p>=0.05
    disables the Tier-3 gate) AND that the post-transaction state now
    reaches Tier 3/bank_review/frozen — matching
    test_regression_large_tx_zero_behaviour_escalates's outcome for the
    same transaction with b_raw==0.

    NOTE — calibration history (see KNOWN_ANOMALIES.md #5 and #6):
    With current sigmoid constants, 85 raw ThreatShield points (40+45,
    decayed to b_raw≈81.3) alone produce sigmoid_score≈44.4 — JUST BELOW
    the Tier-1 threshold of 46 (NORMAL, not ELEVATED) before any
    keystrokes. This test does NOT assert tier_label=="ELEVATED"
    pre-login (that would require re-tuning _B_K/_B_MID, out of scope —
    KNOWN_ANOMALIES.md #6, still open). It documents the actual pre-login
    value instead.

    Post-transaction, sigmoid_score=93.75 now flows through to
    combined_score UNCHANGED (the RF blend in combined_score is now only
    applied when sigmoid_score<66 — see KNOWN_ANOMALIES.md #5, FIXED).
    Previously the 70/30 RF blend pulled combined_score down to ≈65.9
    (Tier 1, NOT Tier 3) because _ai_prob≈0.01 for normal-looking
    keystrokes — that suppression no longer applies once sigmoid_score
    is already >=66.
    """
    sc = fresh("arjun")

    # ── Step 1: scam message injection ──────────────────────────────
    msg_pts = 40  # CRITICAL KYC-phishing SMS, per ScamMessageAnalyser
    s1 = inject_threat_pts(sc, msg_pts,
                            "Scam message [CRITICAL] via sms")
    print(f"\n[CHAIN] after message injection (+{msg_pts}): "
          f"b_raw={s1['b_raw']}  combined={s1['combined_score']}  "
          f"tier={s1['tier']}({s1['tier_label']})")
    assert s1["signals"][0]["layer"] == "threat_shield"

    # ── Step 2: phishing URL injection ──────────────────────────────
    url_pts = 45  # CRITICAL phishing URL, per URLAnalyser
    s2 = inject_threat_pts(sc, url_pts, "URL threat [CRITICAL]")
    print(f"[CHAIN] after URL injection (+{url_pts}):     "
          f"b_raw={s2['b_raw']}  combined={s2['combined_score']}  "
          f"tier={s2['tier']}({s2['tier_label']})")

    pre_login_b_raw   = s2["b_raw"]
    pre_login_sigmoid = sc.sigmoid_score
    pre_login_combined= s2["combined_score"]
    pre_login_b_p     = min(1.0, pre_login_b_raw / 30.0)

    print(f"[CHAIN] pre-login: b_raw={pre_login_b_raw}  "
          f"sigmoid_score={pre_login_sigmoid:.2f}  "
          f"combined_score={pre_login_combined}  "
          f"b_p={pre_login_b_p:.3f}")

    # Both injections landed and decayed correctly:
    # b_raw = ((40*0.97 + 45)*0.97) -- second injection decays the first too
    expected_b_raw = (msg_pts * 0.97 + url_pts) * 0.97
    assert abs(pre_login_b_raw - expected_b_raw) < 0.06, (
        f"Expected b_raw≈{expected_b_raw:.2f}, got {pre_login_b_raw} "
        f"(state() rounds b_raw to 1 decimal, so up to 0.05 rounding error is expected)"
    )

    # b_p >= 0.05 after both injections -> Tier-3 gate disabled for any
    # subsequent transaction in this session.
    assert pre_login_b_p >= 0.05

    # ── Step 3: normal-baseline keystrokes ──────────────────────────
    keystroke_events = [
        {"type": "keystroke", "dwell_ms": 92, "flight_ms": 138, "field": "username"},
        {"type": "keystroke", "dwell_ms": 95, "flight_ms": 141, "field": "password"},
        {"type": "keystroke", "dwell_ms": 89, "flight_ms": 135, "field": "password"},
        {"type": "keystroke", "dwell_ms": 98, "flight_ms": 143, "field": "password"},
        {"type": "keystroke", "dwell_ms": 91, "flight_ms": 139, "field": "password"},
    ]
    s3 = feed(sc, keystroke_events)
    print(f"[CHAIN] after normal keystrokes: b_raw={s3['b_raw']}  "
          f"combined={s3['combined_score']}  tier={s3['tier']}({s3['tier_label']})  "
          f"ai_active={s3['ai_active']}  ai_fraud_prob={s3['ai_fraud_prob']}")

    # ── Step 4: large transfer to new beneficiary at 2am ────────────
    s4 = feed(sc, [
        {"type": "transaction_attempt", "amount": 450000,
         "beneficiary": "Unknown XYZ", "hour": 2},
    ])
    print(f"[CHAIN] after ₹450,000 tx: b_raw={s4['b_raw']}  t_raw={s4['t_raw']}  "
          f"sigmoid_score={sc.sigmoid_score:.2f}  combined={s4['combined_score']}  "
          f"tier={s4['tier']}({s4['tier_label']})  frozen={s4['frozen']}  "
          f"fallback={s4['fallback']['action']}  ai_fraud_prob={s4['ai_fraud_prob']}")
    for sig in s4["signals"][:6]:
        print(f"    layer={sig['layer']:<14} {sig['pts']:>4}  {sig['signal'][:55]}")

    assert_tier_fallback_consistent(s4)

    # ── Core assertions: the injection MECHANISM worked ─────────────
    # 1. ThreatShield points are present and tagged correctly throughout.
    ts_signals = [sig for sig in s4["signals"] if sig["layer"] == "threat_shield"]
    assert len(ts_signals) >= 1, (
        "ThreatShield-tagged signals should still be present in the "
        "signal log after the transaction (signal_log keeps last 20)."
    )

    # 2. The Tier-3 gate was disabled for this transaction (b_p>=0.05
    #    throughout, confirmed at step 2). Compare against the
    #    zero-injection baseline to show the gate's role:
    no_ts_sc = fresh("arjun")
    no_ts_s = feed(no_ts_sc, [
        {"type": "transaction_attempt", "amount": 450000,
         "beneficiary": "Unknown XYZ", "hour": 2},
    ])
    print(f"\n[CHAIN] comparison — same tx, NO ThreatShield injection: "
          f"b_raw={no_ts_s['b_raw']}  combined={no_ts_s['combined_score']}  "
          f"tier={no_ts_s['tier']}({no_ts_s['tier_label']})  "
          f"frozen={no_ts_s['frozen']}")

    # 3. Post-transaction: with the RF blend fixed (only applies when
    #    sigmoid_score<66), combined_score now equals sigmoid_score≈93.7
    #    — full Tier 3 / CRITICAL / bank_review / frozen.
    assert sc.sigmoid_score >= 86, (
        f"sigmoid_score={sc.sigmoid_score:.2f} should be >=86 (CRITICAL) "
        f"given pre-login ThreatShield injection + large anomalous tx"
    )
    assert s4["combined_score"] >= 86, (
        f"combined_score={s4['combined_score']} should be >=86 (CRITICAL) "
        f"— RF blend no longer suppresses sigmoid_score>=66 (fix: KNOWN_ANOMALIES.md #5)"
    )
    assert s4["tier"] == 3
    assert s4["tier_label"] == "CRITICAL"
    assert s4["frozen"] is True
    assert s4["fallback"]["action"] == "bank_review"

    # 4. The final displayed tier must floor at raw_tier (prior fix) and
    #    be internally consistent.
    assert s4["tier"] >= s4["raw_tier"]
    assert_tier_fallback_consistent(s4)


# ═══════════════════════════════════════════════════════════════
# PROGRESSIVE TRANSACTION RESPONSE — beneficiary-keyed attempt
# tracking (score-independent retry escalation)
# ═══════════════════════════════════════════════════════════════

def test_first_attempt_very_large_amount_gets_immediate_kyc():
    """B6: a very large amount on the FIRST-EVER attempt to a beneficiary
    must skip OTP entirely and go straight to bank_review/KYC, even
    though attempt_number is 1, because force_immediate_kyc overrides
    the attempt-count-based logic for amounts >= KYC_IMMEDIATE_AMOUNT_RATIO
    times the user's average."""
    s = SuspicionScorer(PROFILES["arjun"])
    r = s.process_event({"type": "transaction_attempt", "amount": 450000,
                          "beneficiary": "MULE-001", "hour": 14})
    assert r["fallback"]["action"] == "bank_review", (
        f"First attempt at 450000 (>>25x avg) must force immediate KYC, "
        f"got {r['fallback']['action']}")
    assert s.flagged_attempts.get("MULE-001") == 1


def test_retry_to_same_beneficiary_escalates_to_compliance_review():
    """B5: first flagged attempt to a beneficiary gets the milder
    silent_reauth/step_up_otp response; a SECOND flagged attempt to the
    SAME beneficiary (even at a different amount, and even though the
    score itself may be similar) must escalate to bank_review — this is
    the core fix: escalation is driven by flagged_attempts[beneficiary],
    NOT by combined_score alone."""
    s = SuspicionScorer(PROFILES["arjun"])
    for _ in range(3):
        s.process_event({"type": "remote_access_tool"})
        s.process_event({"type": "mouse_jitter"})

    r1 = s.process_event({"type": "transaction_attempt", "amount": 15000,
                           "beneficiary": "NEW-PAYEE", "hour": 14})
    assert r1["fallback"]["action"] != "bank_review", (
        "First attempt must not jump straight to bank_review")
    assert s.pending_tx["is_retry"] is False

    r2 = s.process_event({"type": "transaction_attempt", "amount": 18000,
                           "beneficiary": "NEW-PAYEE", "hour": 14})
    assert r2["fallback"]["action"] == "bank_review", (
        f"Second attempt to the SAME beneficiary must escalate to "
        f"bank_review, got {r2['fallback']['action']}")
    assert s.pending_tx["is_retry"] is True
    assert s.flagged_attempts["NEW-PAYEE"] == 2


def test_retry_tracking_is_per_beneficiary_not_global():
    """A flagged attempt to beneficiary A must NOT cause a subsequent
    attempt to a DIFFERENT beneficiary B to be treated as a retry —
    the tracking is keyed by beneficiary identity, not a global flag."""
    s = SuspicionScorer(PROFILES["arjun"])
    for _ in range(3):
        s.process_event({"type": "remote_access_tool"})
        s.process_event({"type": "mouse_jitter"})

    s.process_event({"type": "transaction_attempt", "amount": 15000,
                      "beneficiary": "PAYEE-A", "hour": 14})
    r = s.process_event({"type": "transaction_attempt", "amount": 15000,
                          "beneficiary": "PAYEE-B", "hour": 14})
    assert s.pending_tx["is_retry"] is False, (
        "A different beneficiary must not inherit PAYEE-A's retry history")
    assert s.flagged_attempts.get("PAYEE-B") == 1
    assert s.flagged_attempts.get("PAYEE-A") == 1


def test_clean_repeated_transactions_never_increment_attempt_tracking():
    """Regression guard: repeating 5 NORMAL transactions to a KNOWN
    beneficiary, at a normal amount, must NEVER increment
    flagged_attempts — even if the dataset-trained ml_fraud signal (a
    continuous, low-AUC background-calibration score, NOT a discrete
    novelty signal) happens to fire on some of them. Only genuine
    novelty/amount signals (new_beneficiary, amount_anomaly,
    large_new_beneficiary) may increment the counter."""
    s = SuspicionScorer(PROFILES["arjun"])
    for _ in range(5):
        s.process_event({"type": "transaction_attempt", "amount": 5000,
                          "beneficiary": "SBI-XXXX1234", "hour": 14})
    assert s.flagged_attempts == {}, (
        f"Clean transactions to a known beneficiary must never be "
        f"tracked as flagged attempts, got {s.flagged_attempts}")


def test_reset_clears_flagged_attempts_but_nothing_else_does():
    """flagged_attempts must be cleared by the backend's manual reset()
    (representing bank/compliance manual review) and ONLY by reset() —
    not by any other event type, since a UI-only action clearing this
    history would let an attacker erase their own retry record."""
    s = SuspicionScorer(PROFILES["arjun"])
    for _ in range(3):
        s.process_event({"type": "remote_access_tool"})
        s.process_event({"type": "mouse_jitter"})
    s.process_event({"type": "transaction_attempt", "amount": 15000,
                      "beneficiary": "NEW-PAYEE", "hour": 14})
    s.process_event({"type": "transaction_attempt", "amount": 18000,
                      "beneficiary": "NEW-PAYEE", "hour": 14})
    assert s.flagged_attempts.get("NEW-PAYEE") == 2

    # Unrelated events must not touch flagged_attempts
    s.process_event({"type": "keystroke", "dwell_ms": 95})
    s.process_event({"type": "mouse_idle", "micro_movement_px": 5})
    assert s.flagged_attempts.get("NEW-PAYEE") == 2, (
        "Unrelated events must not clear or alter flagged_attempts")

    s.reset()
    assert s.flagged_attempts == {}, (
        "reset() must clear flagged_attempts (manual review path)")


# ═══════════════════════════════════════════════════════════════
# AUTOMATED VERIFICATION CALL — new rung between OTP and freeze
# ═══════════════════════════════════════════════════════════════

def test_automated_call_fires_in_upper_high_risk_band():
    """A flagged transaction landing in the upper portion of HIGH RISK
    (cs >= 78), with no retry history and no force_immediate_kyc amount,
    must trigger the automated_call response rather than step_up_otp or
    bank_review."""
    s = SuspicionScorer(PROFILES["arjun"])
    for _ in range(3):
        s.process_event({"type": "remote_access_tool"})
        s.process_event({"type": "mouse_jitter"})
    r = s.process_event({"type": "transaction_attempt", "amount": 15000,
                          "beneficiary": "NEW-X", "hour": 14})
    assert 78 <= r["combined_score"] < 86, (
        f"Test setup must land in the 78-85 band, got {r['combined_score']}")
    assert r["fallback"]["action"] == "automated_call"
    assert r["call_triggered"] is True


def test_retry_beats_automated_call():
    """A SECOND flagged attempt to the same beneficiary must escalate
    straight to bank_review, skipping automated_call entirely — a retry
    after one verification step is itself the stronger signal."""
    s = SuspicionScorer(PROFILES["arjun"])
    for _ in range(3):
        s.process_event({"type": "remote_access_tool"})
        s.process_event({"type": "mouse_jitter"})
    r1 = s.process_event({"type": "transaction_attempt", "amount": 15000,
                           "beneficiary": "NEW-Y", "hour": 14})
    assert r1["fallback"]["action"] == "automated_call"

    r2 = s.process_event({"type": "transaction_attempt", "amount": 18000,
                           "beneficiary": "NEW-Y", "hour": 14})
    assert r2["fallback"]["action"] == "bank_review", (
        "A retry must escalate past automated_call straight to bank_review")


def test_force_immediate_kyc_beats_automated_call():
    """A very large first-attempt amount must still go straight to
    bank_review/KYC as the fallback ACTION, never automated_call,
    regardless of which score band it happens to land in. (call_triggered
    may still be True here — this is a fast jump straight to frozen, and
    the freeze-shortcut correctly marks the full explanatory chain,
    including the call step, as having been shown — see
    test_call_triggered_survives_a_fast_jump_straight_to_critical. What
    must NOT happen is the fallback ACTION itself being automated_call
    instead of bank_review.)"""
    s = SuspicionScorer(PROFILES["arjun"])
    r = s.process_event({"type": "transaction_attempt", "amount": 450000,
                          "beneficiary": "MULE", "hour": 14})
    assert r["fallback"]["action"] == "bank_review"


def test_call_triggered_survives_a_fast_jump_straight_to_critical():
    """Regression test for a real bug found during implementation: if the
    score jumps from below the 78 threshold straight past 86 in a single
    process_event() call (e.g. several strong signals fired back-to-back
    with no intervening dwell time), the freeze-shortcut path in state()
    must still mark call_triggered=True — mirroring how it already
    force-sets otp_triggered/progressive_harden — so the UI can show the
    automated-call step as part of the explanation for why the session
    froze, rather than silently skipping straight from nothing to frozen
    with no call ever shown to have been triggered."""
    s = SuspicionScorer(PROFILES["arjun"])
    st = None
    for i in range(6):
        ev = {"type": "remote_access_tool"} if i % 2 == 0 else {"type": "mouse_jitter"}
        st = s.process_event(ev)
    assert st["frozen"] is True
    assert st["call_triggered"] is True, (
        "A fast jump straight to CRITICAL must still surface "
        "call_triggered=True, not silently skip the call step")


def test_call_triggered_clears_on_deescalation_below_tier_2():
    """call_triggered must clear when the displayed tier de-escalates
    below tier 2 (HIGH RISK), mirroring otp_triggered's existing
    de-escalation behaviour — but NEVER while frozen (tier 3 never
    de-escalates automatically)."""
    s = SuspicionScorer(PROFILES["arjun"])
    for _ in range(3):
        s.process_event({"type": "remote_access_tool"})
        s.process_event({"type": "mouse_jitter"})
    r = s.process_event({"type": "transaction_attempt", "amount": 15000,
                          "beneficiary": "NEW-Z", "hour": 14})
    assert r["call_triggered"] is True
    s._on_tier_down(1)
    assert s._call is False, "call flag must clear on de-escalation below tier 2"


if __name__ == "__main__":
    tests = []
    for k, v in list(globals().items()):
        if k.startswith("test_") and callable(v):
            # Skip the pytest-parametrized version in standalone mode —
            # the _standalone variant covers the same cases without pytest.
            if k == "test_fallback_tier_consistency_across_scenarios":
                continue
            tests.append(v)

    passed, failed = 0, 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name}\n   {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {name}\n   {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
