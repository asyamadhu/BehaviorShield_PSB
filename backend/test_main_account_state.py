# =============================================================
# test_main_account_state.py — Integration tests for main.py's
# cross-session ACCOUNT_STATE (freeze + probation persistence).
#
# test_scorer.py exercises SuspicionScorer directly and does NOT
# cover this: a fresh SuspicionScorer per websocket connection means
# scorer-level tests can never observe what happens ACROSS a
# reconnect. This file drives the actual FastAPI app (websocket +
# HTTP) via TestClient, the same way a real browser reload would.
#
# Run: python3 test_main_account_state.py
# =============================================================

import os, sys, io, contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

with contextlib.redirect_stderr(io.StringIO()):
    from fastapi.testclient import TestClient
    import main as main_module


REAL_ANSWERS = {"sq1": "Bunty", "sq2": "St Xaviers", "sq3": "Rocky",
                "sq4": "Lucknow", "sq5": "Rajma Chawal", "sq6": "Hero Splendor"}


def _client():
    return TestClient(main_module.app)


def _reset(c, profile):
    c.post(f"/reset/{profile}")


def _push_to_high_risk(ws):
    """Realistic signal volume -- calibrated against the live server,
    NOT the lighter 3-event helper used in test_scorer.py (which never
    verified it actually reaches HIGH RISK, only that KBA's internal
    methods work in isolation)."""
    st = None
    for ev in ["paste_password", "mouse_jitter", "mouse_jitter", "mouse_jitter", "mouse_jitter"]:
        ws.send_json({"type": ev})
        st = ws.receive_json()
    return st


def _grant_probation(c, profile):
    with c.websocket_connect(f"/ws/{profile}") as ws:
        ws.receive_json()  # initial state
        _push_to_high_risk(ws)
        q1 = c.get(f"/kba/question/{profile}").json()
        c.post(f"/kba/verify/{profile}", json={"question_id": q1["question_id"], "answer": "wrong"})
        q2 = c.get(f"/kba/question/{profile}").json()
        c.post(f"/kba/verify/{profile}", json={"question_id": q2["question_id"], "answer": "wrong"})
        q3 = c.get(f"/kba/question/{profile}").json()
        r = c.post(f"/kba/verify/{profile}",
                    json={"question_id": q3["question_id"],
                          "answer": REAL_ANSWERS[q3["question_id"]], "via_call": True}).json()
    return r


def _fail_login_freeze(c, profile):
    with c.websocket_connect(f"/ws/{profile}") as ws:
        ws.receive_json()
        st = None
        for _ in range(3):
            ws.send_json({"type": "failed_login"})
            st = ws.receive_json()
    return st


# ── Tests ────────────────────────────────────────────────────

def test_freeze_persists_across_reconnect_without_reset():
    """The core bug: a frozen session must stay frozen on the NEXT
    websocket connection (simulating a page reload) even though a
    brand-new SuspicionScorer is created for it. Only /reset should
    ever clear this."""
    c = _client()
    _reset(c, "arjun")

    st1 = _fail_login_freeze(c, "arjun")
    assert st1["frozen"] is True
    assert st1["tier_label"] == "CRITICAL"

    with c.websocket_connect("/ws/arjun") as ws2:
        st2 = ws2.receive_json()
    assert st2["frozen"] is True, "freeze must survive a reconnect"
    assert st2["tier_label"] == "CRITICAL", (
        "tier_label must stay consistent with frozen=True, not silently "
        "read NORMAL just because b_raw reset to 0 on the new scorer")

    _reset(c, "arjun")


def test_freeze_persists_across_multiple_reconnects():
    """Not just a one-time grace reconnect -- must hold indefinitely
    until an explicit reset."""
    c = _client()
    _reset(c, "arjun")
    _fail_login_freeze(c, "arjun")

    for _ in range(3):
        with c.websocket_connect("/ws/arjun") as ws:
            st = ws.receive_json()
        assert st["frozen"] is True

    _reset(c, "arjun")


def test_reset_is_the_only_way_out_of_freeze():
    c = _client()
    _reset(c, "arjun")
    _fail_login_freeze(c, "arjun")

    acct = c.get("/account/status/arjun").json()
    assert acct["frozen"] is True

    c.post("/reset/arjun")
    acct2 = c.get("/account/status/arjun").json()
    assert acct2["frozen"] is False

    with c.websocket_connect("/ws/arjun") as ws:
        st = ws.receive_json()
    assert st["frozen"] is False
    assert st["tier_label"] == "NORMAL"


def test_account_status_reflects_frozen_even_without_live_session():
    """/account/status must be queryable (e.g. by a dashboard) without
    requiring an active websocket connection."""
    c = _client()
    _reset(c, "arjun")
    _fail_login_freeze(c, "arjun")
    # No websocket open at this point -- purely cross-session lookup.
    acct = c.get("/account/status/arjun").json()
    assert acct["frozen"] is True
    _reset(c, "arjun")


def test_probation_and_freeze_do_not_interfere():
    """A probation-capped account reconnecting must NOT be
    misidentified as frozen, and vice versa -- they are independent
    account states."""
    c = _client()
    _reset(c, "attacker")

    r = _grant_probation(c, "attacker")
    assert r["probation"] is True
    assert r["frozen"] is False

    with c.websocket_connect("/ws/attacker") as ws:
        st = ws.receive_json()
    assert st["probation"] is True
    assert st["frozen"] is False
    assert st["tx_limit"] == r["tx_limit"]

    _reset(c, "attacker")


def test_call_verification_failure_freezes_and_persists_too():
    """The OTHER freeze path (failing the automated call, not just 3
    failed logins) must also persist across reconnect."""
    c = _client()
    _reset(c, "arjun")

    with c.websocket_connect("/ws/arjun") as ws:
        ws.receive_json()
        _push_to_high_risk(ws)
        q1 = c.get("/kba/question/arjun").json()
        c.post("/kba/verify/arjun", json={"question_id": q1["question_id"], "answer": "wrong"})
        q2 = c.get("/kba/question/arjun").json()
        c.post("/kba/verify/arjun", json={"question_id": q2["question_id"], "answer": "wrong"})
        q3 = c.get("/kba/question/arjun").json()
        r = c.post("/kba/verify/arjun",
                    json={"question_id": q3["question_id"], "answer": "still wrong", "via_call": True}).json()
    assert r["frozen"] is True

    with c.websocket_connect("/ws/arjun") as ws2:
        st2 = ws2.receive_json()
    assert st2["frozen"] is True, "freeze from a failed call must also survive reconnect"

    _reset(c, "arjun")


def test_reset_on_never_frozen_account_is_a_no_op():
    """Sanity: resetting an account that was never frozen/on probation
    must not error or produce a contradictory state."""
    c = _client()
    _reset(c, "arjun_new_device")
    acct = c.get("/account/status/arjun_new_device").json()
    assert acct["frozen"] is False
    assert acct["probation"] is False


def test_kba_episode_survives_websocket_reconnect():
    """The actual bug reported from a live deployment: reach the
    automated call step (2 wrong KBA answers), then have the websocket
    drop and reconnect BEFORE the call resolves -- e.g. a brief proxy
    hiccup, far more common on a hosted platform than on a stable
    localhost connection, which is why this never surfaced in local
    testing. Previously only frozen/probation survived a reconnect;
    kba_failed_count/asked_kba_ids/call_pending did not, so the
    reconnect silently reset progress to zero -- reaching the call,
    then appearing to "fall back to only KBA" as if the 2nd wrong
    answer never happened."""
    c = _client()
    _reset(c, "arjun")

    with c.websocket_connect("/ws/arjun") as ws:
        ws.receive_json()
        _push_to_high_risk(ws)
        q1 = c.get("/kba/question/arjun").json()
        c.post("/kba/verify/arjun", json={"question_id": q1["question_id"], "answer": "wrong"})
        q2 = c.get("/kba/question/arjun").json()
        r2 = c.post("/kba/verify/arjun", json={"question_id": q2["question_id"], "answer": "wrong"}).json()
        assert r2["call_pending"] is True, "setup should reach call_pending before reconnecting"

    # Reconnect (simulates a dropped/reestablished websocket) BEFORE
    # the call is ever answered.
    with c.websocket_connect("/ws/arjun") as ws2:
        st = ws2.receive_json()
        assert st["call_pending"] is True, (
            "call_pending must survive a reconnect -- this is the exact bug: "
            "silently reverting to 'only KBA' after reaching the call")
        assert st["kba_failed_count"] == 2

    # The call must still be answerable after reconnecting, and still
    # resolve exactly like an uninterrupted session would.
    q3 = c.get("/kba/question/arjun").json()
    r3 = c.post("/kba/verify/arjun", json={"question_id": q3["question_id"], "answer": "wrong", "via_call": True}).json()
    assert r3["frozen"] is True

    _reset(c, "arjun")


def test_kba_episode_progress_cleared_once_resolved():
    """An episode that concludes (frozen, here) must not leave stale
    kba_failed_count/asked_kba_ids/call_pending sitting in
    ACCOUNT_STATE to confuse a future reset()+retry."""
    c = _client()
    _reset(c, "arjun")

    with c.websocket_connect("/ws/arjun") as ws:
        ws.receive_json()
        _push_to_high_risk(ws)
        q1 = c.get("/kba/question/arjun").json()
        c.post("/kba/verify/arjun", json={"question_id": q1["question_id"], "answer": "wrong"})
        q2 = c.get("/kba/question/arjun").json()
        c.post("/kba/verify/arjun", json={"question_id": q2["question_id"], "answer": "wrong"})
        q3 = c.get("/kba/question/arjun").json()
        c.post("/kba/verify/arjun", json={"question_id": q3["question_id"], "answer": "wrong", "via_call": True})

    acct = c.get("/account/status/arjun").json()
    assert acct["frozen"] is True
    assert acct["kba_failed_count"] == 0, "episode bookkeeping must clear once frozen concludes it"
    assert acct["call_pending"] is False

    _reset(c, "arjun")


if __name__ == "__main__":
    tests = []
    for k, v in list(globals().items()):
        if k.startswith("test_") and callable(v):
            tests.append((k, v))

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name} -- {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {name} -- {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
