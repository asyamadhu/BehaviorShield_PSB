# =============================================================
# main.py — BehaviorShield Backend
# Run: uvicorn main:app --reload --port 8000
# =============================================================

import json
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scorer import SuspicionScorer, CLEAN_SESSIONS_TO_CLEAR
from profiles import PROFILES

try:
    from threat_shield import (
        URLAnalyser, ScamMessageAnalyser, FakePageAnalyser,
        comprehensive_check, SCAM_PATTERNS, PAGE_SIGNAL_WEIGHTS,
    )
    _threat_enabled = True
    print("[BehaviorShield] ThreatShield layer loaded — URL / Scam / FakePage detection active")
except ImportError as _e:
    print(f"[BehaviorShield] WARNING: ThreatShield module failed to load: {_e}")
    _threat_enabled = False

app = FastAPI(title="BehaviorShield")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# One scorer per active session
sessions: dict[str, SuspicionScorer] = {}

# ══════════════════════════════════════════════════════════════
# ACCOUNT-LEVEL STATE — survives across logins (websocket reconnects)
# within this server process. Deliberately separate from `sessions`,
# which is a fresh SuspicionScorer per connection and forgets
# everything (even _frozen) on reconnect. Probation from a passed
# call-verification needs to outlive that, so it's tracked here and
# re-seeded into each new scorer at connect time.
#
# Resets on server restart — fine for a demo; production would persist
# this on the account's DB record instead of an in-memory dict.
# ══════════════════════════════════════════════════════════════
ACCOUNT_STATE: dict[str, dict] = {}
# profile_name -> {"probation": bool, "tx_limit": float|None, "clean_streak": int,
#                   "frozen": bool}


def _get_account_state(profile_name: str) -> dict:
    return ACCOUNT_STATE.setdefault(profile_name, {
        "probation": False, "tx_limit": None, "clean_streak": 0,
        "frozen": False,
    })


def _seed_scorer_from_account(scorer: SuspicionScorer, profile_name: str):
    """Re-apply cross-session state (if any) to a freshly-created
    scorer, so a capped/monitored OR frozen account stays that way
    across a logout/login (e.g. a page reload) rather than resetting
    to full trust — only an explicit /reset (manual bank review)
    should ever clear either of these.

    Freezing a fresh scorer is more than just flipping `_frozen`: the
    displayed tier_label/tier_action are derived from combined_score
    (via display_tier = max(shown, fallback_tier, raw) in state()),
    NOT from `_frozen` directly — a brand-new scorer starts at
    b_raw=0, so combined_score=0 and display_tier would read 0
    (NORMAL) even with `_frozen=True` set, producing a contradictory
    "NORMAL but blocked" state. Force the same tier-3 fields the
    in-session freeze block itself sets (see state()'s
    `display_tier == 3` branch) so a reloaded frozen session shows
    CRITICAL/bank_review consistently, not just a blocked button.
    """
    acct = _get_account_state(profile_name)
    if acct["probation"]:
        scorer._probation          = True
        scorer._probation_tx_limit = acct["tx_limit"]
        scorer._tx_limit           = acct["tx_limit"]
    if acct["frozen"]:
        scorer._frozen       = True
        scorer._otp          = True
        scorer._call         = True
        scorer._harden       = True
        scorer._shown_tier   = 3
        scorer._tier_times[3] = time.time()
        scorer.max_shown_tier = 3


def _settle_account_on_disconnect(scorer: SuspicionScorer, profile_name: str):
    """Called when a session ends. Persists a freeze (if this session
    reached one) forward unconditionally — freeze is absolute and has
    no clean-streak recovery path, only reset() clears it, so there is
    nothing further to compute once it's set. Otherwise persists
    newly-granted probation, and — for accounts already on probation —
    advances or resets the consecutive-clean-session streak, clearing
    probation once CLEAN_SESSIONS_TO_CLEAR calm sessions have passed in
    a row. A single session that re-escalates past Tier 1 resets the
    streak to 0, so an attacker can't wait out probation with idle
    logins."""
    acct = _get_account_state(profile_name)

    if scorer._frozen:
        acct["frozen"] = True
        # A frozen session is not simultaneously eligible for the
        # ordinary probation-clean-streak bookkeeping below — freeze
        # is the more severe outcome and reset() is the only way out
        # of either, so there's nothing more to settle this round.
        return

    if scorer._probation and not acct["probation"]:
        # Probation was granted THIS session (call verification just
        # passed) — persist it forward as the account's new baseline.
        acct["probation"]    = True
        acct["tx_limit"]     = scorer._probation_tx_limit
        acct["clean_streak"] = 0
        return

    if not acct["probation"]:
        return  # nothing to settle

    if scorer.max_shown_tier <= 1:
        acct["clean_streak"] += 1
    else:
        acct["clean_streak"] = 0

    if acct["clean_streak"] >= CLEAN_SESSIONS_TO_CLEAR:
        acct["probation"]    = False
        acct["tx_limit"]     = None
        acct["clean_streak"] = 0


@app.websocket("/ws/{profile_name}")
async def ws_endpoint(ws: WebSocket, profile_name: str):
    await ws.accept()
    profile = PROFILES.get(profile_name, PROFILES["arjun"])
    scorer  = SuspicionScorer(profile)
    _seed_scorer_from_account(scorer, profile_name)
    sessions[profile_name] = scorer
    print(f"[+] {profile_name}")
    await ws.send_json(scorer.state())
    try:
        while True:
            data  = await ws.receive_text()
            event = json.loads(data)
            st    = scorer.process_event(event)
            await ws.send_json(st)
    except WebSocketDisconnect:
        print(f"[-] {profile_name}")
        _settle_account_on_disconnect(scorer, profile_name)
        sessions.pop(profile_name, None)


class TxRequest(BaseModel):
    amount:      float
    beneficiary: str
    hour:        Optional[int] = None
    profile:     Optional[str] = "arjun"

@app.post("/transaction/{profile_name}")
async def score_transaction(profile_name: str, req: TxRequest):
    scorer = _get_or_create_session(profile_name)
    return scorer.process_event({
        "type":        "transaction_attempt",
        "amount":      req.amount,
        "beneficiary": req.beneficiary,
        "hour":        req.hour if req.hour is not None else datetime.now().hour,
    })

@app.post("/reset/{profile_name}")
async def reset(profile_name: str):
    if profile_name in sessions:
        sessions[profile_name].reset()
    # Manual review clears the account-level probation AND frozen
    # records too — matches SuspicionScorer.reset() clearing both at
    # the session level. This endpoint is the ONLY legitimate way back
    # in from either state; a reconnect alone must not do this.
    if profile_name in ACCOUNT_STATE:
        ACCOUNT_STATE[profile_name] = {
            "probation": False, "tx_limit": None, "clean_streak": 0,
            "frozen": False,
        }
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════════
# KBA (SECURITY QUESTIONS) + AUTOMATED CALL ROUTES
# ══════════════════════════════════════════════════════════════

def _get_or_create_session(profile_name: str) -> SuspicionScorer:
    if profile_name not in sessions:
        scorer = SuspicionScorer(PROFILES.get(profile_name, PROFILES["arjun"]))
        _seed_scorer_from_account(scorer, profile_name)
        sessions[profile_name] = scorer
    return sessions[profile_name]


class KBAVerifyRequest(BaseModel):
    question_id: str
    answer:      str
    via_call:    Optional[bool] = False   # True only for the automated call's question C


@app.get("/kba/question/{profile_name}")
async def kba_question(profile_name: str):
    """Return a security question not yet asked this hardening episode
    (question A, then B, then — via the automated call — C). Never
    repeats a question within the same episode."""
    scorer = _get_or_create_session(profile_name)
    q = scorer.next_kba_question()
    if q is None:
        # Pool exhausted — fail safe rather than repeat a question.
        return {"error": "question_pool_exhausted"}
    return q


@app.post("/kba/verify/{profile_name}")
async def kba_verify(profile_name: str, req: KBAVerifyRequest):
    """Verify an answer to a previously-issued question. Set
    via_call=true only for the automated call's question C — that path
    decides probation (pass) vs. permanent freeze (fail) instead of the
    ordinary +points-and-continue path used for questions A/B."""
    scorer = _get_or_create_session(profile_name)
    return scorer.verify_kba_answer(req.question_id, req.answer, via_call=bool(req.via_call))


@app.get("/account/status/{profile_name}")
async def account_status(profile_name: str):
    """Cross-session account state (probation cap, clean-session streak)
    — independent of whether a live session/websocket is connected.
    Lets the frontend show a persistent 'account under monitoring'
    banner even before any events fire this login."""
    acct = _get_account_state(profile_name)
    return {
        "profile": profile_name,
        **acct,
        "clean_sessions_to_clear": CLEAN_SESSIONS_TO_CLEAR,
    }


@app.get("/health")
async def health():
    return {"status": "running", "sessions": list(sessions.keys())}


@app.get("/model/status")
async def model_status():
    """Return loaded model metadata — shown on dashboard."""
    try:
        from tx_fraud_scorer import model_summary
        tx_info = model_summary()
    except Exception:
        tx_info = {"loaded": False}
    return {
        "behavioural_model": "model.joblib",
        "transaction_model": "tx_fraud_model.joblib",
        "tx_model_info": tx_info,
        "dataset": {
            "name":       "PSB Hackathon 2026 — DataSet_1.csv",
            "rows":       9082,
            "fraud_rate": "18.7%",
            "features":   "3,924 raw → 109 engineered",
            "label":      "F3900 (1=fraud, 0=legitimate)",
        },
    }


# ══════════════════════════════════════════════════════════════
# THREATSHIELD ROUTES  — Layer 0 Pre-Entry Protection
# ══════════════════════════════════════════════════════════════

class URLCheckRequest(BaseModel):
    url:      str
    referrer: Optional[str] = ""
    profile:  Optional[str] = None   # if set, inject suspicion pts into session

class MessageCheckRequest(BaseModel):
    text:    str
    channel: Optional[str] = "sms"   # sms | email | whatsapp
    profile: Optional[str] = None

class PageSignalRequest(BaseModel):
    signals: dict   # keys match PAGE_SIGNAL_WEIGHTS in threat_shield.py
    profile: Optional[str] = None

class ComprehensiveCheckRequest(BaseModel):
    url:             Optional[str]  = None
    message:         Optional[str]  = None
    message_channel: Optional[str]  = "sms"
    page_signals:    Optional[dict] = None
    profile:         Optional[str]  = None


def _inject_threat_pts(profile_name: str, pts: int, label: str):
    """Inject ThreatShield suspicion points into an active session scorer."""
    if profile_name and profile_name in sessions and pts > 0:
        sessions[profile_name].process_event({
            "type":   "threat_shield_signal",
            "pts":    pts,
            "label":  label,
        })


@app.post("/threat/check-url")
async def check_url(req: URLCheckRequest):
    """
    Check a URL for phishing / fake-banking-site risk.
    Optionally injects suspicion points into an active session.

    Example:
        POST /threat/check-url
        { "url": "http://sbi-kyc-update.xyz/login", "profile": "arjun" }
    """
    if not _threat_enabled:
        return {"error": "ThreatShield not loaded"}
    result = URLAnalyser().analyse(req.url, req.referrer or "")
    _inject_threat_pts(req.profile, result["suspicion_pts_for_session"],
                       f"URL threat [{result['risk_label']}]: {req.url[:60]}")
    return result


@app.post("/threat/check-message")
async def check_message(req: MessageCheckRequest):
    """
    Check an SMS / email / WhatsApp message for scam patterns.

    Example:
        POST /threat/check-message
        { "text": "Your SBI KYC will expire. Click: bit.ly/xyz", "channel": "sms" }
    """
    if not _threat_enabled:
        return {"error": "ThreatShield not loaded"}
    result = ScamMessageAnalyser().analyse(req.text, req.channel or "sms")
    _inject_threat_pts(req.profile, result["suspicion_pts_for_session"],
                       f"Scam message [{result['risk_label']}] via {req.channel}")
    return result


@app.post("/threat/check-page")
async def check_page(req: PageSignalRequest):
    """
    Receive DOM/TLS signals from the frontend JS SDK and score page risk.
    Called automatically by behaviorsignal.js on page load.

    Example signals:
        { "form_action_external": true, "no_ssl": true, "typosquat_detected": true }
    """
    if not _threat_enabled:
        return {"error": "ThreatShield not loaded"}
    result = FakePageAnalyser().analyse(req.signals)
    _inject_threat_pts(req.profile, result["suspicion_pts_for_session"],
                       f"Fake page [{result['risk_label']}]: {len(result['signals'])} signals")
    return result


@app.post("/threat/check-all")
async def check_all(req: ComprehensiveCheckRequest):
    """
    Run all applicable ThreatShield checks in one call.
    Returns combined risk assessment with session suspicion points.
    """
    if not _threat_enabled:
        return {"error": "ThreatShield not loaded"}
    result = comprehensive_check(
        url=req.url,
        message=req.message,
        message_channel=req.message_channel or "sms",
        page_signals=req.page_signals,
    )
    _inject_threat_pts(req.profile, result["session_suspicion_pts"],
                       f"Comprehensive threat [{result['overall_risk']}]")
    return result


@app.get("/threat/status")
async def threat_status():
    """Return ThreatShield module status."""
    return {
        "enabled":              _threat_enabled,
        "modules": {
            "url_analyser":     _threat_enabled,
            "scam_message":     _threat_enabled,
            "fake_page":        _threat_enabled,
        },
        "psb_domains_whitelisted": 20,
        "scam_patterns":           len(__import__('threat_shield').SCAM_PATTERNS) if _threat_enabled else 0,
        "page_signal_checks":      len(__import__('threat_shield').PAGE_SIGNAL_WEIGHTS) if _threat_enabled else 0,
        "description": (
            "ThreatShield is Layer 0 of BehaviorShield. It provides pre-entry "
            "protection by detecting phishing URLs, scam SMS/email messages, and "
            "fake banking websites before the user enters any credentials. "
            "Detected threats inject suspicion points into the behavioural session scorer."
        )
    }


# ══════════════════════════════════════════════════════════════
# DEMO CONTROL ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/demo/device-mode")
async def set_device_mode(body: dict):
    """
    Toggle device trust mode for presentations.

    POST /demo/device-mode  { "all_trusted": true }
        → Every device treated as fully trusted (default for demos)

    POST /demo/device-mode  { "all_trusted": false }
        → Real fingerprint matching — new device = probationary mode

    Use this to demonstrate the new-device scenario during a presentation
    without needing to clear browser data or switch machines.
    """
    from scorer import DeviceTrustEngine
    mode = bool(body.get("all_trusted", True))
    DeviceTrustEngine.DEMO_ALL_TRUSTED = mode
    return {
        "demo_all_trusted": mode,
        "message": (
            "All devices now trusted — fingerprint check bypassed."
            if mode else
            "Real device fingerprinting enabled — new device = probationary."
        )
    }

@app.get("/demo/device-mode")
async def get_device_mode():
    from scorer import DeviceTrustEngine
    return {"demo_all_trusted": DeviceTrustEngine.DEMO_ALL_TRUSTED}
