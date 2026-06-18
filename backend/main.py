# =============================================================
# main.py — BehaviorShield Backend
# Run: uvicorn main:app --reload --port 8000
# =============================================================

import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scorer import SuspicionScorer
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


@app.websocket("/ws/{profile_name}")
async def ws_endpoint(ws: WebSocket, profile_name: str):
    await ws.accept()
    profile = PROFILES.get(profile_name, PROFILES["arjun"])
    scorer  = SuspicionScorer(profile)
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
        sessions.pop(profile_name, None)


class TxRequest(BaseModel):
    amount:      float
    beneficiary: str
    hour:        Optional[int] = None
    profile:     Optional[str] = "arjun"

@app.post("/transaction/{profile_name}")
async def score_transaction(profile_name: str, req: TxRequest):
    if profile_name not in sessions:
        sessions[profile_name] = SuspicionScorer(
            PROFILES.get(profile_name, PROFILES["arjun"]))
    return sessions[profile_name].process_event({
        "type":        "transaction_attempt",
        "amount":      req.amount,
        "beneficiary": req.beneficiary,
        "hour":        req.hour if req.hour is not None else datetime.now().hour,
    })

@app.post("/reset/{profile_name}")
async def reset(profile_name: str):
    if profile_name in sessions:
        sessions[profile_name].reset()
    return {"status": "ok"}

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
