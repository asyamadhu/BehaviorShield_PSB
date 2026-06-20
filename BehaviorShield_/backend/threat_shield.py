# =============================================================
# threat_shield.py — BehaviorShield Pre-Entry Threat Detection
#
# PURPOSE
#   Layer 0 of BehaviorShield — runs BEFORE the behavioural
#   session begins. Protects the user at the entry point by
#   detecting:
#     1. Phishing URLs  — fake bank domains, typosquats, IDN
#     2. Scam SMS/Email — OTP harvest, KYC urgency, lottery
#     3. Fake Websites  — DOM tampering, SSL anomalies, form
#                         hijacking, visual spoofing signals
#
# POSITION IN THE STACK
#   ┌───────────────────────────────────────────────────────┐
#   │ LAYER 0  ThreatShield  (THIS FILE)                    │
#   │          ↓ blocks / warns before session starts       │
#   ├───────────────────────────────────────────────────────┤
#   │ LAYER 1  4-Zone Login Canvas  (index.html)            │
#   │ LAYER 2  BehavioralScorer     (scorer.py)             │
#   │ LAYER 3  TX Fraud ML          (tx_fraud_scorer.py)    │
#   │ LAYER 4  Staged Escalation    (scorer.py)             │
#   └───────────────────────────────────────────────────────┘
#
# DESIGN PRINCIPLES
#   • Rule-based + heuristic (no external API dependency)
#   • Works offline — no network calls at runtime
#   • Returns structured JSON — frontend decides how to warn
#   • Suspicion points feed into BehavioralScorer b_raw
#     (fake-site context makes behavioural score stricter)
#   • Never blocks unilaterally — always bank-framed response
#
# API (called from main.py)
#   POST /threat/check-url     { "url": "...", "referrer": "..." }
#   POST /threat/check-message { "text": "...", "channel": "sms|email|whatsapp" }
#   POST /threat/check-page    { "signals": {...} }  (from JS SDK)
#   GET  /threat/stats         Returns detection summary
# =============================================================

import re
import math
import hashlib
import unicodedata
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from typing import Optional


# ══════════════════════════════════════════════════════════════
# KNOWN-GOOD BANK DOMAINS (PSB whitelist)
# ══════════════════════════════════════════════════════════════

PSB_LEGITIMATE_DOMAINS = {
    # State Bank of India
    "onlinesbi.sbi", "retail.onlinesbi.com", "onlinesbi.com",
    "sbi.co.in", "sbiyono.sbi",
    # Punjab National Bank
    "netpnb.com", "pnbindia.in",
    # Bank of Baroda
    "bobibanking.com", "bankofbaroda.in",
    # Canara Bank
    "canarabank.in", "netbanking.canarabank.in",
    # Union Bank
    "unionbankofindia.co.in", "unibankonline.co.in",
    # Central Bank of India (hackathon sponsor)
    "centralbankofindia.co.in", "netbanking.centralbankofindia.co.in",
    # HDFC / ICICI / Axis (private, but commonly spoofed)
    "hdfcbank.com", "netbanking.hdfcbank.com",
    "icicibank.com", "infinityicici.com",
    "axisbank.com",
    # UPI / NPCI
    "upi.npci.org.in", "npci.org.in",
    # RBI
    "rbi.org.in",
}

# Frequently typosquatted brand tokens
PSB_BRAND_TOKENS = {
    "sbi", "pnb", "bob", "boi", "canara", "union", "central",
    "hdfc", "icici", "axis", "kotak", "npci", "upi", "yono",
    "netbanking", "onlinebanking", "mobilebank", "ibanking",
}

# High-risk TLDs for banking phishing
SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",   # free Freenom TLDs
    ".xyz", ".top", ".club", ".site",
    ".online", ".live", ".click", ".link",
    ".info", ".biz",                      # commonly abused
}

# Safe TLDs for Indian banking
TRUSTED_TLDS = {".in", ".co.in", ".org.in", ".net.in", ".sbi", ".bank"}


# ══════════════════════════════════════════════════════════════
# SCAM MESSAGE PATTERNS
# ══════════════════════════════════════════════════════════════

SCAM_PATTERNS = [
    # OTP harvest
    (r'\botp\b.{0,40}\bexpire', 50,
     "OTP expiry urgency — classic credential harvest trigger"),
    (r'share.{0,20}\botp\b', 60,
     "Asks user to share OTP — never legitimate from any bank"),
    (r'do not share.{0,30}otp.{0,30}bank', 5,
     "Legitimate bank warning — safe"),   # negative signal

    # KYC fraud
    (r'kyc.{0,30}(expire|block|suspend|update|urgent|immediat)', 55,
     "KYC expiry/block urgency — common PSB phishing vector"),
    (r'account.{0,20}(block|suspend|freeze|deactivat)', 45,
     "Account block threat — urgency manipulation"),
    (r'verify.{0,20}(account|kyc|pan|aadhaar).{0,30}(link|click|tap)', 50,
     "Verify via link — credential phishing pattern"),

    # Lottery / prize
    (r'(congratulat|won|winner|prize|reward|lucky).{0,50}(lakh|crore|₹|\$|usd|inr)', 65,
     "Prize/lottery lure — financial scam"),
    (r'claim.{0,30}(reward|prize|cashback|refund).{0,30}(click|link|tap)', 60,
     "Claim reward via link — phishing"),

    # Fake refund / income tax
    (r'(income.?tax|it.?refund|tds.?refund).{0,40}(click|link|deposit|account)', 55,
     "Fake tax refund — government impersonation phishing"),
    (r'(refund|cashback).{0,30}(₹|\brs\b|\binr\b).{0,20}\d{3,}', 45,
     "Specific refund amount via message — suspicious"),

    # Remote access
    (r'(anydesk|teamviewer|quicksupport|remote.{0,10}(access|help|assist))', 70,
     "Remote access tool mention — account takeover setup"),
    (r'install.{0,30}(app|software|apk).{0,30}(bank|secure|verify|update)', 60,
     "Install app for bank — sideload malware vector"),

    # Urgency phrases
    (r'(within \d+ (hour|minute|day)|last chance|final (notice|warning)|immediately contact)', 30,
     "Urgency language — psychological manipulation"),

    # Fake customer care
    (r'(customer.?care|helpline|toll.?free).{0,30}(\+91|0\d{10}|\d{10})', 45,
     "Fake customer care number — vishing setup"),

    # SIM swap
    (r'(sim.{0,10}(swap|block|expire)|mobile.{0,15}(number|no).{0,15}(update|change|verify))', 50,
     "SIM swap social engineering pattern"),
]


# ══════════════════════════════════════════════════════════════
# FAKE WEBSITE DOM SIGNALS  (from JS SDK)
# ══════════════════════════════════════════════════════════════

# These are signals the frontend JS collects and POSTs to /threat/check-page
# Each maps to a suspicion contribution
PAGE_SIGNAL_WEIGHTS = {
    "form_action_external":    65,   # login form submits to external domain
    "iframe_over_form":        70,   # overlaid iframe on input fields
    "favicon_hash_mismatch":   40,   # favicon doesn't match legitimate bank
    "ssl_self_signed":         55,   # self-signed or invalid certificate
    "ssl_recently_issued":     25,   # cert issued < 7 days ago
    "no_ssl":                  80,   # plain HTTP for a banking page
    "domain_age_new":          35,   # domain registered < 30 days
    "typosquat_detected":      60,   # edit-distance ≤ 2 from PSB domain
    "idn_homoglyph":           70,   # unicode lookalike characters in domain
    "missing_security_headers":20,   # no HSTS / CSP headers
    "clipboard_hijack_attempt":75,   # JS attempting to read clipboard silently
    "keyboard_logger_detected":80,   # suspicious keydown listeners on page
    "redirect_chain_long":     40,   # >3 redirects before landing
    "brand_logo_without_domain":50,  # SBI/PNB logo on non-PSB domain
    "form_field_count_high":   20,   # >6 form fields (unusual for banking)
    "geolocation_requested":   30,   # page asks for location (not needed for banking)
    "unusual_port":            50,   # serving on port other than 80/443
}


# ══════════════════════════════════════════════════════════════
# URL ANALYSER
# ══════════════════════════════════════════════════════════════

class URLAnalyser:
    """
    Scores a URL for phishing / fake-banking-site risk.
    Returns score [0–100] and list of fired signals.
    """

    def analyse(self, url: str, referrer: str = "") -> dict:
        signals = []
        raw_pts = 0

        try:
            parsed = urlparse(url if "://" in url else "https://" + url)
        except Exception:
            return self._result(100, [{"signal": "Unparseable URL", "pts": 100}], url)

        host    = parsed.netloc.lower().lstrip("www.")
        path    = parsed.path.lower()
        scheme  = parsed.scheme.lower()
        tld     = self._tld(host)
        full    = url.lower()

        # ── 1. Legitimate domain check ─────────────────────
        if host in PSB_LEGITIMATE_DOMAINS:
            # Check scheme — HTTP on a known bank domain is still wrong
            if scheme == "http":
                signals.append({"signal": f"Known bank domain but HTTP (no SSL): {host}", "pts": 60})
                raw_pts += 60
            else:
                return self._result(0, [{"signal": f"Verified legitimate PSB domain: {host}", "pts": 0}], url)

        # ── 2. No SSL ──────────────────────────────────────
        if scheme == "http":
            signals.append({"signal": "No SSL (HTTP) — banking page requires HTTPS", "pts": 80})
            raw_pts += 80

        # ── 3. IP address instead of domain ───────────────
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', host):
            signals.append({"signal": "IP address URL — no legitimate bank uses bare IPs", "pts": 85})
            raw_pts += 85

        # ── 4. Suspicious TLD ─────────────────────────────
        if tld in SUSPICIOUS_TLDS:
            signals.append({"signal": f"High-risk TLD: {tld} — commonly used in phishing", "pts": 45})
            raw_pts += 45

        # ── 5. Brand token in wrong domain ─────────────────
        host_parts = re.split(r'[\.\-]', host)
        matched_brands = [t for t in PSB_BRAND_TOKENS if t in host_parts]
        if matched_brands and tld not in TRUSTED_TLDS:
            pts = 55
            signals.append({
                "signal": f"PSB brand token '{matched_brands[0]}' in untrusted domain: {host}",
                "pts": pts
            })
            raw_pts += pts

        # ── 6. Typosquat detection ─────────────────────────
        for legit in PSB_LEGITIMATE_DOMAINS:
            legit_host = legit.lstrip("www.")
            dist = self._edit_distance(host, legit_host)
            if 0 < dist <= 2 and len(host) > 5:
                pts = 65
                signals.append({
                    "signal": f"Typosquat: '{host}' is edit-distance {dist} from '{legit_host}'",
                    "pts": pts
                })
                raw_pts += pts
                break

        # ── 7. IDN / homoglyph detection ──────────────────
        try:
            decoded = host.encode('ascii').decode('ascii')
        except UnicodeDecodeError:
            decoded = None
        if decoded is None or 'xn--' in host:
            signals.append({
                "signal": f"IDN / homoglyph domain detected: {host} — visual spoofing risk",
                "pts": 70
            })
            raw_pts += 70

        # ── 8. Subdomain depth ────────────────────────────
        subdomain_count = host.count('.')
        if subdomain_count >= 4:
            pts = 30
            signals.append({
                "signal": f"Deep subdomain ({subdomain_count} levels) — obfuscation tactic",
                "pts": pts
            })
            raw_pts += pts

        # ── 9. Suspicious path keywords ───────────────────
        phish_path_tokens = ["login", "verify", "secure", "update",
                              "confirm", "account", "kyc", "otp", "validate"]
        path_hits = [t for t in phish_path_tokens if t in path]
        if len(path_hits) >= 2 and host not in PSB_LEGITIMATE_DOMAINS:
            pts = 35
            signals.append({
                "signal": f"Multiple phishing path keywords: {path_hits}",
                "pts": pts
            })
            raw_pts += pts

        # ── 10. Unusually long URL ─────────────────────────
        if len(url) > 150:
            pts = 20
            signals.append({
                "signal": f"Unusually long URL ({len(url)} chars) — obfuscation risk",
                "pts": pts
            })
            raw_pts += pts

        # ── 11. URL-encoded characters ─────────────────────
        encoded_count = url.count('%')
        if encoded_count > 5:
            pts = 25
            signals.append({
                "signal": f"{encoded_count} URL-encoded characters — obfuscation",
                "pts": pts
            })
            raw_pts += pts

        # ── 12. Referrer mismatch ──────────────────────────
        if referrer:
            ref_host = urlparse(referrer).netloc.lower().lstrip("www.")
            if ref_host and ref_host != host and ref_host not in PSB_LEGITIMATE_DOMAINS:
                if any(brand in ref_host for brand in PSB_BRAND_TOKENS):
                    pts = 40
                    signals.append({
                        "signal": f"Referred from suspicious PSB-branded domain: {ref_host}",
                        "pts": pts
                    })
                    raw_pts += pts

        return self._result(raw_pts, signals, url)

    def _tld(self, host: str) -> str:
        parts = host.split(".")
        if len(parts) >= 3 and parts[-2] in ("co", "net", "org", "gov"):
            return "." + ".".join(parts[-2:])
        return "." + parts[-1] if parts else ""

    def _edit_distance(self, a: str, b: str) -> int:
        """Levenshtein distance — fast enough for short domain strings."""
        if abs(len(a) - len(b)) > 3:
            return 99
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]

    def _result(self, raw_pts: int, signals: list, url: str) -> dict:
        # Cap at 100 via sigmoid-lite
        score = min(100, int(raw_pts * 0.9)) if raw_pts > 0 else 0
        if score >= 70:   risk = "CRITICAL"
        elif score >= 50: risk = "HIGH"
        elif score >= 25: risk = "MODERATE"
        else:             risk = "LOW"
        return {
            "url":         url,
            "score":       score,
            "risk_label":  risk,
            "signals":     signals,
            "suspicion_pts_for_session": _url_score_to_session_pts(score),
            "bank_response": _bank_response(risk, "url"),
            "timestamp":   datetime.now().isoformat(),
        }


def _url_score_to_session_pts(score: int) -> int:
    """Map URL threat score → b_raw contribution in session scorer."""
    if score >= 70: return 45
    if score >= 50: return 30
    if score >= 25: return 15
    return 0


# ══════════════════════════════════════════════════════════════
# SCAM MESSAGE ANALYSER
# ══════════════════════════════════════════════════════════════

class ScamMessageAnalyser:
    """
    Detects phishing SMS, scam WhatsApp messages, fraudulent emails.
    Returns score [0–100] with matched patterns.
    """

    def analyse(self, text: str, channel: str = "sms") -> dict:
        signals   = []
        raw_pts   = 0
        text_low  = text.lower()

        for pattern, pts, label in SCAM_PATTERNS:
            if re.search(pattern, text_low, re.IGNORECASE):
                if pts > 0:
                    signals.append({"signal": label, "pts": pts, "pattern": pattern})
                    raw_pts += pts

        # URL extraction — scan embedded links
        urls_found = re.findall(
            r'https?://[^\s<>"]+|www\.[^\s<>"]+|[a-z0-9\-]+\.[a-z]{2,}(?:/[^\s]*)?',
            text_low
        )
        url_analyser = URLAnalyser()
        for u in urls_found[:5]:  # analyse up to 5 embedded URLs
            ur = url_analyser.analyse(u)
            if ur["score"] >= 25:
                signals.append({
                    "signal": f"Embedded suspicious URL: {u} [{ur['risk_label']}]",
                    "pts":    ur["score"] // 2   # partial credit
                })
                raw_pts += ur["score"] // 2

        # Channel-specific boosts
        if channel == "sms" and re.search(r'(bank|sbi|pnb|hdfc|icici)', text_low):
            if raw_pts > 20:
                raw_pts += 10   # SMS claiming to be from bank and already suspicious

        score = min(100, raw_pts)
        if score >= 70:   risk = "CRITICAL"
        elif score >= 50: risk = "HIGH"
        elif score >= 25: risk = "MODERATE"
        else:             risk = "LOW"

        return {
            "text_preview": text[:120] + ("..." if len(text) > 120 else ""),
            "channel":      channel,
            "score":        score,
            "risk_label":   risk,
            "signals":      signals,
            "urls_found":   urls_found[:5],
            "suspicion_pts_for_session": _msg_score_to_session_pts(score),
            "bank_response": _bank_response(risk, "message"),
            "advice":       _user_advice(risk, "message"),
            "timestamp":    datetime.now().isoformat(),
        }


def _msg_score_to_session_pts(score: int) -> int:
    if score >= 70: return 40
    if score >= 50: return 25
    if score >= 25: return 12
    return 0


# ══════════════════════════════════════════════════════════════
# FAKE PAGE SIGNAL ANALYSER  (from JS SDK)
# ══════════════════════════════════════════════════════════════

class FakePageAnalyser:
    """
    Receives DOM/TLS/network signals from the frontend JS SDK
    and scores the current page for fake-site risk.
    These signals are collected passively — zero user friction.
    """

    def analyse(self, signals: dict) -> dict:
        fired   = []
        raw_pts = 0

        for signal_name, pts in PAGE_SIGNAL_WEIGHTS.items():
            if signals.get(signal_name):
                fired.append({"signal": signal_name, "pts": pts,
                              "description": _page_signal_desc(signal_name)})
                raw_pts += pts

        # Compound: form hijacking + no SSL = near-certain phishing page
        if signals.get("form_action_external") and signals.get("no_ssl"):
            fired.append({
                "signal": "COMPOUND: Form hijack + No SSL — near-certain credential theft page",
                "pts": 30
            })
            raw_pts += 30

        # Compound: keyboard logger + iframe overlay
        if signals.get("keyboard_logger_detected") and signals.get("iframe_over_form"):
            fired.append({
                "signal": "COMPOUND: Keyboard logger + iframe overlay — active credential skimming",
                "pts": 40
            })
            raw_pts += 40

        score = min(100, raw_pts)
        if score >= 70:   risk = "CRITICAL"
        elif score >= 50: risk = "HIGH"
        elif score >= 25: risk = "MODERATE"
        else:             risk = "LOW"

        return {
            "score":       score,
            "risk_label":  risk,
            "signals":     fired,
            "suspicion_pts_for_session": _page_score_to_session_pts(score),
            "bank_response": _bank_response(risk, "page"),
            "advice":       _user_advice(risk, "page"),
            "timestamp":    datetime.now().isoformat(),
        }


def _page_score_to_session_pts(score: int) -> int:
    if score >= 70: return 50   # Near-certain fake site — high weight
    if score >= 50: return 35
    if score >= 25: return 20
    return 0


def _page_signal_desc(name: str) -> str:
    descs = {
        "form_action_external":    "Login form submits data to a domain different from the page — credential theft",
        "iframe_over_form":        "Invisible iframe overlaid on form fields — captures keystrokes",
        "favicon_hash_mismatch":   "Favicon does not match the real bank's — visual spoofing",
        "ssl_self_signed":         "SSL certificate is self-signed, not from a trusted CA",
        "ssl_recently_issued":     "SSL cert issued < 7 days ago — disposable phishing cert",
        "no_ssl":                  "Page served over HTTP — all data submitted in plaintext",
        "domain_age_new":          "Domain registered < 30 days ago — throwaway phishing domain",
        "typosquat_detected":      "Domain name closely resembles a real PSB domain",
        "idn_homoglyph":           "Unicode lookalike characters in domain — visual spoofing",
        "missing_security_headers":"No HSTS/CSP headers — real bank sites always have these",
        "clipboard_hijack_attempt":"JavaScript silently reading clipboard — credential theft attempt",
        "keyboard_logger_detected":"Suspicious keydown event listeners — possible keylogger",
        "redirect_chain_long":     "More than 3 redirects before landing — obfuscation",
        "brand_logo_without_domain":"Bank logo present but domain is not the bank's real domain",
        "form_field_count_high":   "Unusually many form fields — harvesting extra PII",
        "geolocation_requested":   "Page requests GPS location — not needed for banking",
        "unusual_port":            "Served on non-standard port — real banks use 443 only",
    }
    return descs.get(name, name)


# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════

def _bank_response(risk: str, context: str) -> str:
    """RBI-compliant response language — bank frames the action."""
    if risk == "CRITICAL":
        return (f"Bank security alert: This {context} has been flagged as a likely phishing "
                f"attempt. Do not proceed. Your bank will never ask for OTP, password, or "
                f"card details via SMS/link. If in doubt, call your branch directly.")
    if risk == "HIGH":
        return (f"Bank caution: Suspicious {context} detected. Verify the sender/URL "
                f"independently before entering any credentials.")
    if risk == "MODERATE":
        return (f"Bank notice: Proceed with caution. Some signals on this {context} are "
                f"inconsistent with normal banking channels.")
    return "No threat signals detected."


def _user_advice(risk: str, context: str) -> str:
    advice_map = {
        ("CRITICAL", "message"): (
            "🚨 Do NOT click any link in this message. Do NOT share OTP or PIN. "
            "Your bank will never send unsolicited messages asking for credentials. "
            "Forward to 1930 (National Cybercrime Helpline) if received."
        ),
        ("HIGH", "message"): (
            "⚠️ This message contains patterns common in banking scams. "
            "Verify by calling your bank's official number before taking any action."
        ),
        ("CRITICAL", "page"): (
            "🚨 This page is likely a fake banking site. Close immediately. "
            "Do not enter any username, password, or OTP. "
            "Report to your bank and www.cybercrime.gov.in."
        ),
        ("HIGH", "page"): (
            "⚠️ This page has multiple suspicious signals. "
            "Verify the URL carefully — your bank's URL should match exactly."
        ),
    }
    return advice_map.get((risk, context), "Stay alert and verify through official channels.")


# ══════════════════════════════════════════════════════════════
# COMPOSITE THREAT CHECK
# ══════════════════════════════════════════════════════════════

def comprehensive_check(
    url: Optional[str] = None,
    message: Optional[str] = None,
    message_channel: str = "sms",
    page_signals: Optional[dict] = None,
) -> dict:
    """
    Run all applicable checks and return combined result.
    Called by main.py to get a unified threat assessment.
    """
    results = {}
    max_score = 0
    total_session_pts = 0

    if url:
        r = URLAnalyser().analyse(url)
        results["url"] = r
        max_score = max(max_score, r["score"])
        total_session_pts += r["suspicion_pts_for_session"]

    if message:
        r = ScamMessageAnalyser().analyse(message, message_channel)
        results["message"] = r
        max_score = max(max_score, r["score"])
        total_session_pts += r["suspicion_pts_for_session"]

    if page_signals:
        r = FakePageAnalyser().analyse(page_signals)
        results["page"] = r
        max_score = max(max_score, r["score"])
        total_session_pts += r["suspicion_pts_for_session"]

    if max_score >= 70:   overall_risk = "CRITICAL"
    elif max_score >= 50: overall_risk = "HIGH"
    elif max_score >= 25: overall_risk = "MODERATE"
    else:                 overall_risk = "LOW"

    return {
        "overall_score":         max_score,
        "overall_risk":          overall_risk,
        "session_suspicion_pts": min(80, total_session_pts),  # capped — single layer
        "results":               results,
        "bank_response":         _bank_response(overall_risk, "session"),
        "block_session":         max_score >= 70,  # suggest blocking login if critical
        "timestamp":             datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════
# STANDALONE DEMO
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("BehaviorShield ThreatShield — Demo")
    print("=" * 62)

    # ── URL tests ──────────────────────────────────────────
    print("\n▶ URL ANALYSIS")
    test_urls = [
        ("https://onlinesbi.sbi/",                         "Legitimate SBI"),
        ("http://sbi-netbanking-update.xyz/login/verify",  "Phishing URL"),
        ("https://onlinesb1.com/secure/login",             "Typosquat"),
        ("http://192.168.1.1/banking/login",               "IP address URL"),
        ("https://secure-sbi-kyc-update.in/verify/otp",   "PSB brand abuse"),
    ]
    ua = URLAnalyser()
    for url, label in test_urls:
        r = ua.analyse(url)
        print(f"\n  [{label}]")
        print(f"  URL   : {url[:70]}")
        print(f"  Score : {r['score']}/100  [{r['risk_label']}]")
        print(f"  → Session pts: +{r['suspicion_pts_for_session']}")
        for s in r["signals"][:3]:
            print(f"    • {s['signal']}")

    # ── SMS tests ──────────────────────────────────────────
    print("\n▶ SMS / MESSAGE ANALYSIS")
    test_msgs = [
        ("Your SBI OTP is 482910. Valid for 10 mins. Do not share.", "sms",   "Legitimate OTP SMS"),
        ("Dear customer your SBI account will be blocked. Update KYC immediately: http://sbi-kyc.xyz/verify", "sms", "Phishing SMS"),
        ("Congratulations! You have won ₹50 lakh. Click to claim: bit.ly/xyz123", "whatsapp", "Lottery scam"),
        ("Install AnyDesk app to get your SBI refund processed by bank executive.", "sms", "Remote access scam"),
    ]
    ma = ScamMessageAnalyser()
    for text, channel, label in test_msgs:
        r = ma.analyse(text, channel)
        print(f"\n  [{label}]")
        print(f"  Text  : {text[:80]}...")
        print(f"  Score : {r['score']}/100  [{r['risk_label']}]")
        print(f"  → Session pts: +{r['suspicion_pts_for_session']}")
        for s in r["signals"][:2]:
            print(f"    • {s['signal']}")

    # ── Page signals test ──────────────────────────────────
    print("\n▶ FAKE PAGE DETECTION")
    fake_page = {
        "form_action_external": True,
        "no_ssl": True,
        "typosquat_detected": True,
        "brand_logo_without_domain": True,
        "ssl_recently_issued": True,
    }
    rp = FakePageAnalyser().analyse(fake_page)
    print(f"  Score : {rp['score']}/100  [{rp['risk_label']}]")
    print(f"  → Session pts: +{rp['suspicion_pts_for_session']}")
    for s in rp["signals"]:
        print(f"    • {s['signal']} (+{s['pts']} pts)")

    print("\n" + "=" * 62)
