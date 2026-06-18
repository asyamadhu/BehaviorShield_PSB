/* =============================================================
   behaviorsignal.js — BehaviorShield JS SDK
   Captures all signals from Section 3.1 (4-Zone Login Canvas)
   and sends them to the backend via WebSocket.

   Zone 1 — Swipe / touch gesture (velocity, curvature)
   Zone 2 — Cognitive challenge response time
   Zone 3 — Keystroke dynamics (dwell, flight, paste)
   Zone 4 — Post-submit idle micro-movement

   NEW: Device fingerprint sent on connect.
        Backend DeviceTrustEngine checks it against trusted list.
   ============================================================= */

class BehaviorShield {
  constructor(wsUrl) {
    this.wsUrl         = wsUrl;
    this.ws            = null;
    this.keydownTimes  = {};
    this.lastKeyup     = null;
    this.keyCount      = 0;
    this.sessionStart  = performance.now();
    this.mouseX        = 0;
    this.mouseY        = 0;
    this.mouseHistory  = [];
    this.touchStart    = null;
    this.onScore       = null;   // callback: set from outside

    this._connect();
    this._listen();
    this._idleMonitor();
  }

  // ── WebSocket ──────────────────────────────────────────────
  _connect() {
    this.ws = new WebSocket(this.wsUrl);

    this.ws.onopen = () => {
      console.log('[BS] Connected');
      // Send device fingerprint immediately — backend checks trust list
      this._send({
        type:               'device_check',
        device_fingerprint: this._fingerprint(),
        screen:             `${screen.width}x${screen.height}`,
        timezone:           Intl.DateTimeFormat().resolvedOptions().timeZone,
        language:           navigator.language,
        platform:           navigator.platform,
      });
    };

    this.ws.onmessage = (e) => {
      try {
        const st = JSON.parse(e.data);
        if (this.onScore) this.onScore(st);
      } catch (_) {}
    };

    this.ws.onerror = () => {};
    this.ws.onclose = () => {
      setTimeout(() => this._connect(), 2000);
    };
  }

  // ── Device fingerprint ─────────────────────────────────────
  // Format: Browser-OS-ScreenResolution-Timezone
  // Matches the fingerprints in profiles.py trusted_devices.
  _fingerprint() {
    const ua  = navigator.userAgent;
    const br  = ua.includes('Chrome') ? 'Chrome'
              : ua.includes('Firefox') ? 'Firefox'
              : ua.includes('Safari')  ? 'Safari' : 'Browser';
    const os  = ua.includes('Windows NT 10') ? 'Win11'
              : ua.includes('Windows')        ? 'Win'
              : ua.includes('Mac')            ? 'Mac'
              : ua.includes('Android')        ? 'Android'
              : ua.includes('iPhone')         ? 'iOS'
              : ua.includes('Linux')          ? 'Linux' : 'OS';
    const sc  = `${screen.width}x${screen.height}`;
    const tz  = Intl.DateTimeFormat().resolvedOptions().timeZone;
    return `${br}-${os}-${sc}-${tz}`;
  }

  // ── Attach all listeners ───────────────────────────────────
  _listen() {
    document.addEventListener('keydown',    e => this._kdown(e));
    document.addEventListener('keyup',      e => this._kup(e));
    document.addEventListener('paste',      e => this._paste(e));
    document.addEventListener('mousemove',  e => this._mmove(e));
    document.addEventListener('touchstart', e => this._tstart(e), {passive:true});
    document.addEventListener('touchend',   e => this._tend(e),   {passive:true});
  }

  // ── Zone 3: Keystroke dynamics ─────────────────────────────
  _kdown(e) { this.keydownTimes[e.code] = performance.now(); }

  _kup(e) {
    const now   = performance.now();
    const dwell = now - (this.keydownTimes[e.code] || now);
    const flight= this.lastKeyup ? now - this.lastKeyup : null;
    this.lastKeyup = now;
    this.keyCount++;

    this._send({ type: 'keystroke', dwell_ms: Math.round(dwell),
      flight_ms: flight ? Math.round(flight) : null,
      field: document.activeElement?.id || 'unknown' });

    if (this.keyCount % 10 === 0) {
      const min = (now - this.sessionStart) / 60000;
      const wpm = min > 0 ? Math.round((this.keyCount / 5) / min) : 0;
      this._send({ type: 'typing_speed', wpm });
    }
  }

  // ── Zone 3: Paste detection ────────────────────────────────
  _paste(e) {
    const el = document.activeElement;
    const isPass = el?.type === 'password'
                || (el?.id || '').toLowerCase().includes('pass');
    this._send({ type: isPass ? 'paste_password' : 'paste',
                 field: el?.id || 'unknown' });
  }

  // ── Mouse jitter (RAT detection) ───────────────────────────
  // Sends a jitter event ONLY when movement variance is extremely low
  // (near-perfectly straight / robotic) sustained across multiple windows.
  // A human moving a mouse naturally has high directional variance — curves,
  // overshoots, micro-corrections. A RAT/script moves in straight lines.
  // Threshold 0.3 (was 0.5) catches only truly robotic movement.
  // Also requires 3 consecutive low-jitter windows before sending to
  // avoid false positives from brief straight movements.
  _mmove(e) {
    const dx = e.clientX - this.mouseX;
    const dy = e.clientY - this.mouseY;
    this.mouseX = e.clientX;
    this.mouseY = e.clientY;
    this.mouseHistory.push({ dx, dy });
    if (this.mouseHistory.length > 20) this.mouseHistory.shift();
    if (this.mouseHistory.length === 20) {
      const j = this._jitter();
      if (j < 0.3) {
        this._lowJitterCount = (this._lowJitterCount || 0) + 1;
        // Only flag after 3 consecutive low-jitter windows (~60 mouse events)
        if (this._lowJitterCount === 3) {
          this._send({ type: 'mouse_jitter', jitter_score: j });
        }
        // After initial flag, fire again only every 10 more windows
        else if (this._lowJitterCount > 3 && (this._lowJitterCount - 3) % 10 === 0) {
          this._send({ type: 'mouse_jitter', jitter_score: j });
        }
      } else {
        this._lowJitterCount = 0; // reset on any normal curved movement
      }
    }
  }

  _jitter() {
    const dirs = this.mouseHistory.map(p => Math.atan2(p.dy, p.dx));
    const mean = dirs.reduce((a, b) => a + b, 0) / dirs.length;
    const v    = dirs.reduce((a, b) => a + (b - mean) ** 2, 0) / dirs.length;
    return Math.sqrt(v);
  }

  // ── Zone 1: Swipe gesture ──────────────────────────────────
  _tstart(e) {
    const t = e.touches[0];
    this.touchStart = { x: t.clientX, y: t.clientY, t: performance.now() };
  }

  _tend(e) {
    if (!this.touchStart) return;
    const t   = e.changedTouches[0];
    const dx  = t.clientX - this.touchStart.x;
    const dy  = t.clientY - this.touchStart.y;
    const dur = performance.now() - this.touchStart.t;
    const dst = Math.sqrt(dx*dx + dy*dy);
    this._send({ type: 'swipe', velocity: dst/dur, distance: Math.round(dst),
                 angle: Math.round(Math.atan2(dy,dx)*180/Math.PI),
                 duration_ms: Math.round(dur) });
    this.touchStart = null;
  }

  // ── Zone 4: Post-submit idle ───────────────────────────────
  _idleMonitor() {
    let px = 0, py = 0;
    setInterval(() => {
      const mv = Math.abs(this.mouseX - px) + Math.abs(this.mouseY - py);
      this._send({ type: 'mouse_idle', micro_movement_px: Math.round(mv) });
      px = this.mouseX; py = this.mouseY;
    }, 3000);
  }

  // ── Public: fire a custom event (demo controller) ──────────
  trigger(eventObj) { this._send(eventObj); }

  _send(data) {
    if (this.ws?.readyState === WebSocket.OPEN)
      this.ws.send(JSON.stringify({ ts: Date.now(), ...data }));
  }

  disconnect() { this.ws?.close(); }
}


// ══════════════════════════════════════════════════════════════
// THREATSHIELD PAGE SCANNER
// Automatically collects DOM/TLS/network integrity signals
// and reports them to the ThreatShield backend on page load.
// Runs silently — zero user friction.
// ══════════════════════════════════════════════════════════════

class ThreatShieldScanner {
  constructor(apiBase = 'http://localhost:8000', profile = null) {
    this.api     = apiBase;
    this.profile = profile;
    this.result  = null;
  }

  /** Collect all page integrity signals passively */
  collectSignals() {
    const signals = {};

    // 1. SSL / HTTPS check
    signals.no_ssl = location.protocol === 'http:';

    // 2. Form action external domain check
    const forms = document.querySelectorAll('form');
    signals.form_action_external = Array.from(forms).some(f => {
      const action = f.getAttribute('action') || '';
      if (!action.startsWith('http')) return false;
      try {
        return new URL(action).hostname !== location.hostname;
      } catch { return false; }
    });

    // 3. Iframe overlay detection (iframes on page with inputs)
    const iframes = document.querySelectorAll('iframe');
    signals.iframe_over_form = iframes.length > 0 && forms.length > 0;

    // 4. IDN / homoglyph in current domain
    signals.idn_homoglyph = /xn--/.test(location.hostname) ||
      [...location.hostname].some(c => c.charCodeAt(0) > 127);

    // 5. Unusual port
    const p = location.port;
    signals.unusual_port = !!(p && !['80','443','8000','3000','5500','5173',''].includes(p));

    // 6. Redirect chain depth
    try {
      const nav = performance.getEntriesByType('navigation')[0];
      signals.redirect_chain_long = (nav?.redirectCount || 0) > 3;
    } catch { signals.redirect_chain_long = false; }

    // 7. Excessive form fields (PII harvest indicator)
    signals.form_field_count_high = document.querySelectorAll('input').length > 8;

    // 8. Suspicious inline scripts referencing keylogger patterns
    const scripts = Array.from(document.querySelectorAll('script'))
      .map(s => s.textContent || '');
    signals.keyboard_logger_detected = scripts.some(s =>
      /document\.addEventListener\s*\(\s*['"]keydown/i.test(s) &&
      /fetch|XMLHttpRequest|sendBeacon/i.test(s)
    );

    // 9. Clipboard hijack attempt
    signals.clipboard_hijack_attempt = scripts.some(s =>
      /navigator\.clipboard|clipboardData/i.test(s) &&
      !/paste/i.test(s)   // reading without paste event = suspicious
    );

    // 10. Geolocation request
    signals.geolocation_requested = scripts.some(s =>
      /navigator\.geolocation\.getCurrentPosition/i.test(s)
    );

    // 11. External resources claiming bank brand
    const psbBrands = ['sbi','pnb','hdfc','icici','canara','union','centralbank'];
    const imgs = Array.from(document.querySelectorAll('img'));
    signals.brand_logo_without_domain = imgs.some(img => {
      const src = (img.src || '').toLowerCase();
      const alt = (img.alt || '').toLowerCase();
      const hasBrand = psbBrands.some(b => src.includes(b) || alt.includes(b));
      if (!hasBrand) return false;
      try {
        return new URL(img.src).hostname !== location.hostname;
      } catch { return false; }
    });

    // 12. Missing security headers (best-effort from meta tags)
    const metas = document.querySelectorAll('meta[http-equiv]');
    const headerNames = Array.from(metas).map(m =>
      (m.getAttribute('http-equiv') || '').toLowerCase()
    );
    signals.missing_security_headers = !headerNames.some(h =>
      h.includes('content-security-policy') || h.includes('strict-transport')
    );

    return signals;
  }

  /** Run scan and POST to backend. Returns result promise. */
  async scan() {
    const signals = this.collectSignals();
    try {
      const resp = await fetch(`${this.api}/threat/check-page`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ signals, profile: this.profile }),
      });
      this.result = await resp.json();
    } catch {
      // Backend unavailable — silent fallback, don't break page
      const flagCount = Object.values(signals).filter(Boolean).length;
      this.result = {
        score:      flagCount * 8,
        risk_label: flagCount === 0 ? 'LOW' : flagCount < 3 ? 'MODERATE' : 'HIGH',
        signals:    [],
        local_scan: true,
      };
    }

    // If CRITICAL, show a visible bank-framed warning banner
    if (this.result?.risk_label === 'CRITICAL') {
      this._showWarningBanner(this.result);
    }

    return this.result;
  }

  /** Render a non-intrusive warning banner (RBI-compliant language) */
  _showWarningBanner(result) {
    if (document.getElementById('ts-warning-banner')) return;
    const banner = document.createElement('div');
    banner.id = 'ts-warning-banner';
    banner.style.cssText = [
      'position:fixed', 'top:0', 'left:0', 'right:0', 'z-index:99999',
      'background:#7f1d1d', 'color:#fef2f2', 'padding:12px 20px',
      'font-size:13px', 'font-weight:600', 'display:flex',
      'align-items:center', 'justify-content:space-between',
      'border-bottom:2px solid #dc2626', 'box-shadow:0 4px 20px rgba(0,0,0,0.5)',
    ].join(';');
    banner.innerHTML = `
      <span>
        🚨 <strong>Bank Security Alert:</strong>
        ${result.bank_response ||
          'This page has been flagged as potentially unsafe. ' +
          'Do not enter any credentials. Verify the URL carefully.'}
      </span>
      <button onclick="document.getElementById('ts-warning-banner').remove()"
              style="background:none;border:1px solid #fca5a5;color:#fca5a5;
                     padding:4px 12px;border-radius:5px;cursor:pointer;
                     font-size:11px;margin-left:16px;flex-shrink:0">
        Dismiss
      </button>`;
    document.body.insertBefore(banner, document.body.firstChild);
  }
}

// Auto-run on page load (non-blocking)
if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    // Skip auto page scan on localhost / local dev — http:// and missing
    // security headers are always present in dev environments and produce
    // a false CRITICAL score on every page load. The scanner remains
    // fully functional when called manually (e.g. runTsCheck() in index.html).
    const host = window.location && window.location.hostname;
    const isLocalDev = ['localhost','127.0.0.1','0.0.0.0',''].includes(host)
                    || (host && (host.startsWith('192.168.') || host.startsWith('10.')));
    if (isLocalDev) return;

    const scanner = new ThreatShieldScanner(
      window.BS_API_BASE || 'http://localhost:8000',
      window.BS_PROFILE  || null
    );
    scanner.scan().then(result => {
      window._threatShieldResult = result;
      window.dispatchEvent(new CustomEvent('threatShieldResult', { detail: result }));
    });
  });
}
