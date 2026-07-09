#!/usr/bin/env python3
"""
Surgically applies the device-toggle + reset fix to an EXISTING,
already-URL-fixed index.html — does NOT overwrite the whole file, so it
can't undo the earlier localhost:8000 -> relative-path deployment fix.

Usage: python3 apply_device_fix.py backend/frontend/index.html
"""
import sys, re

path = sys.argv[1] if len(sys.argv) > 1 else "backend/frontend/index.html"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

original = content

# ── Fix 1: toggleDeviceMode's broken re-fire (window._bs never assigned) ──
old_toggle_block = """  // Re-fire device check so the session immediately reflects the new mode
  if (window._bs) {
    window._bs.trigger({ type: 'device_check', device_fingerprint: _getFingerprint() });
  }
}

function _getFingerprint() {
  return [
    navigator.userAgent.match(/(Chrome|Firefox|Safari|Edge)/)?.[1] || 'Browser',
    navigator.platform || 'Unknown',
    `${screen.width}x${screen.height}`,
    Intl.DateTimeFormat().resolvedOptions().timeZone
  ].join('-');
}"""

new_toggle_block = """  // Re-fire device check so the session immediately reflects the new mode.
  // Previously checked `window._bs`, which is never assigned anywhere in
  // this file (the actual instance is `shield`) -- so this never actually
  // ran, and the on-screen device badge only updated later, by coincidence,
  // whenever some unrelated event happened to trigger a fresh device_check
  // (e.g. a profile switch). That's why toggling could feel like it
  // "doesn't switch off in one go" / gets stuck on New Device.
  // Also uses shield._fingerprint() (behaviorsignal.js's method, which has
  // the IANA timezone-alias normalisation fix) instead of the separate,
  // unfixed _getFingerprint() duplicate that used to live here -- one
  // fingerprint implementation now, not two that can drift apart.
  if (shield) {
    shield.trigger({ type: 'device_check', device_fingerprint: shield._fingerprint() });
  }
}"""

if old_toggle_block in content:
    content = content.replace(old_toggle_block, new_toggle_block)
    print("[OK] toggleDeviceMode re-fire block replaced")
else:
    print("[SKIP] toggleDeviceMode old block not found verbatim -- check manually (see diff below)")

# ── Fix 2: resetSession firing an unhandled '__reset__' event ──
old_reset_pattern = re.compile(
    r"function resetSession\(\)\{.*?"
    r"if\(shield\) shield\.trigger\(\{type:'__reset__'\}\);\s*"
    r"fetch\('[^']*?/reset/'\+curProfile,\{method:'POST'\}\)\.catch\(\(\)=>\{\}\);",
    re.DOTALL
)

new_reset_head = """async function resetSession(){
  bRaw=0; tRaw=0; deviceState=null; backendUp=false; window._lastState=null;
  document.getElementById('devBadge').className='device-badge';
  document.getElementById('txWarn').classList.remove('show');
  document.getElementById('phrase').value='';
  document.getElementById('kbaWrap').classList.remove('show');
  document.getElementById('callVerifyWrap').classList.remove('show');
  document.getElementById('kbaAnswer').value='';
  document.getElementById('callAnswer').value='';
  curKbaQuestion=null; curCallQuestion=null;
  checkHardening(0,false);
  checkCallBanner(false);
  setDemoTier(0); // clear any active tier lock

  // Perform the REAL backend-side reset first (awaited), THEN re-fire a
  // fresh device_check so the UI immediately reflects the true
  // post-reset state. Previously this fired '__reset__' over the
  // websocket -- an event type the backend doesn't actually handle (a
  // silent no-op) -- before the real reset POST had even resolved, and
  // nothing ever followed up afterward. The UI was left waiting on
  // whatever ambient event happened to arrive next, which is why
  // device state (and KBA/probation/frozen badges) could appear to
  // "stick" on the pre-reset value for a while after clicking Reset.
  try {
    await fetch('/reset/'+curProfile, {method:'POST'});
  } catch(e) { /* backend offline -- nothing further to sync */ }
  if(shield) shield.trigger({type:'device_check', device_fingerprint: shield._fingerprint()});"""

match = old_reset_pattern.search(content)
if match:
    content = content[:match.start()] + new_reset_head + content[match.end():]
    print("[OK] resetSession function replaced")
else:
    print("[SKIP] resetSession old pattern not found -- check manually")

if content == original:
    print("\n[WARNING] No changes were made. File may already be fixed, or"
          " the old code doesn't match exactly (whitespace/quote differences).")
    sys.exit(1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nDone. Wrote changes to {path}")
