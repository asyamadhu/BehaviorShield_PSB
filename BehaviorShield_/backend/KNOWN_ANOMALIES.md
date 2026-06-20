# BehaviorShield — Known Anomalies & Open Items

Discovered while implementing the Tier-3 gate fix and building
`test_scorer.py`. None of these are fixed by that change; they are
documented here per the "flag anomalies even if out of scope" request.

---

## #1 — `tx_fraud_scorer` ML fraud_prob is non-monotonic in `amount_ratio`

**Where:** `tx_fraud_scorer.get_tx_fraud_prob()` / `_build_feature_vector()`,
called from `scorer._eval_transaction()`.

**Symptom:** For a new-beneficiary transfer with zero behavioural signal,
sweeping the transfer amount from ₹100 to ₹10,00,000 produces a
**non-monotonic** `combined_score`:

```
amt=  8,000  ratio=1.000  combined=28.40  raw_tier=0
amt= 16,000  ratio=2.000  combined=18.10  raw_tier=0   <- DECREASE
```

**Root cause:** `_build_feature_vector()` sets:

```python
vec[0] = win1_mean ≈ min(1.0, (amount_ratio - 1) / 5 + bene_risk * 0.4 + is_new_bene * 0.3)
```

As `amount_ratio` rises from 1.0 to 2.0, `win1_mean` rises from ~0.54 to
~0.64 to ~0.74. Over this range, the GBM+RF ensemble's `fraud_prob`
output actually **decreases** (37% → 35% → 33%), which drops
`score_to_points()` from +15 to +8, reducing `t_raw` and thus
`combined_score` — even though the transaction amount (the thing a human
would consider "more anomalous") increased.

**Why this is plausible:** The ensemble is trained on the PSB dataset's
109 engineered features (mean/std/min/max/q75/nnz per time-window), and
`win1_mean` is just one of those 109 inputs. A GBM/RF decision boundary
is not guaranteed to be monotonic in any single input feature, especially
one that's a synthetic blend (`amount_ratio`, `bene_risk`, `is_new_bene`
folded into one number) rather than something the model was trained to
treat as ordinally meaningful on its own.

**Impact:** Bounded — `_combine()` itself (this fix's target) IS
monotonic in `t_raw` (verified by `test_combine_monotonic_in_t_raw_gate_active`).
The non-monotonicity is confined to the `t_raw` *input* in a narrow
ratio band (~1x-3x baseline) and doesn't change tier outcomes in the
cases tested (both ₹8,000 and ₹16,000 land in raw_tier=0 either way).

**Suggested fix (out of scope here):** Either (a) feed `amount_ratio`
directly as its own model feature rather than blending it into
`win1_mean`, or (b) clip/smooth `score_to_points()`'s output so small
`fraud_prob` deltas (28%→37%) don't cause `t_raw` jumps of ±7 points, or
(c) retrain with monotonicity constraints on `amount_ratio` if the
sklearn version in use supports them.

---

## #2 — `_combine()` "valley": a weak second signal can LOWER combined_score

**Where:** `scorer._combine()` — pre-existing, not introduced by this fix.

**Symptom:** For a fixed `b_raw` (e.g. 74, giving `b_alone ≈ 34.7`),
adding a *small* `t_raw` (roughly 5-28) produces `combined_score`
**below** `b_alone`:

```
b_alone (t_raw=0)  = 34.72
t_raw=10  -> combined=31.15   (-3.56)
t_raw=20  -> combined=30.10   (-4.61)   <- worst dip
t_raw=30  -> combined=34.99   (+0.28)   back above b_alone
```

**Root cause:** The synergy term `synergy = b_p * t_p * b_norm * t_norm * 30`
requires `t_norm = max(0, (t_c - 20) / 60) > 0`, i.e. `t_c > 20`, which
needs `t_raw` roughly >= 28 (given `_T_K=0.08, _T_MID=50`). For
`t_raw < 28`, `t_norm = 0` so `synergy = 0`, but the cross-penalty term
`b_p * b_c * (1 - t_p * 0.3)` still reduces the behavioural contribution
by up to 30% (since `t_p = min(1, t_raw/25)` reaches 1.0 once
`t_raw >= 25`) — and `term2 = t_p * t_c * (1 - b_p * 0.3)` is too small
(low `t_c`) to compensate.

**Impact:** A session with a strongly elevated behavioural score
(`b_raw≈74`) that ALSO shows a *weak* transaction anomaly (`t_raw≈10-25`)
scores LOWER than if the transaction anomaly hadn't happened at all.
This is counter-intuitive: weak corroborating evidence shouldn't reduce
suspicion. In the test suite, `test_known_anomaly_combine_valley_for_weak_second_signal`
documents the current magnitude (~4.7 points) as a regression tripwire —
if it grows significantly, that's a signal something else changed too.

**Suggested fix (out of scope here):** Lower the synergy activation
threshold (e.g. `t_norm = max(0, (t_c - 5) / 75)` instead of `(t_c-20)/60`)
so synergy contributes something even for weak secondary signals, or
make the cross-penalty terms themselves a function of `t_norm`/`b_norm`
(i.e. only apply the 30% reduction once the *other* layer is itself
"elevated", not merely "present").

---

## #3 — Gate boundary discontinuities (this fix, documented/bounded)

**Where:** `scorer._combine()`, the new amount-aware + t_raw-magnitude gate.

**Symptom:** Two step-function jumps exist at the gate boundaries:

```
amount_ratio: 1.00 -> 1.01 (b_raw=0, t_raw=65)   : 65.90 -> 76.43  (+10.5)
t_raw:        69.9 -> 70.0 (b_raw=0, ratio<=1)   : 65.90 -> 82.89  (+17.0)
```

**Why this exists:** Any hard cap/gate produces a discontinuity at its
boundary by construction — `min(combined, 65.9)` either applies fully or
not at all. There's no continuous interpolation.

**Impact:** Bounded and ONE-DIRECTIONAL (always an upward jump when
crossing OUT of "capped" territory into "uncapped" — i.e. crossing
toward MORE suspicion, never less). This means:
- A transaction at exactly the average amount (ratio=1.00) is capped at
  Tier 1; one rupee more (ratio=1.0000125 for avg=₹8,000, i.e. ₹8,000.10)
  is uncapped and can jump to Tier 2. In practice amounts aren't this
  precise, so this is mostly theoretical, but worth knowing.
- Similarly, `t_raw=69.9` (capped, Tier 1) vs `t_raw=70.0` (uncapped,
  jumps to ~83, Tier 2) — one additional point of accumulated suspicion
  can cross a tier boundary. This is arguably *correct* behaviour for a
  threshold (something has to be the deciding point), but the SIZE of
  the jump (17 points, crossing TWO tier bands in one step from the
  user's perspective: 65.9 is Tier1, 82.89 is Tier2) means progressive
  hardening's dwell timers become the only thing standing between
  "Tier 1, monitor" and "Tier 2, OTP+hold" for a borderline session.

**Suggested fix (out of scope here):** Replace the hard `min(combined, 65.9)`
with a smooth interpolation — e.g. blend between capped and uncapped
values proportionally to how close `amount_ratio`/`t_raw` is to the
threshold, over some transition band (e.g. `amount_ratio in [0.9, 1.1]`
or `t_raw in [60, 80]`).

---

## #4 — Profile-dependent constant tuning

**Where:** `_combine()` constants (`T_RAW_GATE_BYPASS = 70.0`,
`amount_ratio` thresholds), `DEESCALATION_STREAK = 10`,
`DEESCALATION_MARGIN = 1`, dwell timers (`DWELL`).

**Concern:** These constants were tuned against the `arjun` profile
(`avg_transfer_amount=8000`, `avg_dwell_ms=95`). For a profile with a
very different `avg_transfer_amount` (e.g. a high-net-worth account with
`avg_transfer_amount=500000`), the *absolute* ML/rule-based point values
in `TRANSACTION_SIGNALS` (amount_anomaly=+35, etc.) don't scale with the
profile — they're flat point additions regardless of how large "3x
baseline" is in absolute rupees. The `amount_ratio` used by the new gate
IS profile-relative (computed as `amount / avg_transfer_amount`), so the
*gate itself* should behave consistently across profiles — but
`T_RAW_GATE_BYPASS=70` is an absolute `t_raw` threshold, and `t_raw`'s
composition (which signals fire, at what point values) is NOT
profile-relative beyond the `amount_ratio`-scaled `new_beneficiary`
signal.

**Practical effect:** For a profile where `TRANSACTION_SIGNALS` point
values happen to total < 70 even under a "should be Tier 3" structuring
pattern (e.g. if `velocity_spike` and `new_beneficiary` signals are
individually smaller for some reason), the structuring exemption might
not kick in. This wasn't observed with the `arjun` profile (structuring
reached `t_raw=188.5`, well past 70), but hasn't been tested against
other profile shapes.

**Suggested fix (out of scope here):** If new profiles are added with
very different `avg_transfer_amount` or signal-weight expectations,
re-run `test_scorer.py`'s sweep/structuring tests against them and adjust
`T_RAW_GATE_BYPASS` if needed — or derive it as a multiple of the sum of
the two smallest-weight `TRANSACTION_SIGNALS` entries that plausibly fire
together (so it scales with however the signal table is configured,
rather than being a hardcoded magic number).

---

## #5 — ~~RF blend (`combined_score`) can suppress `sigmoid_score` by ~28 points~~ — FIXED

> **STATUS: FIXED.** `combined_score` now only applies the 70/30 RF blend
> when `sigmoid_score < 66` (NORMAL/ELEVATED band). Once `sigmoid_score`
> reaches HIGH RISK/CRITICAL (>=66) based on hard evidence (ThreatShield
> flags + transaction signals), it is authoritative and the RF — which
> only sees keystroke/mouse/device features and has no opinion on
> ThreatShield/transaction evidence — can no longer veto it.
>
> Re-verified `test_phishing_chain_end_to_end`: the same phishing-chain
> scenario described below now produces `sigmoid_score=93.75 ->
> combined_score=93.7, tier=3 (CRITICAL), frozen=True,
> fallback.action="bank_review"` — previously `combined_score=65.9,
> tier=1 (ELEVATED)`. All 22/22 tests still pass.
>
> The analysis below is retained for historical context (it explains
> WHY the old blend was wrong and WHY the new threshold is 66, not some
> other value).

**Where:** `SuspicionScorer.combined_score` property —
`0.70 * sigmoid_score + 0.30 * _ai_prob * 100`.

**Symptom:** In the phishing-chain end-to-end trace (ThreatShield
injects +40/+45 pre-login, then 5 normal-baseline keystrokes, then a
₹4,50,000/new-bene/2am transaction):

```
sigmoid_score  = 93.75   (HIGH RISK / CRITICAL territory, _combine() alone)
_ai_prob       = 0.0104  (RF: "1% fraud probability" — keystrokes look normal)
combined_score = 0.70*93.75 + 0.30*1.04 = 65.94   (ELEVATED — Tier 1)
```

A 28-point drop, crossing from CRITICAL-adjacent (raw_tier would be 3 at
93.75) down to ELEVATED (raw_tier=1 at 65.9).

**Root cause:** `_ai_prob` comes from `model.joblib`, a RandomForest
trained (per `train_model.py`) on synthetic session data with 12
features including `amount_ratio`, `new_bene`, `hour_deviation` (see
`SessionFeatures.to_vector()`). In this scenario those features ARE
populated correctly (`amount_ratio=56.25`, `new_bene=1`,
`hour_deviation=16`) — but the model still outputs `_ai_prob≈0.01`,
because `avg_dwell_ms≈93` (within ~2% of the 95ms baseline) apparently
dominates the model's decision for this input combination. The RF was
trained on a SYNTHETIC dataset (1000 attacker / 3000 legit sessions per
the v1 report) where attacker sessions presumably correlate bot-speed
typing WITH large/new-bene transactions — a "normal-typing attacker with
a large anomalous transaction" combination may be under-represented or
absent in that training distribution, so the model has never learned
this combination is high-risk.

**Impact:** This is THE reason `test_phishing_chain_end_to_end` does not
reach Tier 3/`bank_review` as originally hoped in the fix request — not
the ThreatShield injection mechanism (which works correctly, confirmed
by `sigmoid_score=93.75`), but the RF blend stage AFTER it. A session
with pre-login ThreatShield CRITICAL flags + a textbook
amount/beneficiary/hour anomaly, executed with normal typing, lands at
Tier 1 (ELEVATED, silent monitoring) rather than Tier 3 (frozen,
bank_review) — specifically because the attacker typed normally.

**Why this matters for the THREAT MODEL:** An attacker who has
compromised credentials via phishing (hence the ThreatShield flags) and
either (a) types normally themselves, or (b) automates only the
transaction submission while a human types the login fields, would
receive Tier 1 treatment for a ₹4.5L transfer to a brand-new beneficiary
at 2am — silent monitoring only, transaction proceeds.

**Fix implemented:** `combined_score` now only blends in `_ai_prob` when
`sigmoid_score < 66` — sigmoid_score>=66 is treated as authoritative
regardless of RF opinion, analogous to how the Tier-3 gate already
treats `_combine()`'s output as authoritative once enough evidence
accumulates. Retraining `model.joblib` with "normal typing + anomalous
transaction" as a positive example remains a good longer-term
improvement (it would let the RF itself contribute correctly to
HIGH-RISK-band decisions too, not just NORMAL/ELEVATED), but is no
longer required for the phishing-chain scenario to reach Tier 3.

---

## #6 — 85 raw ThreatShield points land just BELOW the Tier-1 threshold pre-login

**Where:** `_combine()` / `_sigmoid(b_raw, _B_K=0.055, _B_MID=85)`.

**Symptom:** Two CRITICAL ThreatShield injections (scam message +40,
phishing URL +45 — both at the maximum `suspicion_pts_for_session` for
their respective analysers) decay to `b_raw=81.3`. With `t_raw=0` (no
transaction yet), `sigmoid_score = combined_score = 44.4` —
`tier_label="NORMAL"`, NOT `"ELEVATED"` (Tier 1 starts at 46).

**Root cause:** `_sigmoid(81.3, k=0.055, mid=85)≈40.4`; with `t_p=0`
(no transaction), `combined ≈ b_p * b_c = 1.0 * 40.4 ≈ 40` (the actual
value 44.4 differs slightly due to a synergy/term2 contribution even at
`t_raw=0` — see `_combine`'s exact formula). Either way, `_B_MID=85`
means `b_raw` needs to be AT `_B_MID` for `b_c=50`; `b_raw=81.3` is just
under that, yielding `b_c<50` and `combined<46`.

**Impact:** The v3 report's Stage 4 expectation ("pre-elevated session,
Tier 1 ELEVATED, before any keystrokes") does not occur with current
constants — the maximum realistic pre-login ThreatShield injection (two
CRITICAL flags = 85 raw pts) falls 1.6 points short of Tier 1. A THIRD
CRITICAL flag (e.g. also calling `/threat/check-page` with a fake-page
result, +50 pts per `_page_score_to_session_pts`) would very likely cross
into Tier 1 — but two flags alone do not.

**Suggested fix (out of scope here):** Lower `_B_MID` slightly (e.g. to
75-80) so that 2 CRITICAL ThreatShield flags alone cross into Tier 1 —
but this would also make ordinary behavioural-only sessions reach Tier 1
with less accumulated `b_raw`, so it's a global tuning change requiring
re-validation of ALL `b_raw`-only test cases (`test_bot_speed_typing_...`,
de-escalation tests, etc.), not a ThreatShield-specific fix.

---

## #7 — The 80-pt ThreatShield injection cap only applies to `/threat/check-all`, not sequential individual calls

**Where:** `threat_shield.py`'s `comprehensive_check()` —
`"session_suspicion_pts": min(80, total_session_pts)`. The three
individual endpoints (`/threat/check-url`, `/threat/check-message`,
`/threat/check-page`) each call `_inject_threat_pts()` independently
with their own `suspicion_pts_for_session` (max 45/40/50 respectively),
with NO cross-call cap.

**Symptom:** Calling `/threat/check-message` (CRITICAL, +40) then
`/threat/check-url` (CRITICAL, +45) sequentially — exactly the v3
Scenario 2 phishing-chain flow — injects 40+45=85 raw points total,
EXCEEDING the 80-pt cap that `comprehensive_check()` (i.e.
`/threat/check-all`) would have applied to the SAME two inputs combined
into one call (`session_suspicion_pts=80` confirmed by direct test).

**Root cause:** The 80-pt cap is implemented as a LOCAL `min()` inside
`comprehensive_check()`'s return value — it has no session-level
counterpart. `_inject_threat_pts()` in `main.py` has no awareness of how
many ThreatShield points have ALREADY been injected into a given
session's `b_raw` this "phishing chain" — each call is independent.

**Impact:** A frontend that calls `/threat/check-message` then
`/threat/check-url` separately (e.g. as the user pastes a suspicious
message, then clicks the link inside it — two distinct user actions,
two distinct API calls) gets a LARGER session-level injection (85) than
if it had bundled both inputs into one `/threat/check-all` call (80) —
an inconsistency between two code paths that are conceptually "the same
information, different delivery."

In practice this is a minor over-injection (85 vs 80, a 5-point
difference) and doesn't change this fix's test outcomes — but it's
worth noting the cap is NOT a hard session-level invariant, just a
per-call convenience cap in one specific endpoint.

**Suggested fix (out of scope here):** Either (a) track cumulative
ThreatShield-injected points per session (e.g. a
`self._threat_shield_total` counter on `SuspicionScorer`, capped at 80,
with `_add_b` reducing injected pts once the cumulative total would
exceed 80), or (b) remove the cap from `comprehensive_check()` entirely
for consistency (let `_add_b`'s existing `b_raw` cap of 500 be the only
limit, same as every other signal source).

---

## #8 — `layer="threat_shield"` has no dedicated dashboard styling

**Where:** `frontend/dashboard.html`'s `pushSig()` /
`window.addEventListener('message', ...)` handler (around line 850):

```javascript
const l = sig.layer === 'transaction' ? 't' : 'b';
const isDev = sig.layer === 'device';
pushSig(l, sig.signal||'', sig.pts||'', sig.response||'', isDev);
```

**Symptom:** `layer="threat_shield"` signals (added by this fix) route
to the behavioural ('b') column — correct, since these ARE `b_raw`
contributions — but `isDev` only checks `sig.layer === 'device'`, so
ThreatShield signals get NO special highlighting (the `dev` CSS class
that `device_check` signals receive via `row.className =
'sig-row'+(isDev?' dev':'')`).

**Impact:** Cosmetic only. ThreatShield-sourced signals (e.g. "URL
threat [CRITICAL]: http://sbi-kyc-update.xyz...") will appear in the
behavioural signal feed indistinguishable from ordinary keystroke/mouse
signals — a judge/analyst watching the dashboard during a phishing-chain
demo won't visually distinguish "this score increase came from
ThreatShield Layer 0" vs "this score increase came from session
biometrics," even though the backend's `signal_log` DOES carry
`layer="threat_shield"` (confirmed:
`s["signals"][0]["layer"] == "threat_shield"`).

**Suggested fix (out of scope here):** Extend the dashboard's `isDev`
check (or add a parallel `isThreat = sig.layer === 'threat_shield'`) and
add a corresponding CSS class (e.g. `.sig-row.threat { border-left:
3px solid #f59e0b; }`) so Layer 0 detections are visually distinct in
the live signal feed — this is the natural place to surface "this
session was flagged by ThreatShield before login" to a security analyst.

---

## #9 — `display_tier` floor (fix from previous round) can cause single-step badge flicker at the `t_raw=70` gate-bypass boundary

**Where:** Interaction between the `display_tier = max(shown,
fallback_tier, raw)` floor (added to fix the 4×₹6,000 structuring
under-report) and the `T_RAW_GATE_BYPASS=70` gate condition in
`_combine()` (added to fix the structuring exemption, see #4-related
discussion).

**Symptom:** Starting from `t_raw=71.20` (`combined=84.20`,
`raw_tier=2`, `display=2`, `fallback.action="silent_reauth"`), a SINGLE
decay step (`t_raw *= 0.98` → `69.80`, no new signals) produces:

```
combined: 84.20 -> 65.90   (an 18.3-point drop in one step)
raw_tier: 2 -> 1
display:  2 -> 1           (HIGH RISK -> ELEVATED, one decay tick)
fallback: silent_reauth -> none
```

`combined_score` then PLATEAUS at exactly `65.90` for the next ~9 decay
events (while `t_raw` drops from 69.80 to 58.20) before resuming a
smooth decline — because `_combine()`'s gate caps the result at 65.9
for the entire `t_raw < 70` (and `amount_ratio<=1`, `b_p<0.05`) range.

**Root cause:** This is the SAME step-function discontinuity documented
in #3 (gate boundary discontinuities), but #3 was framed as "crossing
INTO uncapped territory jumps UP" (e.g. `t_raw: 69.9->70.0` causes
`65.90->82.89`). This is the REVERSE direction under passive decay:
crossing OUT of uncapped territory (as `t_raw` naturally decreases over
time) causes a DOWNWARD jump (`84.20->65.90`), and the `display_tier`
floor (this fix) means that downward jump is now visible on the DISPLAYED
badge (previously, with `display_tier=max(shown,fallback_tier)` only,
`shown_tier` would have stayed at 1 throughout — i.e. the OLD code
under-reported `raw_tier=2` at `t_raw=71.2` AS WELL, so the flicker was
invisible because the badge was already "wrong" in the other direction).

**Is this a problem?** Two ways to view it:

- **Correctness**: `display=2` at `combined=84.20` and `display=1` at
  `combined=65.90` are BOTH individually correct per their respective
  `raw_tier` values (2 and 1) — the floor is doing its job at each
  instant. There is no "lie" being told at either moment.
- **UX**: A security analyst watching the dashboard in real time would
  see the HIGH RISK badge appear for one polling interval then revert to
  ELEVATED with no new signals firing — which could look like a glitch
  or an inconsistent system, even though each individual reading is
  accurate.

**This is a genuine tradeoff, not silently suppressed**: fixing the
under-report (this fix's purpose) necessarily makes the gate's
pre-existing step-function (anomaly #3) visible on the badge, where
before it was hidden by a DIFFERENT under-report. Smoothing the gate
itself (per #3's suggested fix — interpolate over a transition band
instead of a hard `min()`) would resolve BOTH the upward-jump (#3) and
downward-flicker (#9) cases simultaneously, since `combined_score` would
no longer have a flat plateau/discontinuity at the `t_raw=70` boundary.

**Suggested fix (out of scope here):** Same as #3 — replace the hard
`t_raw < T_RAW_GATE_BYPASS` (70) cutoff with a smooth interpolation over
e.g. `t_raw in [60, 80]`, so `combined_score` (and therefore `raw_tier`
and `display_tier`) changes continuously across this boundary in both
directions (escalating and decaying).

---

## #10 — PS-3 Goal 4 tests were unreachable by the standalone runner (FIXED in this revision)

**Where:** `test_scorer.py` — the 5 tests for `failed_login` / PS-3 Goal 4
hardening (`test_first_failed_login_activates_security_phrase`,
`test_second_failed_login_escalates_to_otp`,
`test_third_failed_login_triggers_call_and_freeze`,
`test_failed_login_count_cleared_by_reset_only`,
`test_normal_session_never_increments_failed_login_count`).

**Symptom:** When the `failed_login` feature was added, its 5 tests
were appended to the BOTTOM of `test_scorer.py`, AFTER the
`if __name__ == "__main__":` block that collects and runs every
`test_*` function from `globals()`. Running `python test_scorer.py`
printed `32 passed, 0 failed` — identical to the count BEFORE the
feature was added — because the test-runner loop executes before the
Python interpreter reaches the function definitions located below it
in the file. The 5 new functions existed in the module but were never
collected into the `tests` list, so they silently never ran.

**Root cause:** `for k, v in list(globals().items())` only sees names
that have been bound by the time that line executes. Functions defined
textually below the `if __name__ == "__main__":` guard are not yet
bound when the guard's body runs (top-to-bottom module execution),
even though they exist later in the same file.

**Verification:** Manually importing and calling the 5 functions
directly (bypassing the runner) confirmed all 5 pass — the
*underlying logic* was correct from the start; only the runner's
visibility into them was broken. This means the feature itself was
never broken, but the test suite was silently overstating its own
coverage: anyone trusting the "32 passed" printout as confirmation
that the new feature was tested would have been wrong.

**Fix applied:** Moved all 5 `test_failed_login_*` /
`test_*_failed_login_*` function definitions (plus their section
header comment) to immediately ABOVE the
`if __name__ == "__main__":` block, alongside every other test
function in the file. No logic was changed — this was a pure
relocation. Re-running `python test_scorer.py` now correctly reports
`37 passed, 0 failed`.

**Lesson for future additions:** Any new `test_*` function added to
this file MUST be placed above the `if __name__ == "__main__":` guard
near the end of the file, never below it. Consider adding a one-line
assertion at the top of the runner block (e.g.
`assert len(tests) >= <expected_count>`) as a tripwire against this
exact mistake recurring silently in a future revision.
