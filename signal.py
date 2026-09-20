# -*- coding: utf-8 -*-
"""AIS0.2 — the complete strategy logic, extracted verbatim from the live bot.

This is everything the bot needed to decide *when* to enter and *where* to exit.
Everything else in the production code (exchange auth, order routing, DB, health
checks) is plumbing and is intentionally not included.

Data assumption: a trailing window of 1-minute closes (400 bars in production),
scanned once per minute. All functions are causal — nothing below looks past
the last bar of the window.
"""
import math

# ---------------------------------------------------------------- parameters
# Values are the live configuration as of 2026-09-08 (the last live trade).
RISE_MIN = 7.0          # minimum single-leg pump, %  (log-space check below)
RETRACE = 100.0         # entry when price gives back this % of the pump
TP_R = 0.60             # take-profit at 0.6 x pump size (log-space)
SL_R = 1.00             # stop-loss at 1.0 x pump size (log-space), before cap
HOLD_MIN = 30           # forced market exit after 30 bars
LEV = 2                 # actual leverage (was 9 until 2026-09-03)
SL_REF_LEV = 9.0        # leverage used for the stop cap  (see apply_sl_cap)
SL_LIQ_RATIO = 0.70     # stop sits at 70% of the distance to liquidation
MMR_MAX = 0.05          # symbols with maintenance margin rate > 5% are excluded
PIVK, CROSS_WIN, WINDOW = 4, 120, 400   # pivot half-width, cross window, scan window


# ---------------------------------------------------------------- signal
def zig(closes):
    """4-bar fractal pivots on log closes -> alternating hi/lo pivot list -> legs.

    A bar i is a 'hi' pivot if close[i] >= close[i±1..4]; 'lo' if <=.
    Consecutive pivots of the same kind are merged, keeping the extreme, so the
    result strictly alternates hi/lo. Each adjacent pivot pair is a leg.
    """
    n = len(closes)
    lg = [math.log(max(1e-12, c)) for c in closes]
    piv = [(0, 'end')]
    for i in range(PIVK, n - PIVK):
        hi = all(closes[i] >= closes[i - j] and closes[i] >= closes[i + j] for j in range(1, PIVK + 1))
        lo = all(closes[i] <= closes[i - j] and closes[i] <= closes[i + j] for j in range(1, PIVK + 1))
        t = 'hi' if hi else ('lo' if lo else None)
        if not t:
            continue
        if piv and piv[-1][1] == t:
            if (t == 'hi' and closes[i] > closes[piv[-1][0]]) or \
               (t == 'lo' and closes[i] < closes[piv[-1][0]]):
                piv[-1] = (i, t)
        else:
            piv.append((i, t))
    piv.append((n - 1, 'end'))
    legs = []
    for a in range(len(piv) - 1):
        s, e = piv[a][0], piv[a + 1][0]
        legs.append(dict(si=s, ei=e, sLg=lg[s], eLg=lg[e], d=lg[e] - lg[s]))
    return lg, legs


def percent_retrace_signal(closes, tol=2):
    """Return an entry dict if the LAST bar (within `tol`) crossed the 100% retrace
    level of a >= RISE_MIN% up-leg, else None.

    For each candidate up-leg (newest first):
      level = log(top) - RETRACE/100 * rise            # = log(leg start) at 100%
      look for the first bar t in [top+PIVK, top+CROSS_WIN) with log close <= level
      fire only if that t is within the last `tol` bars  (i.e. the cross just happened)

    The PIVK offset means a top is only 'known' 4 bars after it prints — the
    detector never uses a pivot before it is confirmed.
    """
    n = len(closes)
    if n < 80:
        return None
    lg, legs = zig(closes)
    candidates = []
    for leg in legs:
        rise = leg['d']
        if rise > 0 and math.expm1(rise) * 100 >= RISE_MIN:
            candidates.append((leg, rise))
    for leg, rise in reversed(candidates):
        top = leg['ei']
        level = leg['eLg'] - (RETRACE / 100.0) * rise
        for t in range(max(top + PIVK, top + 1), min(n, top + CROSS_WIN)):
            if lg[t] <= level:
                if t >= n - 1 - tol:
                    E = lg[t]
                    return {'E': E, 'R': rise, 'entry': math.exp(E),
                            'rise_pct': math.expm1(rise) * 100,
                            'top': math.exp(leg['eLg']),
                            'tp': math.exp(E - TP_R * rise),
                            'sl': math.exp(E + SL_R * rise)}
                break
    return None


# ---------------------------------------------------------------- stop cap
def apply_sl_cap(sig, mmr):
    """Cap the stop so it always sits inside the liquidation price.

    liq_adv  = distance to liquidation at the ACTUAL leverage   (1/LEV - MMR)
    cap_adv  = distance used for the cap, at SL_REF_LEV          (1/9   - MMR)
    stop     = min(1.0R stop, entry * (1 + cap_adv * 0.70))

    With SL_REF_LEV = 9 and MMR = 2..5%, cap_adv * 0.70 = 4.3..6.4%, which is
    tighter than 1.0R for any pump >= 7%. So in practice the stop was fixed by
    MMR, not by the strategy — the structural flaw discussed in the write-up.
    """
    liq_adv = (1.0 / LEV - mmr) * 100.0
    cap_adv = (1.0 / SL_REF_LEV - mmr) * 100.0
    if liq_adv <= 0 or cap_adv <= 0:
        return None
    cap_adv = min(cap_adv, liq_adv)          # never let the cap sit outside liquidation
    cap = sig['entry'] * (1.0 + cap_adv * SL_LIQ_RATIO / 100.0)
    sig['liq_adv_pct'] = round(liq_adv, 4)
    sig['sl_cap_pct'] = round(cap_adv, 4)
    sig['mmr'] = mmr
    sig['sl_capped'] = cap < sig['sl']
    sig['sl'] = min(sig['sl'], cap)
    return sig


# ---------------------------------------------------------------- exit (backtest form)
def simulate_exit(highs, lows, closes, ent, e, rise, n, mmr=MMR_MAX):
    """Walk forward from entry index `ent` at fill price `e`.
    Priority inside a bar: stop/liquidation first, then take-profit (conservative).
    Returns (exit_index, reason, exit_price). Reasons: SL, LIQ, TP, TIME.
    """
    liq_adv = (1.0 / LEV - mmr) * 100.0
    cap_adv = min((1.0 / SL_REF_LEV - mmr) * 100.0, liq_adv)
    tp = e * math.exp(-TP_R * rise)
    sl = min(e * math.exp(SL_R * rise), e * (1 + cap_adv * SL_LIQ_RATIO / 100.0))
    liq = e * (1 + liq_adv / 100.0)
    w1, p1, w2, p2 = ("SL", sl, "LIQ", liq) if sl <= liq else ("LIQ", liq, "SL", sl)
    end = min(n - 1, ent + HOLD_MIN)
    last = e
    for j in range(ent + 1, end + 1):
        if highs[j] >= p1:
            return j, w1, p1
        if highs[j] >= p2:
            return j, w2, p2
        if lows[j] <= tp:
            return j, "TP", tp
        last = closes[j]
    return end, "TIME", last


if __name__ == "__main__":
    # Tiny smoke test: a synthetic 7%+ pump that retraces to its origin.
    import random
    random.seed(1)
    px = [100.0]
    for _ in range(150):
        px.append(px[-1] * (1 + random.gauss(0, 0.001)))
    base = px[-1]
    for k in range(40):                       # pump +8% over 40 bars
        px.append(base * (1 + 0.08 * (k + 1) / 40))
    top = px[-1]
    for k in range(30):                       # retrace back to base
        px.append(top - (top - base) * (k + 1) / 30)
    px.append(base * 0.999)                   # cross the level
    sig = percent_retrace_signal(px)
    assert sig is not None, "signal should fire on a full retrace"
    sig = apply_sl_cap(sig, 0.05)
    print("fired: rise %.2f%%  entry %.4f  tp %.4f  sl %.4f  (capped=%s, cap %.2f%%)"
          % (sig['rise_pct'], sig['entry'], sig['tp'], sig['sl'], sig['sl_capped'], sig['sl_cap_pct'] * SL_LIQ_RATIO))
