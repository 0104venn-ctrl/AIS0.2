# -*- coding: utf-8 -*-
"""AIS0.2 진입 타이밍 변형 — 신호는 그대로, 진입만 바꾼다 (user 2026-09-11).

기준: backtest_ais02_live_2y.py 의 라이브 설정 (2배 · SL_REF_LEV 9 · TP 0.6R · 30봉 · $100).
변형
  delay=N   신호봉 종가가 아니라 N봉 뒤 종가에 진입 (TP/SL/타임아웃은 새 진입가·진입시각 기준)
  bounce=B  신호 후 M봉 안에 고가가 신호가 x (1+B%) 에 닿으면 그 가격에 진입, 안 닿으면 포기
신호 검출은 한 번만 하고 변형별로 청산을 각각 돌린다 (같은 신호 집합 = 공정 비교).
"""
import glob
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import backtest_ais02_live_2y as L   # noqa: E402  (파라미터·검출기 재사용)

VARIANTS = [("기준 (신호봉 종가)", 0, 0.0, 0)] + \
           [("지연 %d분" % d, d, 0.0, 0) for d in (1, 2, 3, 5, 10)] + \
           [("반등 %.1f%% 대기 (%d분)" % (b, m), 0, b, m) for b in (0.5, 1.0, 1.5, 2.0) for m in (5, 10)]


def signals_of(ts, closes, highs, lows):
    """causal_trades 와 동일하게 신호 시각 t 와 rise 만 뽑는다 (청산은 안 함)."""
    n = len(closes)
    if n < L.MIN_BARS:
        return []
    lg = np.log(np.maximum(closes, 1e-12))
    raw, merged, candidates, sigs = [], [], [], []
    dirty = True

    def remerge(t):
        nonlocal merged, candidates
        lo_bound = t - L.WINDOW + 1
        merged = []
        for idx, kind in raw:
            if idx < lo_bound:
                continue
            if merged and merged[-1][1] == kind:
                old = merged[-1][0]
                if (kind == 'hi' and closes[idx] > closes[old]) or (kind == 'lo' and closes[idx] < closes[old]):
                    merged[-1] = (idx, kind)
            else:
                merged.append((idx, kind))
        old_state = {c[1]: c[3] for c in candidates}
        candidates = []
        prev_idx = max(0, lo_bound)
        for k, (idx, kind) in enumerate(merged):
            if kind == 'hi':
                start = merged[k - 1][0] if k > 0 else prev_idx
                rise = lg[idx] - lg[start]
                if rise > 0 and math.expm1(rise) * 100 >= L.RISE_MIN:
                    candidates.append([start, idx, lg[start], old_state.get(idx, False)])
        candidates.reverse()

    for t in range(n):
        i = t - L.PIVK
        if i >= L.PIVK:
            ci = closes[i]
            hi = lo = True
            for j in range(1, L.PIVK + 1):
                if not (ci >= closes[i - j] and ci >= closes[i + j]):
                    hi = False
                if not (ci <= closes[i - j] and ci <= closes[i + j]):
                    lo = False
                if not hi and not lo:
                    break
            kind = 'hi' if hi else ('lo' if lo else None)
            if kind:
                if raw and raw[-1][1] == kind:
                    old = raw[-1][0]
                    if (kind == 'hi' and ci > closes[old]) or (kind == 'lo' and ci < closes[old]):
                        raw[-1] = (i, kind)
                else:
                    raw.append((i, kind))
                dirty = True
        if raw and raw[0][0] < t - L.WINDOW + 1:
            while raw and raw[0][0] < t - L.WINDOW + 1:
                raw.pop(0)
            dirty = True
        if dirty:
            remerge(t)
            dirty = False
        if t < L.MIN_BARS:
            continue
        for cand in candidates:
            start, top, start_lg, crossed = cand
            if crossed or t <= top + L.PIVK or t >= top + L.CROSS_WIN:
                continue
            if lg[t] <= start_lg:
                cand[3] = True
                sigs.append((t, lg[top] - start_lg))
                break
    return sigs


def run_variant(sigs, ts, closes, highs, lows, delay, bounce, bwin):
    n = len(closes)
    out, open_until = [], -1
    for t, rise in sigs:
        if t <= open_until:
            continue
        if bounce > 0:
            target = closes[t] * (1 + bounce / 100.0)
            ent = None
            for j in range(t + 1, min(n - 1, t + bwin) + 1):
                if highs[j] >= target:
                    ent, e = j, target
                    break
            if ent is None:
                continue
        else:
            ent = t + delay
            if ent >= n - 1:
                continue
            e = closes[ent]
        j, why, px = L.simulate_exit(highs, lows, closes, ent, e, rise, n)
        out.append((int(ts[t]), int(ts[ent]), int(ts[j]), float(e), float(px), why, round(L.pnl_of(why, e, px), 4)))
        open_until = j
    return out


def one(path):
    sym = os.path.basename(path)[:-4]
    z = np.load(path)
    ts, c, h, lo = z["ts"], z["c"], z["h"], z["l"]
    sigs = signals_of(ts, c, h, lo)
    res = {}
    for name, d, b, m in VARIANTS:
        res[name] = [(sym,) + r for r in run_variant(sigs, ts, c, h, lo, d, b, m)]
    return res


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "binance_1m", "*.npz")))
    print(f"{len(files)}종목 · 변형 {len(VARIANTS)}개 · 라이브 설정 (LEV {L.LEV:.0f}배 · 손절캡 {L.CAP_ADV*L.SL_LIQ_RATIO:.2f}% · TP {L.TP_R}R)", flush=True)
    acc = {name: [] for name, *_ in VARIANTS}
    done = 0
    with ProcessPoolExecutor(14) as ex:
        for res in ex.map(one, files, chunksize=4):
            for k, v in res.items():
                acc[k].extend(v)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(files)}", flush=True)
    cols = ["symbol", "sig_ts", "entry_ts", "exit_ts", "entry", "exit", "why", "pnl"]
    CUT = pd.Timestamp("2026-03-01").value // 10**6
    D = chr(36)
    print()
    print("%-24s %7s %7s %8s %9s %7s   |  %8s %7s   |  %8s %7s" % ("변형", "거래", "승률", "건당" + D, "합계" + D, "t", "앞18개월", "t", "뒤6개월", "t"))
    print("-" * 112)
    rows = []
    for name, *_ in VARIANTS:
        d = pd.DataFrame(acc[name], columns=cols)
        if len(d) < 30:
            print("%-24s %7d  (표본 부족)" % (name, len(d)))
            continue

        def st(x):
            p = x.pnl.values
            sd = p.std(ddof=1)
            return p.mean(), p.sum(), p.mean() / (sd / math.sqrt(len(p))) if sd else 0, 100.0 * (p > 0).mean()
        mu, tot, t, win = st(d)
        a = d[d.sig_ts < CUT]
        b = d[d.sig_ts >= CUT]
        ta = st(a)[1:3] if len(a) > 30 else (float('nan'), float('nan'))
        tb = st(b)[1:3] if len(b) > 30 else (float('nan'), float('nan'))
        print("%-24s %7d %6.1f%% %+8.3f %+9.0f %+7.2f   |  %+8.0f %+7.2f   |  %+8.0f %+7.2f"
              % (name, len(d), win, mu, tot, t, ta[0], ta[1], tb[0], tb[1]))
        rows.append(dict(variant=name, n=len(d), win=win, mu=mu, tot=tot, t=t))
        d.to_csv(os.path.join(ROOT, "exports", "ENTRY_TIMING_%s.csv" % name.split()[0].replace('(', '').replace('%', '')), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "exports", "ENTRY_TIMING_SUMMARY.csv"), index=False)


if __name__ == "__main__":
    main()
