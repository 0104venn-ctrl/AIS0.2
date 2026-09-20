# -*- coding: utf-8 -*-
"""AIS0.2 — VPS 라이브 실제 설정으로 24개월 백테스트.

로직은 scripts/backtest_ais02_bybit_causal.py 의 causal_trades 와 동일
(400봉 트레일링 윈도우 · 4봉 지연 피벗 · 첫 이탈 진입 · 미래정보 없음).
파라미터만 **2026-09-05 VPS status/env 실측값**으로 교체:

  rise>=7.0 · retrace 100% · TP 0.6R · SL 1.0R · 30봉 · max_open 2 · 시장가
  LEV = 2            (AIS02B_LEVERAGE=2)
  SL_REF_LEV = 9     (손절캡만 9배 기준 → 역행 +4.28% 에서 손절, 청산은 +45%)
  MMR = 0.05         (AIS02B_MMR_MAX=0.05, 조회 실패 시 백엔드 폴백과 동일)

데이터: data/binance_1m (805종목, 2024-09 ~ 2026-08)
실행: python scripts/backtest_ais02_live_2y.py
"""
import glob
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

RISE_MIN, RETRACE = 7.0, 100.0
TP_R, SL_R, HOLD = 0.60, 1.00, 30
LEV, SL_REF_LEV, MMR, SL_LIQ_RATIO = 2.0, 9.0, 0.05, 0.70
MARGIN = 100.0
FEE = 0.11 * LEV                       # 왕복 노셔널 0.11% → 증거금 %
PIVK, CROSS_WIN, WINDOW, MIN_BARS = 4, 120, 400, 80

LIQ_ADV = (1.0 / LEV - MMR) * 100.0                       # 45.00%
CAP_ADV = min((1.0 / SL_REF_LEV - MMR) * 100.0, LIQ_ADV)  # 6.11%
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pnl_of(why, e, px):
    if why == "LIQ":
        return -MARGIN
    return max(-MARGIN, ((e - px) / e * LEV * 100 - FEE) / 100 * MARGIN)


def simulate_exit(highs, lows, closes, ent, e, rise, n):
    tp = e * math.exp(-TP_R * rise)
    sl = min(e * math.exp(SL_R * rise), e * (1 + CAP_ADV * SL_LIQ_RATIO / 100.0))
    liq = e * (1 + LIQ_ADV / 100.0)
    w1, p1, w2, p2 = ("SL", sl, "LIQ", liq) if sl <= liq else ("LIQ", liq, "SL", sl)
    end = min(n - 1, ent + HOLD)
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


def causal_trades(ts, closes, highs, lows, qv, tbqv):
    """backtest_ais02_bybit_causal.causal_trades 와 동일 알고리즘."""
    n = len(closes)
    if n < MIN_BARS:
        return []
    lg = np.log(np.maximum(closes, 1e-12))
    raw, merged, candidates, trades = [], [], [], []
    dirty, open_until = True, -1

    def remerge(t):
        nonlocal merged, candidates
        lo_bound = t - WINDOW + 1
        merged = []
        for idx, kind in raw:
            if idx < lo_bound:
                continue
            if merged and merged[-1][1] == kind:
                old = merged[-1][0]
                if (kind == 'hi' and closes[idx] > closes[old]) or \
                   (kind == 'lo' and closes[idx] < closes[old]):
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
                if rise > 0 and math.expm1(rise) * 100 >= RISE_MIN:
                    candidates.append([start, idx, lg[start], old_state.get(idx, False)])
        candidates.reverse()

    for t in range(n):
        i = t - PIVK
        if i >= PIVK:
            ci = closes[i]
            hi = lo = True
            for j in range(1, PIVK + 1):
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
                    if (kind == 'hi' and ci > closes[old]) or \
                       (kind == 'lo' and ci < closes[old]):
                        raw[-1] = (i, kind)
                else:
                    raw.append((i, kind))
                dirty = True
        if raw and raw[0][0] < t - WINDOW + 1:
            while raw and raw[0][0] < t - WINDOW + 1:
                raw.pop(0)
            dirty = True
        if dirty:
            remerge(t)
            dirty = False
        if t <= open_until or t < MIN_BARS:
            continue
        for cand in candidates:
            start, top, start_lg, crossed = cand
            if crossed or t <= top + PIVK or t >= top + CROSS_WIN:
                continue
            if lg[t] <= start_lg:
                cand[3] = True
                e = closes[t]
                rise = lg[top] - start_lg
                j, why, px = simulate_exit(highs, lows, closes, t, e, rise, n)
                w0 = max(0, t - 60)
                q60 = float(qv[w0:t].sum())
                tb60 = float(tbqv[w0:t].sum())
                d0 = max(0, t - 1440)
                seg = closes[w0:t]
                trades.append((int(ts[t]), int(ts[j]), float(e), float(px), why,
                               round(pnl_of(why, e, px), 4),
                               round(math.expm1(rise) * 100, 2),
                               int(top - start),          # 상승 소요 봉수
                               int(t - top),              # 반납 소요 봉수
                               float(qv[d0:t].sum()),     # 24h 거래대금
                               round(tb60 / q60, 4) if q60 > 0 else 0.5,   # 테이커 매수비율 60분
                               round(float(np.std(np.diff(np.log(np.maximum(seg, 1e-12))))) * 1e4, 2)
                               if len(seg) > 2 else 0.0,  # 60분 변동성(bp)
                               int(len(candidates))))     # 동시 활성 레그 수
                open_until = j
                break
    return trades


def one(path):
    sym = os.path.basename(path)[:-4]
    z = np.load(path)
    tr = causal_trades(z["ts"], z["c"], z["h"], z["l"], z["qv"], z["tbqv"])
    return [(sym,) + t for t in tr]


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "binance_1m", "*.npz")))
    print(f"{len(files)}종목 · LEV {LEV:.0f}배 · 손절 역행 +{CAP_ADV*SL_LIQ_RATIO:.2f}% "
          f"· 청산 +{LIQ_ADV:.0f}% · TP {TP_R}R", flush=True)
    rows, done = [], 0
    with ProcessPoolExecutor(14) as ex:
        for r in ex.map(one, files, chunksize=4):
            rows.extend(r)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(files)} · 누적 {len(rows):,}건", flush=True)

    d = pd.DataFrame(rows, columns=["symbol", "entry_ts", "exit_ts", "entry",
                                    "exit", "why", "pnl", "rise_pct",
                                    "rise_bars", "retrace_bars", "qv24h",
                                    "taker_buy_60", "vol60_bp", "n_legs"])
    d["entry_at"] = pd.to_datetime(d.entry_ts, unit="ms")
    d = d.sort_values("entry_ts").reset_index(drop=True)
    d.to_csv(os.path.join(ROOT, "exports", "AIS02_LIVE_2Y_FEATURES.csv"),
             index=False, encoding="utf-8")

    print(f"\n총 {len(d):,}건 · {d.symbol.nunique()}종목 · "
          f"{d.entry_at.min():%Y-%m-%d} ~ {d.entry_at.max():%Y-%m-%d}")
    print(f"합계 ${d.pnl.sum():+,.0f} · 건당 ${d.pnl.mean():+.3f} · "
          f"승률 {(d.pnl>0).mean():.1%} · t {d.pnl.mean()/d.pnl.std()*np.sqrt(len(d)):.2f}")
    print(f"청산사유: {d.why.value_counts().to_dict()}")
    m = d.pnl.groupby(d.entry_at.dt.to_period('M')).agg(["count", "sum"])
    m.columns = ["건수", "손익$"]
    print("\n=== 월별 ===")
    print(m.to_string(float_format=lambda v: f"{v:,.1f}"))
    print(f"\n플러스 월: {(m['손익$']>0).sum()} / {len(m)}")


if __name__ == "__main__":
    main()
