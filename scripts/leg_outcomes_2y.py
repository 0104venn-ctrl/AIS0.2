# -*- coding: utf-8 -*-
"""7%+ 급등 레그 전수 — '100% 되돌림에 도달할 것인가'를 미리 알 수 있나 (user 2026-09-11).

지금까지 본 실거래는 전부 '100% 도달한 것'만 모은 표본이라 선택편향이 있다.
여기서는 레그가 고점 확정(피벗 +4봉)된 순간부터 추적해서
  - 30/50/70/90/100% 되돌림 레벨을 언제 처음 깼는지
  - 100% 도달 전에 고점을 갱신했는지 (레그 실패)
  - 120봉 안에 아무것도 안 났는지 (시간초과)
를 기록한다. 레그 검출은 backtest_ais02_live_2y.causal_trades 와 동일 (4봉 지연 피벗 · 400봉 창).

각 레벨 X 에서 '숏 진입 → 100% 레벨 익절 / 고점 갱신 손절 / 120봉 시간초과' 의 손익도 같이 낸다.
실행: python scripts/leg_outcomes_2y.py
"""
import glob
import math
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

RISE_MIN = 7.0
PIVK, CROSS_WIN, WINDOW, MIN_BARS = 4, 120, 400, 80
LEVELS = (30, 50, 70, 90, 100)
FEE_PCT = 0.11            # 왕복 노셔널 %
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def legs_of(ts, closes, highs, lows, qv, tbqv):
    """피벗(4봉 프랙탈, 종가)을 구해 연속 같은 종류는 극값만 남겨 hi/lo 교대열로 만든 뒤,
    lo→hi 쌍 중 상승 7%+ 를 레그로 본다. 고점은 top+PIVK 에 확정된다고 보고 그 뒤부터 추적."""
    n = len(closes)
    if n < MIN_BARS:
        return []
    lg = np.log(np.maximum(closes, 1e-12))
    raw = []
    for i in range(PIVK, n - PIVK):
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
        if not kind:
            continue
        if raw and raw[-1][1] == kind:
            old = raw[-1][0]
            if (kind == 'hi' and ci > closes[old]) or (kind == 'lo' and ci < closes[old]):
                raw[-1] = (i, kind)
        else:
            raw.append((i, kind))
    out = []
    for k in range(1, len(raw)):
        idx, kind = raw[k]
        if kind != 'hi' or raw[k - 1][1] != 'lo':
            continue
        start, top = raw[k - 1][0], idx
        if top - start > WINDOW:
            continue
        rise = lg[top] - lg[start]
        if rise <= 0 or math.expm1(rise) * 100 < RISE_MIN:
            continue
        top_lg, start_lg = lg[top], lg[start]
        confirm = top + PIVK
        end = min(n - 1, top + CROSS_WIN)
        cross = {lv: None for lv in LEVELS}
        fail_at = None
        for u in range(confirm + 1, end + 1):
            if cross[100] is None and fail_at is None and highs[u] > closes[top] * 1.0:
                fail_at = u
            for lv in LEVELS:
                if cross[lv] is None and fail_at is None and lg[u] <= top_lg - lv / 100.0 * rise:
                    cross[lv] = u
            if cross[100] is not None or fail_at is not None:
                break
        rec = dict(sym=None, top_ts=int(ts[top]), top=float(closes[top]), start=float(closes[start]),
                   rise_pct=round(math.expm1(rise) * 100, 2), rise_bars=int(top - start),
                   fail_at=(fail_at - top) if fail_at is not None else None,
                   reached_100=cross[100] is not None)
        for lv in LEVELS:
            rec['x%d' % lv] = (cross[lv] - top) if cross[lv] is not None else None
        for lv in (30, 50, 70, 90):
            u = cross[lv]
            if u is None:
                continue
            e = closes[u]
            tp = math.exp(start_lg)
            sl = closes[top]
            res, px = None, None
            for v in range(u + 1, end + 1):
                if highs[v] >= sl:
                    res, px = 'SL', sl
                    break
                if lows[v] <= tp:
                    res, px = 'TP', tp
                    break
            if res is None:
                res, px = 'TIME', closes[end]
            rec['r%d' % lv] = res
            rec['p%d' % lv] = round((e - px) / e * 100 - FEE_PCT, 4)
            w0 = max(0, u - 15)
            q = float(qv[w0:u].sum())
            b = float(tbqv[w0:u].sum())
            rec['br15_%d' % lv] = round(b / q * 100, 2) if q > 0 else None
            base_q = float(qv[max(0, start - 100):start].mean()) if start > 5 else 0.0
            rec['volx_%d' % lv] = round(float(qv[w0:u].mean()) / base_q, 3) if base_q > 0 else None
            rec['speed_%d' % lv] = int(u - top)
        out.append(rec)
    return out


def one(path):
    sym = os.path.basename(path)[:-4]
    z = np.load(path)
    rows = legs_of(z["ts"], z["c"], z["h"], z["l"], z["qv"], z["tbqv"])
    for r in rows:
        r['sym'] = sym
    return rows


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "binance_1m", "*.npz")))
    print(f"{len(files)}종목 · 상승 {RISE_MIN}%+ 레그 전수 · 추적 {CROSS_WIN}봉", flush=True)
    rows, done = [], 0
    with ProcessPoolExecutor(14) as ex:
        for r in ex.map(one, files, chunksize=4):
            rows.extend(r)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(files)} · 레그 {len(rows):,}", flush=True)
    d = pd.DataFrame(rows)
    d['top_at'] = pd.to_datetime(d.top_ts, unit='ms')
    out = os.path.join(ROOT, "exports", "LEG_OUTCOMES_2Y.csv")
    d.to_csv(out, index=False)
    print(f"저장 {out} · 레그 {len(d):,}건 · {d.sym.nunique()}종목")


if __name__ == "__main__":
    main()
