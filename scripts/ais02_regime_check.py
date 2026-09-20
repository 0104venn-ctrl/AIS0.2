# -*- coding: utf-8 -*-
"""AIS0.2 가 상승장에서 나쁜가? — 24개월 거래를 시장 국면별로 나눠본다 (user 2026-09-18).

국면 지표 (전부 진입일 '이전' 데이터만):
  breadth7  = 직전 7일 수익률이 양수인 종목 비율 (%)      — 시장 폭
  mkt7      = 전 종목 직전 7일 수익률 중앙값 (%)           — 시장 방향
  btc7      = BTC 직전 7일 수익률 (%)
"""
import glob
import math
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = chr(36)

# 1) 일봉 종가 (00:00 UTC 기준 첫 봉의 시가 = 전일 종가 근사)
daily = {}
for p in sorted(glob.glob(os.path.join(ROOT, "data", "binance_1m", "*.npz"))):
    z = np.load(p)
    ts, c = z["ts"], z["c"]
    day = (ts // 86_400_000).astype(np.int64)
    # 각 날의 마지막 봉 종가
    idx = np.flatnonzero(np.diff(day) != 0)
    days = day[idx]
    closes = c[idx]
    if len(days) < 30:
        continue
    daily[os.path.basename(p)[:-4]] = pd.Series(closes, index=days)
px = pd.DataFrame(daily).sort_index()
print("일봉 %d일 × %d종목" % px.shape)
r7 = px.pct_change(7) * 100
breadth7 = (r7 > 0).sum(axis=1) / r7.notna().sum(axis=1) * 100
mkt7 = r7.median(axis=1)
btc7 = r7["BTCUSDT"] if "BTCUSDT" in r7 else mkt7
reg = pd.DataFrame({"breadth7": breadth7, "mkt7": mkt7, "btc7": btc7})
# 진입일에 쓸 값은 '전날'까지의 값 → 하루 shift
reg = reg.shift(1)

# 2) 거래 로드 + 국면 매핑
tr = pd.read_csv(os.path.join(ROOT, "exports", "AIS02_LIVE_2Y_TRADES.csv"))
tr["day"] = (tr.entry_ts // 86_400_000).astype(np.int64)
tr = tr.join(reg, on="day")
tr = tr[tr.mkt7.notna()].copy()
tr["win"] = (tr.pnl > 0).astype(int)
print("거래 %s건 (국면 매핑됨)\n" % format(len(tr), ","))


def stat(g):
    p = g.pnl.values
    sd = p.std(ddof=1) if len(p) > 1 else 0
    t = p.mean() / (sd / math.sqrt(len(p))) if sd else 0
    days = g.day.nunique()
    return len(p), 100.0 * g.win.mean(), p.mean(), p.sum(), t, p.sum() / max(1, days)


for col, name, unit in (("mkt7", "시장 7일 수익률 중앙값", "%"), ("breadth7", "7일 상승 종목 비율", "%"), ("btc7", "BTC 7일 수익률", "%")):
    tr["q"] = pd.qcut(tr[col], 5, labels=False, duplicates="drop")
    print("=== 국면: %s (5분위, 낮음→높음) ===" % name)
    print("  %-4s %14s %7s %7s %8s %9s %7s %9s" % ("분위", "범위", "거래", "승률", "건당" + D, "합계" + D, "t", "일평균" + D))
    print("  " + "-" * 72)
    for q, g in tr.groupby("q"):
        n, w, mu, tot, t, pd_ = stat(g)
        print("  Q%d  %+6.1f~%+6.1f%s %7d %6.1f%% %+8.3f %+9.0f %+7.2f %+9.2f"
              % (q + 1, g[col].min(), g[col].max(), unit, n, w, mu, tot, t, pd_))
    print()

# 3) 단순 이분: 상승장 vs 하락장
print("=== 상승장 vs 하락장 (시장 7일 중앙값 기준) ===")
for lab, m in (("하락장 (mkt7 < -3%)", tr.mkt7 < -3), ("보합 (-3 ~ +3%)", (tr.mkt7 >= -3) & (tr.mkt7 <= 3)), ("상승장 (mkt7 > +3%)", tr.mkt7 > 3)):
    g = tr[m]
    if len(g) < 30:
        continue
    n, w, mu, tot, t, pd_ = stat(g)
    print("  %-22s %6d건  승률 %.1f%%  건당 %s%+.3f  합계 %s%+7.0f  t %+.2f  일평균 %s%+.2f" % (lab, n, w, D, mu, D, tot, t, D, pd_))
print()
# 4) 국면별 기간 분포 (앞18/뒤6 홀드아웃과 국면이 겹치는지)
CUT = pd.Timestamp("2026-03-01").value // 10**6 // 86_400_000
print("=== 홀드아웃 6개월(2026-03~)의 국면 ===")
for lab, m in (("앞 18개월", tr.day < CUT), ("뒤 6개월", tr.day >= CUT)):
    g = tr[m]
    print("  %-10s 시장7일 중앙값 %+.2f%% · 상승비율 %.0f%% · 상승장 거래 %.0f%% / 하락장 %.0f%%"
          % (lab, g.mkt7.median(), g.breadth7.median(), 100.0 * (g.mkt7 > 3).mean(), 100.0 * (g.mkt7 < -3).mean()))
tr.to_csv(os.path.join(ROOT, "exports", "AIS02_LIVE_2Y_TRADES_REGIME.csv"), index=False)
