# -*- coding: utf-8 -*-
"""AIS0.2 라이브 설정 24개월 거래(15,733건)에 '최근 2일 상승률' 조건을 붙이면? (user 2026-09-11)

거래 목록은 exports/AIS02_LIVE_2Y_TRADES.csv (2x · SL_REF_LEV 9 · $100) 그대로 쓰고,
각 진입 시점의 직전 2일(2,880봉) 상승률만 1분봉에서 새로 계산한다.
  ret2d   = 진입가 / 2일 전 종가 - 1          (단순 2일 수익률)
  fromlow = 진입가 / 2일 최저가 - 1           (2일 저점 대비 상승)
두 정의 모두 진입 시점 이전 데이터만 쓴다.
"""
import os, sys, math, glob
import numpy as np
import pandas as pd
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = chr(36)
BARS_2D = 2 * 24 * 60

tr = pd.read_csv(os.path.join(ROOT, "exports", "AIS02_LIVE_2Y_TRADES.csv"))
print("거래 %s건 · %d종목 · %s ~ %s" % (format(len(tr), ','), tr.symbol.nunique(),
      tr.entry_at.min()[:10], tr.entry_at.max()[:10]))

ret2d = np.full(len(tr), np.nan); fromlow = np.full(len(tr), np.nan)
miss = 0
for sym, g in tr.groupby("symbol"):
    p = os.path.join(ROOT, "data", "binance_1m", sym + ".npz")
    if not os.path.exists(p):
        miss += len(g); continue
    z = np.load(p)
    ts, c, l = z["ts"], z["c"], z["l"]
    idx = np.searchsorted(ts, g.entry_ts.values)
    for k, (row_i, i) in enumerate(zip(g.index, idx)):
        if i >= len(ts) or ts[i] != g.entry_ts.iloc[k] or i < BARS_2D:
            continue
        e = tr.at[row_i, "entry"]
        ret2d[row_i] = (e / c[i - BARS_2D] - 1) * 100
        fromlow[row_i] = (e / l[i - BARS_2D:i].min() - 1) * 100
tr["ret2d"] = ret2d; tr["fromlow"] = fromlow
ok = tr.ret2d.notna()
print("2일 상승률 계산 %s건 (데이터 부족 %d)" % (format(int(ok.sum()), ','), int((~ok).sum()) + miss))
tr = tr[ok].copy()

def stat(g):
    if len(g) < 30: return None
    p = g.pnl.values
    sd = p.std(ddof=1) if len(p) > 1 else 0
    t = p.mean() / (sd / math.sqrt(len(p))) if sd else 0
    days = g.entry_at.str[:10].nunique()
    return dict(n=len(p), win=100.0 * (p > 0).mean(), mu=p.mean(), tot=p.sum(),
                t=t, perday=p.sum() / max(1, days))

def show(g, label):
    s = stat(g)
    if not s:
        print("  %-26s %6d건  (표본 부족)" % (label, len(g))); return
    print("  %-26s %6d %6.1f%% %+8.2f %+10.0f %+7.2f %+9.2f"
          % (label, s['n'], s['win'], s['mu'], s['tot'], s['t'], s['perday']))

HDR = "  %-26s %6s %7s %8s %10s %7s %9s" % ("조건", "거래", "승률", "건당"+D, "합계"+D, "t", "일평균"+D)
CUT = "2026-03-01"          # 뒤 6개월을 홀드아웃으로

for col, name in (("ret2d", "단순 2일 수익률"), ("fromlow", "2일 저점 대비 상승")):
    print()
    print("=" * 84)
    print("[%s] 분포: 중앙 %+.1f%% · 75%% %+.1f%% · 90%% %+.1f%% · 30%%+ 비율 %.1f%%"
          % (name, tr[col].median(), tr[col].quantile(.75), tr[col].quantile(.9),
             100.0 * (tr[col] >= 30).mean()))
    print("=" * 84)
    for tag, sub in (("전체 24개월", tr), ("앞 18개월", tr[tr.entry_at < CUT]), ("뒤 6개월 홀드아웃", tr[tr.entry_at >= CUT])):
        print("\n  -- %s --" % tag)
        print(HDR); print("  " + "-" * 80)
        show(sub, "기준선 (조건 없음)")
        for th in (10, 20, 30, 50, 100):
            show(sub[sub[col] >= th], "%s >= %d%%" % (col, th))
        show(sub[sub[col] < 30], "%s < 30%% (반대)" % col)
        print("  " + "-" * 80)
        print("  구간별:")
        for lo, hi in ((-999, 0), (0, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 9999)):
            show(sub[(sub[col] >= lo) & (sub[col] < hi)],
                 "   %s ~ %s" % ("" if lo == -999 else "%d%%" % lo, "" if hi == 9999 else "%d%%" % hi))

tr.to_csv(os.path.join(ROOT, "exports", "AIS02_LIVE_2Y_TRADES_2DRISE.csv"), index=False)
