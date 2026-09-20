# -*- coding: utf-8 -*-
"""재현된 특징의 경제적 의미: 5분위별 실제 손익 (24개월, 라이브 설정 $100)."""
import os, math, warnings
import numpy as np, pandas as pd, ta
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(ROOT, 'exports', 'AIS02_LIVE_2Y_FEATURES.csv'))
rows = []; LOOK = 200
for sym, g in tr.groupby('symbol'):
    p = os.path.join(ROOT, 'data', 'binance_1m', sym + '.npz')
    if not os.path.exists(p): continue
    z = np.load(p); ts = z['ts']; c = z['c']; h = z['h']; l = z['l']
    idx = np.searchsorted(ts, g.entry_ts.values)
    for k, i in enumerate(idx):
        if i >= len(ts) or ts[i] != g.entry_ts.iloc[k] or i < LOOK: continue
        rb, rtb = int(g.rise_bars.iloc[k]), int(g.retrace_bars.iloc[k])
        E = LOOK; top = E - rtb; ls = top - rb
        if ls < 5 or top >= E - 1: continue
        cl = pd.Series(c[i-LOOK:i])
        sma20 = cl.rolling(20).mean(); dev = ((cl / sma20 - 1) * 100).values
        ul = ta.volatility.ulcer_index(cl, 14).values
        roc = ((cl / cl.shift(15) - 1) * 100).values
        w = dev[ls:E]; w = w[~np.isnan(w)]; r = roc[ls:E]; r = r[~np.isnan(r)]
        rows.append(dict(pnl=g.pnl.iloc[k], entry_at=g.entry_at.iloc[k], sym=sym,
                         sma20_min=w.min() if len(w) else np.nan, ulcer=ul[E-1], roc15_min=r.min() if len(r) else np.nan,
                         dd_watch=(cl.values[ls:E].min() / cl.values[ls:top+1].max() - 1) * 100))
d = pd.DataFrame(rows); d['win'] = (d.pnl > 0).astype(int)
d.to_csv(os.path.join(ROOT, 'exports', 'SCREEN_ECON_2Y.csv'), index=False)
D = chr(36); CUT = '2026-03-01'
def table(col, name, sub, tag):
    s = sub.dropna(subset=[col]).copy(); s['q'] = pd.qcut(s[col], 5, labels=False, duplicates='drop')
    print("  [%s] %s" % (tag, name))
    print("    %-4s %14s %7s %7s %9s %9s %7s" % ("분위", "범위", "거래", "승률", "건당"+D, "합계"+D, "t"))
    for q, g in s.groupby('q'):
        p = g.pnl.values; sd = p.std(ddof=1); t = p.mean()/(sd/math.sqrt(len(p))) if sd else 0
        print("    Q%d  %+6.2f~%+6.2f %7d %6.1f%% %+9.3f %+9.0f %+7.2f" % (q+1, g[col].min(), g[col].max(), len(g), 100*g.win.mean(), p.mean(), p.sum(), t))
    print()
print("24개월 %s건\n" % format(len(d), ','))
for col, name in (('sma20_min', '감시구간 SMA20 대비 최저 이탈 % (낮을수록 깊게 파임)'),
                  ('ulcer', '진입 시 Ulcer 지수 (낮을수록 완만)'),
                  ('roc15_min', '감시구간 15분 ROC 최저 %'),
                  ('dd_watch', '감시구간 낙폭 % (고점→최저)')):
    table(col, name, d, '전체 24개월')
    table(col, name, d[d.entry_at >= CUT], '뒤 6개월 홀드아웃')
