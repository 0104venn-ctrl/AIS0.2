# -*- coding: utf-8 -*-
"""실거래 409건에서 나온 '직전 15분 테이커 매수비율↓ → 좋다'가 24개월 백테스트 15,733건에서도 성립하나."""
import os, math
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(ROOT, "exports", "AIS02_LIVE_2Y_TRADES.csv"))
br15 = np.full(len(tr), np.nan); br5 = np.full(len(tr), np.nan)
for sym, g in tr.groupby("symbol"):
    p = os.path.join(ROOT, "data", "binance_1m", sym + ".npz")
    if not os.path.exists(p): continue
    z = np.load(p); ts, qv, tb = z["ts"], z["qv"], z["tbqv"]
    idx = np.searchsorted(ts, g.entry_ts.values)
    for k, (ri, i) in enumerate(zip(g.index, idx)):
        if i >= len(ts) or ts[i] != g.entry_ts.iloc[k] or i < 20: continue
        q = qv[i-15:i].sum(); b = tb[i-15:i].sum()
        if q > 0: br15[ri] = b / q * 100
        q5 = qv[i-5:i].sum(); b5 = tb[i-5:i].sum()
        if q5 > 0: br5[ri] = b5 / q5 * 100
tr["br15"] = br15; tr["br5"] = br5
tr = tr[tr.br15.notna()].copy()
tr["win"] = (tr.pnl > 0).astype(int)
print("24개월 %s건 (매수비율 계산됨)" % format(len(tr), ','))
D = chr(36)
def block(sub, tag):
    W, L = sub[sub.win == 1], sub[sub.win == 0]
    p15 = mannwhitneyu(W.br15, L.br15).pvalue; p5 = mannwhitneyu(W.br5, L.br5).pvalue
    print("\n[%s] 수익 %d · 손실 %d" % (tag, len(W), len(L)))
    print("  직전15분 매수비율  수익 %.2f · 손실 %.2f · 차이 %+.2f · p=%.4f" % (W.br15.median(), L.br15.median(), W.br15.median()-L.br15.median(), p15))
    print("  직전5분  매수비율  수익 %.2f · 손실 %.2f · 차이 %+.2f · p=%.4f" % (W.br5.median(), L.br5.median(), W.br5.median()-L.br5.median(), p5))
    print("  --- 5분위 (br15 낮음→높음) ---")
    sub = sub.copy(); sub["q"] = pd.qcut(sub.br15, 5, labels=False, duplicates='drop')
    for q, g in sub.groupby("q"):
        d = g.entry_at.str[:10].nunique()
        sd = g.pnl.std(ddof=1); t = g.pnl.mean()/(sd/math.sqrt(len(g))) if sd else 0
        print("    Q%d  br15 %5.1f~%5.1f  %5d건  승률 %.1f%%  건당 %s%+.3f  합계 %s%+7.0f  t %+.2f"
              % (q+1, g.br15.min(), g.br15.max(), len(g), 100*g.win.mean(), D, g.pnl.mean(), D, g.pnl.sum(), t))
block(tr, "전체")
block(tr[tr.why == "TIME"], "타임아웃만")
CUT = "2026-03-01"
block(tr[tr.entry_at < CUT], "앞 18개월")
block(tr[tr.entry_at >= CUT], "뒤 6개월 홀드아웃")
