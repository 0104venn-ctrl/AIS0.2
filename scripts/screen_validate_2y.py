# -*- coding: utf-8 -*-
"""409건 스크리닝 상위 후보를 24개월 15,733건에서 아웃샘플 검증.
backtest CSV 의 rise_bars/retrace_bars 로 감시구간(급등시작·고점·진입)을 복원한다."""
import os, math, warnings
import numpy as np, pandas as pd, ta
from scipy.stats import mannwhitneyu, spearmanr
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(ROOT, 'exports', 'AIS02_LIVE_2Y_FEATURES.csv'))
tr['win'] = (tr.pnl > 0).astype(int)

# 409건에서 방향 일치 + p 상위였던 후보 (이름, 계산 방식)
CANDS = ['ichi_conv_base|entry_m_top', 'ulcer|entry_m_top', 'mfi|entry_m_top', 'sma5_20_x|watch_min',
         'zscore60|watch_max', 'roc15|retr_mean', 'ulcer|at_entry', 'sma20_dev|watch_min', 'ema20_dev|retr_mean',
         'ao|watch_min', 'roc15|watch_min', 'bb_pctb|watch_max']
NEED = {c.split('|')[0] for c in CANDS}

def series_for(df):
    cl, hi, lo, vo = df.close, df.high, df.low, df.volume
    S = {}
    if 'ichi_conv_base' in NEED:
        ich = ta.trend.IchimokuIndicator(hi, lo, 9, 26, 52); S['ichi_conv_base'] = (ich.ichimoku_conversion_line() / ich.ichimoku_base_line() - 1) * 100
    if 'ulcer' in NEED: S['ulcer'] = ta.volatility.ulcer_index(cl, 14)
    if 'mfi' in NEED: S['mfi'] = ta.volume.money_flow_index(hi, lo, cl, vo, 14)
    if 'sma5_20_x' in NEED: S['sma5_20_x'] = (ta.trend.sma_indicator(cl, 5) / ta.trend.sma_indicator(cl, 20) - 1) * 100
    if 'zscore60' in NEED: S['zscore60'] = (cl - cl.rolling(60).mean()) / (cl.rolling(60).std() + 1e-12)
    if 'roc15' in NEED: S['roc15'] = ta.momentum.roc(cl, 15)
    if 'sma20_dev' in NEED: S['sma20_dev'] = (cl / ta.trend.sma_indicator(cl, 20) - 1) * 100
    if 'ema20_dev' in NEED: S['ema20_dev'] = (cl / ta.trend.ema_indicator(cl, 20) - 1) * 100
    if 'ao' in NEED: S['ao'] = ta.momentum.awesome_oscillator(hi, lo) / cl * 100
    if 'bb_pctb' in NEED: S['bb_pctb'] = ta.volatility.BollingerBands(cl, 20, 2).bollinger_pband()
    return S

def aggv(s, ls, top, E, how):
    s = np.asarray(s, float)
    if how == 'at_entry': return s[E - 1]
    if how == 'entry_m_top': return s[E - 1] - s[top]
    w = s[ls:E]; w = w[~np.isnan(w)]
    rt = s[top + 1:E]; rt = rt[~np.isnan(rt)]
    if how == 'watch_min': return w.min() if len(w) else np.nan
    if how == 'watch_max': return w.max() if len(w) else np.nan
    if how == 'retr_mean': return rt.mean() if len(rt) else np.nan
    return np.nan

rows = []
LOOK = 200
for sym, g in tr.groupby('symbol'):
    p = os.path.join(ROOT, 'data', 'binance_1m', sym + '.npz')
    if not os.path.exists(p): continue
    z = np.load(p); ts = z['ts']
    idx = np.searchsorted(ts, g.entry_ts.values)
    for k, (ri, i) in enumerate(zip(g.index, idx)):
        if i >= len(ts) or ts[i] != g.entry_ts.iloc[k] or i < LOOK: continue
        rb, rtb = int(g.rise_bars.iloc[k]), int(g.retrace_bars.iloc[k])
        E = LOOK; top = E - rtb; ls = top - rb
        if ls < 5 or top >= E - 1: continue
        sl = slice(i - LOOK, i)
        df = pd.DataFrame({'open': z['o'][sl], 'high': z['h'][sl], 'low': z['l'][sl], 'close': z['c'][sl], 'volume': z['qv'][sl]})
        try: S = series_for(df)
        except Exception: continue
        rec = {'win': int(g.win.iloc[k]), 'pnl': g.pnl.iloc[k], 'entry_at': g.entry_at.iloc[k]}
        for c in CANDS:
            base, how = c.split('|')
            try: rec[c] = aggv(S[base].values, ls, top, E, how)
            except Exception: rec[c] = np.nan
        rows.append(rec)
d = pd.DataFrame(rows)
print("24개월 검증 표본 %s건 (감시구간 복원 가능)" % format(len(d), ','))
print()
print("  %-30s %8s %8s %10s %10s %8s %6s   %s" % ("특징", "409건 방향", "24개월", "수익 중앙", "손실 중앙", "p", "rho", "판정"))
print("  " + "-" * 100)
dir409 = {'ichi_conv_base|entry_m_top': '↑', 'ulcer|entry_m_top': '↓', 'mfi|entry_m_top': '↓', 'sma5_20_x|watch_min': '↑',
          'zscore60|watch_max': '↑', 'roc15|retr_mean': '↑', 'ulcer|at_entry': '↓', 'sma20_dev|watch_min': '↑',
          'ema20_dev|retr_mean': '↑', 'ao|watch_min': '↑', 'roc15|watch_min': '↑', 'bb_pctb|watch_max': '↑'}
for c in CANDS:
    x = d[c].values; m = ~np.isnan(x)
    a, b = x[m & (d.win.values == 1)], x[m & (d.win.values == 0)]
    p = mannwhitneyu(a, b).pvalue; rho = spearmanr(x[m], d.pnl.values[m])[0]
    dr = '↑' if np.median(a) > np.median(b) else '↓'
    same = dr == dir409[c]
    verdict = "재현 (p<0.01, 방향 일치)" if (same and p < 0.01) else ("방향만 일치" if same else "방향 반대")
    print("  %-30s %8s %8s %10.3f %10.3f %8.1e %+6.3f   %s" % (c[:30], dir409[c], dr, np.median(a), np.median(b), p, rho, verdict))
