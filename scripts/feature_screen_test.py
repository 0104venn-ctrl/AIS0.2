# -*- coding: utf-8 -*-
"""879개 특징 전수 검정 + FDR 보정 + 순열 대조군 + 기간 안정성."""
import io, os, math, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
np.random.seed(7)

df = pd.read_csv(os.path.join(ROOT, 'exports', 'FEATURE_SCREEN_409_MATRIX.csv'))
meta = {'id', 'sym', 'pnl', 'pct', 'reason', 'win', 'period'}
feats = [c for c in df.columns if c not in meta and df[c].notna().mean() >= 0.5 and df[c].nunique() > 5]
W, L = df[df.win == 1], df[df.win == 0]
print("거래 %d · 검정 특징 %d개 (결측 50%%↑·상수 제외) · 수익 %d / 손실 %d\n" % (len(df), len(feats), len(W), len(L)))

def screen(labels, target_pct):
    """labels: win(0/1) 배열, target_pct: pct 배열 → 특징별 (p_mw, rho, p_rho)"""
    out = []
    for c in feats:
        x = df[c].values
        m = ~np.isnan(x)
        a, b = x[m & (labels == 1)], x[m & (labels == 0)]
        if len(a) < 20 or len(b) < 20: continue
        p_mw = mannwhitneyu(a, b, alternative='two-sided').pvalue
        rho, p_rho = spearmanr(x[m], target_pct[m])
        out.append((c, p_mw, np.median(a), np.median(b), rho, p_rho))
    return pd.DataFrame(out, columns=['feat', 'p_mw', 'med_win', 'med_loss', 'rho', 'p_rho'])

# ---- 실제 ----
R = screen(df.win.values, df.pct.values)
R['q_mw'] = multipletests(R.p_mw, method='fdr_bh')[1]
R['q_rho'] = multipletests(R.p_rho, method='fdr_bh')[1]
n05 = (R.p_mw < 0.05).sum(); n01 = (R.p_mw < 0.01).sum()
# ---- 순열 대조군: 라벨을 섞어서 같은 검정 20회 ----
perm05, perm01, perm_minp = [], [], []
for _ in range(20):
    lab = np.random.permutation(df.win.values)
    pct = df.pct.values[np.argsort(np.random.permutation(len(df)))]
    P = screen(lab, pct)
    perm05.append((P.p_mw < 0.05).sum()); perm01.append((P.p_mw < 0.01).sum()); perm_minp.append(P.p_mw.min())
print("=== 다중검정 현실 점검 ===")
print("  실제 데이터:   p<0.05 %3d개 · p<0.01 %3d개 · 최소 p %.2e" % (n05, n01, R.p_mw.min()))
print("  라벨 섞음(20회 평균): p<0.05 %5.1f개 · p<0.01 %5.1f개 · 최소 p %.2e"
      % (np.mean(perm05), np.mean(perm01), np.median(perm_minp)))
print("  → 실제가 대조군보다 %s" % ("많음 = 신호 있을 가능성" if n05 > np.mean(perm05) * 1.5 else "비슷함 = 대부분 우연"))
print("  FDR 5%% 통과 (q<0.05): Mann-Whitney %d개 · Spearman %d개" % ((R.q_mw < 0.05).sum(), (R.q_rho < 0.05).sum()))
print()

# ---- 상위 30 ----
R = R.sort_values('p_mw')
print("=== p값 상위 30 (수익 중앙값 vs 손실 중앙값) ===")
print("  %-34s %9s %9s %10s %10s %7s %8s" % ("특징", "p", "q(FDR)", "수익 중앙", "손실 중앙", "rho", "방향"))
print("  " + "-" * 92)
for _, r in R.head(30).iterrows():
    d = "수익↑" if r.med_win > r.med_loss else "수익↓"
    mk = " ***" if r.q_mw < 0.05 else (" *" if r.q_mw < 0.20 else "")
    print("  %-34s %9.2e %9.3f %10.3f %10.3f %+7.3f %8s%s" % (r.feat[:34], r.p_mw, r.q_mw, r.med_win, r.med_loss, r.rho, d, mk))

# ---- 기간 안정성 (1기 vs 2기) 상위 30 ----
print()
print("=== 상위 30의 기간 안정성 — 1기(08-09~26, n=%d)와 2기(08-30~09-08, n=%d)에서 같은 방향인가 ===" % ((df.period == 1).sum(), (df.period == 2).sum()))
print("  %-34s %10s %10s %6s" % ("특징", "1기 방향/p", "2기 방향/p", "일치"))
print("  " + "-" * 66)
stable = []
for _, r in R.head(30).iterrows():
    c = r.feat; row = []
    ok = True
    for pd_ in (1, 2):
        s = df[df.period == pd_]; x = s[c].values; m = ~np.isnan(x)
        a, b = x[m & (s.win.values == 1)], x[m & (s.win.values == 0)]
        if len(a) < 10 or len(b) < 10: row.append("   n부족"); ok = False; continue
        p = mannwhitneyu(a, b).pvalue; d = "↑" if np.median(a) > np.median(b) else "↓"
        row.append("%s %.3f" % (d, p))
    same = ok and row[0][0] == row[1][0]
    if same: stable.append(c)
    print("  %-34s %10s %10s %6s" % (c[:34], row[0], row[1], "✓" if same else "✗"))
print("\n  양 기간 방향 일치: %d / 30" % len(stable))
R.to_csv(os.path.join(ROOT, 'exports', 'FEATURE_SCREEN_409_RESULT.csv'), index=False, encoding='utf-8-sig')
pd.Series(stable).to_csv(os.path.join(ROOT, 'exports', 'FEATURE_SCREEN_409_STABLE.csv'), index=False, header=False)
