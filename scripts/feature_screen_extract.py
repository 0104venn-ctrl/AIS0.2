# -*- coding: utf-8 -*-
"""Bybit 실거래 409건 — 감시 전체 구간의 지표 전수 추출 (user 2026-09-19).

각 거래: 진입 전 200분 1분봉 + Binance 테이커 + Bybit OI/펀딩 + 일봉 횡단면.
시계열 지표 하나당 10가지 집계:
  at_entry     진입 직전 봉 값
  watch_mean/max/min/slope   감시구간(급등 시작→진입) 통계
  pump_mean    급등구간(시작→고점) 평균
  retr_mean    되돌림구간(고점→진입) 평균
  top_val      고점 봉 값
  entry_m_top  진입값 − 고점값
  retr_m_pump  되돌림 평균 − 급등 평균
전부 진입 이전 데이터만 사용 (인과).
"""
import io, os, sys, json, glob, bisect, math, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LW = os.path.join(ROOT, 'data', 'live_windows')
_DAILY1M = json.load(open(os.path.join(ROOT, 'data', 'binance_daily_1m.json')))
import numpy as np
import pandas as pd
import ta
warnings.filterwarnings('ignore')

NPZ = os.path.join(ROOT, 'data', 'binance_1m')
_A = json.load(io.open(os.path.join(LW, 'archive_k1.json'), encoding='utf-8')); _C = json.load(io.open(os.path.join(LW, 'ais02b_k1.json'), encoding='utf-8'))
for _o in _A: _o['period'] = 1
for _o in _C: _o['period'] = 2
T = _A + _C
OI = {r['id']: r for r in json.load(io.open(os.path.join(LW, 'ais02b_oi.json'), encoding='utf-8'))}
DAILY = json.load(io.open(os.path.join(LW, 'bybit_daily.json'), encoding='utf-8'))

# ---------- Binance 테이커 ----------
_npz = {}
def bn_bars(sym, t0, t1):
    out = {}
    p = os.path.join(NPZ, sym + '.npz')
    if os.path.exists(p):
        if sym not in _npz:
            z = np.load(p); _npz[sym] = (z['ts'], z['qv'], z['tbqv'])
        ts, qv, tb = _npz[sym]
        i0, i1 = bisect.bisect_left(ts, t0), bisect.bisect_right(ts, t1)
        for i in range(i0, i1): out[int(ts[i])] = (float(tb[i]), float(qv[i]))
    for key, rows in _DAILY1M.items():
        if not key.startswith(sym + '_'): continue
        for r in rows:
            if t0 <= r[0] <= t1: out[r[0]] = (r[8], r[6])
    return out

# ---------- 일봉 횡단면 (진입일 전날까지) ----------
_day_close = {}
for s, k in DAILY.items():
    _day_close[s] = {int(b[0] // 86400000): b[4] for b in k}
_days = sorted({d for m in _day_close.values() for d in m})
def xsec(day):
    """day 전날 기준: 7일 상승 종목 비율, 7일 중앙 수익률, BTC 7일"""
    d1 = day - 1; d7 = day - 8
    rets = []
    for s, m in _day_close.items():
        if d1 in m and d7 in m and m[d7] > 0: rets.append(m[d1] / m[d7] - 1)
    if len(rets) < 50: return np.nan, np.nan, np.nan
    b = _day_close.get('BTCUSDT', {})
    btc = (b[d1] / b[d7] - 1) * 100 if (d1 in b and d7 in b) else np.nan
    return 100.0 * np.mean([r > 0 for r in rets]), 100.0 * np.median(rets), btc

def slope(y):
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    if len(y) < 3: return np.nan
    x = np.arange(len(y)); return float(np.polyfit(x, y, 1)[0])

def agg(series, ls, top, E, name, feats):
    """시계열 하나를 10개 집계로."""
    s = np.asarray(series, float)
    w = s[ls:E]; pu = s[ls:top + 1]; rt = s[top + 1:E]
    def nm(a): a = a[~np.isnan(a)]; return a
    w_, pu_, rt_ = nm(w), nm(pu), nm(rt)
    feats[name + '|at_entry'] = s[E - 1] if E >= 1 else np.nan
    feats[name + '|watch_mean'] = w_.mean() if len(w_) else np.nan
    feats[name + '|watch_max'] = w_.max() if len(w_) else np.nan
    feats[name + '|watch_min'] = w_.min() if len(w_) else np.nan
    feats[name + '|watch_slope'] = slope(w)
    feats[name + '|pump_mean'] = pu_.mean() if len(pu_) else np.nan
    feats[name + '|retr_mean'] = rt_.mean() if len(rt_) else np.nan
    feats[name + '|top_val'] = s[top]
    feats[name + '|entry_m_top'] = (s[E - 1] - s[top]) if not (np.isnan(s[E - 1]) or np.isnan(s[top])) else np.nan
    feats[name + '|retr_m_pump'] = (rt_.mean() - pu_.mean()) if (len(rt_) and len(pu_)) else np.nan

def features(o):
    b = o['bars']; ets = o['ets'] - o['ets'] % 60000
    idx = {bar[0]: i for i, bar in enumerate(b)}
    if ets not in idx: return None
    E = idx[ets]
    if E < 150: return None
    df = pd.DataFrame(b[:E], columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    H, L, C, V, O = df.high.values, df.low.values, df.close.values, df.volume.values, df.open.values
    top = int(np.argmax(H[max(0, E - 130):])) + max(0, E - 130)
    ls = int(np.argmin(L[max(0, top - 120):top + 1])) + max(0, top - 120)
    if top - ls < 3 or E - top < 2: return None
    F = {}
    # ===== 스칼라 (구조) =====
    F['s|rise_pct'] = (H[top] / L[ls] - 1) * 100
    F['s|rise_bars'] = top - ls
    F['s|top_to_entry'] = E - top
    F['s|retrace_pct'] = (H[top] - C[E - 1]) / (H[top] - L[ls]) * 100
    F['s|pre_trend'] = (C[ls] / C[0] - 1) * 100 if ls > 5 else np.nan
    F['s|range_pos'] = (C[E - 1] - L.min()) / (H.max() - L.min()) * 100
    F['s|rise_speed'] = F['s|rise_pct'] / max(1, top - ls)
    F['s|retr_speed'] = F['s|retrace_pct'] / max(1, E - top)
    F['s|overshoot'] = (L[top + 1:E].min() / L[ls] - 1) * 100 if E > top + 1 else np.nan
    F['s|n_legs_7pct'] = 0
    r = np.diff(np.log(C)); F['s|n_swings_2pct'] = int((np.abs(r) > 0.02).sum())
    F['s|max_1m_ret'] = float(np.nanmax(r) * 100); F['s|min_1m_ret'] = float(np.nanmin(r) * 100)
    F['s|entry_bar_body'] = (C[E - 1] - O[E - 1]) / O[E - 1] * 100
    F['s|entry_bar_range'] = (H[E - 1] - L[E - 1]) / O[E - 1] * 100
    F['s|entry_upper_wick'] = (H[E - 1] - max(O[E - 1], C[E - 1])) / O[E - 1] * 100
    F['s|entry_lower_wick'] = (min(O[E - 1], C[E - 1]) - L[E - 1]) / O[E - 1] * 100
    F['s|consec_down'] = 0
    k = E - 1
    while k > 0 and C[k] < C[k - 1]: F['s|consec_down'] += 1; k -= 1
    F['s|red_ratio_retr'] = float(np.mean(C[top + 1:E] < O[top + 1:E])) if E > top + 1 else np.nan
    F['s|red_ratio_pump'] = float(np.mean(C[ls:top + 1] < O[ls:top + 1]))
    F['s|gap_at_entry'] = (O[E - 1] / C[E - 2] - 1) * 100 if E >= 2 else np.nan
    lr = np.log(C[ls:E]); F['s|autocorr1_watch'] = float(pd.Series(np.diff(lr)).autocorr(1)) if E - ls > 10 else np.nan
    F['s|skew_watch'] = float(pd.Series(np.diff(lr)).skew()) if E - ls > 10 else np.nan
    F['s|kurt_watch'] = float(pd.Series(np.diff(lr)).kurt()) if E - ls > 10 else np.nan
    # 허스트 (단순 R/S, 감시구간)
    seg = np.diff(lr)
    if len(seg) > 20:
        m = seg.mean(); y = np.cumsum(seg - m); R = y.max() - y.min(); S = seg.std()
        F['s|hurst_rs'] = math.log(R / S) / math.log(len(seg)) if S > 0 and R > 0 else np.nan
    else: F['s|hurst_rs'] = np.nan
    # ===== 시계열 지표 (ta) =====
    S = {}
    cl, hi, lo, vo = df.close, df.high, df.low, df.volume
    for n in (5, 10, 20, 50):
        S['sma%d_dev' % n] = (cl / ta.trend.sma_indicator(cl, n) - 1) * 100
        S['ema%d_dev' % n] = (cl / ta.trend.ema_indicator(cl, n) - 1) * 100
    S['sma5_20_x'] = (ta.trend.sma_indicator(cl, 5) / ta.trend.sma_indicator(cl, 20) - 1) * 100
    S['ema10_50_x'] = (ta.trend.ema_indicator(cl, 10) / ta.trend.ema_indicator(cl, 50) - 1) * 100
    mac = ta.trend.MACD(cl); S['macd'] = mac.macd() / cl * 100; S['macd_sig'] = mac.macd_signal() / cl * 100; S['macd_hist'] = mac.macd_diff() / cl * 100
    adx = ta.trend.ADXIndicator(hi, lo, cl, 14); S['adx'] = adx.adx(); S['di_diff'] = adx.adx_pos() - adx.adx_neg()
    ar = ta.trend.AroonIndicator(hi, lo, 25); S['aroon_osc'] = ar.aroon_up() - ar.aroon_down()
    S['psar_dist'] = (cl / ta.trend.PSARIndicator(hi, lo, cl).psar() - 1) * 100
    S['vortex_diff'] = ta.trend.VortexIndicator(hi, lo, cl, 14).vortex_indicator_diff()
    S['trix'] = ta.trend.trix(cl, 15)
    S['dpo'] = ta.trend.dpo(cl, 20) / cl * 100
    S['kst'] = ta.trend.kst(cl)
    S['cci'] = ta.trend.CCIIndicator(hi, lo, cl, 20).cci()
    S['mass'] = ta.trend.MassIndex(hi, lo).mass_index()
    ich = ta.trend.IchimokuIndicator(hi, lo, 9, 26, 52); S['ichi_conv_base'] = (ich.ichimoku_conversion_line() / ich.ichimoku_base_line() - 1) * 100
    S['rsi7'] = ta.momentum.rsi(cl, 7); S['rsi14'] = ta.momentum.rsi(cl, 14); S['rsi28'] = ta.momentum.rsi(cl, 28)
    st = ta.momentum.StochasticOscillator(hi, lo, cl, 14, 3); S['stoch_k'] = st.stoch(); S['stoch_d'] = st.stoch_signal()
    S['stoch_rsi'] = ta.momentum.stochrsi(cl, 14) * 100
    S['willr'] = ta.momentum.williams_r(hi, lo, cl, 14)
    for n in (1, 3, 5, 10, 15, 30, 60): S['roc%d' % n] = ta.momentum.roc(cl, n)
    S['tsi'] = ta.momentum.tsi(cl); S['uo'] = ta.momentum.ultimate_oscillator(hi, lo, cl)
    S['ao'] = ta.momentum.awesome_oscillator(hi, lo) / cl * 100
    S['ppo'] = ta.momentum.ppo(cl); S['kama_dev'] = (cl / ta.momentum.kama(cl) - 1) * 100
    S['atr_pct'] = ta.volatility.average_true_range(hi, lo, cl, 14) / cl * 100
    bb = ta.volatility.BollingerBands(cl, 20, 2); S['bb_pctb'] = bb.bollinger_pband(); S['bb_width'] = bb.bollinger_wband()
    kc = ta.volatility.KeltnerChannel(hi, lo, cl, 20); S['kc_pos'] = (cl - kc.keltner_channel_lband()) / (kc.keltner_channel_hband() - kc.keltner_channel_lband() + 1e-12)
    dc = ta.volatility.DonchianChannel(hi, lo, cl, 20); S['dc_pos'] = dc.donchian_channel_pband()
    S['ulcer'] = ta.volatility.ulcer_index(cl, 14)
    lr_s = np.log(cl).diff()
    for n in (10, 30, 60): S['hv%d' % n] = lr_s.rolling(n).std() * 100
    S['parkinson20'] = np.sqrt((np.log(hi / lo) ** 2).rolling(20).mean() / (4 * math.log(2))) * 100
    S['range_pct'] = (hi - lo) / cl * 100
    S['range_ratio20'] = S['range_pct'] / S['range_pct'].rolling(20).mean()
    S['vol_ratio20'] = vo / vo.rolling(20).mean(); S['vol_ratio60'] = vo / vo.rolling(60).mean()
    S['vol_z60'] = (vo - vo.rolling(60).mean()) / (vo.rolling(60).std() + 1e-12)
    S['obv_slope20'] = ta.volume.on_balance_volume(cl, vo).diff(20) / (vo.rolling(20).sum() + 1e-12)
    S['cmf'] = ta.volume.chaikin_money_flow(hi, lo, cl, vo, 20)
    S['mfi'] = ta.volume.money_flow_index(hi, lo, cl, vo, 14)
    S['force'] = ta.volume.force_index(cl, vo, 13) / (cl * vo.rolling(13).mean() + 1e-12)
    S['eom'] = ta.volume.ease_of_movement(hi, lo, vo, 14)
    S['vpt_slope'] = ta.volume.volume_price_trend(cl, vo).diff(10) / (vo.rolling(10).sum() + 1e-12)
    S['nvi'] = ta.volume.negative_volume_index(cl, vo).pct_change(20) * 100
    vwap = (cl * vo).cumsum() / (vo.cumsum() + 1e-12); S['vwap_dev'] = (cl / vwap - 1) * 100
    S['adi_slope'] = ta.volume.acc_dist_index(hi, lo, cl, vo).diff(20) / (vo.rolling(20).sum() + 1e-12)
    S['up_dn_vol20'] = (vo.where(cl >= df.open, 0).rolling(20).sum()) / (vo.where(cl < df.open, 0).rolling(20).sum() + 1e-12)
    S['dd_from_max'] = (cl / cl.cummax() - 1) * 100
    S['body_pct'] = (cl - df.open).abs() / df.open * 100
    S['wick_ratio'] = ((hi - lo) - (cl - df.open).abs()) / ((hi - lo) + 1e-12)
    S['lr_slope10'] = cl.rolling(10).apply(lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=True) / cl * 100
    S['lr_slope30'] = cl.rolling(30).apply(lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=True) / cl * 100
    S['zscore20'] = (cl - cl.rolling(20).mean()) / (cl.rolling(20).std() + 1e-12)
    S['zscore60'] = (cl - cl.rolling(60).mean()) / (cl.rolling(60).std() + 1e-12)
    S['pct_rank60'] = cl.rolling(60).rank(pct=True) * 100
    # ===== Binance 테이커 시계열 =====
    bb_ = bn_bars(o['sym'], b[0][0], b[E - 1][0])
    if bb_:
        buy = np.array([bb_.get(bar[0], (np.nan, np.nan))[0] for bar in b[:E]])
        tot = np.array([bb_.get(bar[0], (np.nan, np.nan))[1] for bar in b[:E]])
        br = pd.Series(buy / np.where(tot > 0, tot, np.nan) * 100)
        S['tk_buy1'] = br; S['tk_buy5'] = br.rolling(5).mean(); S['tk_buy15'] = br.rolling(15).mean(); S['tk_buy60'] = br.rolling(60).mean()
        delta = pd.Series(2 * buy - tot); S['cvd_norm'] = delta.cumsum() / (pd.Series(tot).cumsum() + 1e-12) * 100
        S['delta_z20'] = (delta - delta.rolling(20).mean()) / (delta.rolling(20).std() + 1e-12)
        S['delta_ratio5'] = delta.rolling(5).sum() / (pd.Series(tot).rolling(5).sum() + 1e-12) * 100
        F['s|bn_cov'] = 1
    else: F['s|bn_cov'] = 0
    # ===== OI (5분) — 진입 전 24h =====
    oi = OI.get(o['id'], {}).get('oi') or []
    if len(oi) >= 24:
        ots = [x[0] for x in oi]; ov = np.array([x[1] for x in oi], float)
        cur = ov[-1]
        def oi_at(min_ago):
            i = bisect.bisect_right(ots, ets - min_ago * 60000) - 1
            return ov[i] if i >= 0 else np.nan
        for m, lab in ((30, '30m'), (60, '1h'), (240, '4h'), (1440, '24h')):
            F['oi|chg_%s' % lab] = (cur / oi_at(m) - 1) * 100 if oi_at(m) > 0 else np.nan
        F['oi|z24h'] = (cur - ov.mean()) / (ov.std() + 1e-12)
        F['oi|slope_4h'] = slope(ov[-48:]) / (ov.mean() + 1e-12) * 100
        # 급등구간·되돌림구간 OI 변화
        t_ls, t_top = b[ls][0], b[top][0]
        i_ls = bisect.bisect_right(ots, t_ls) - 1; i_top = bisect.bisect_right(ots, t_top) - 1
        if i_ls >= 0 and i_top > i_ls:
            F['oi|chg_pump'] = (ov[i_top] / ov[i_ls] - 1) * 100
            F['oi|chg_retr'] = (cur / ov[i_top] - 1) * 100
            # 4분면: 급등 중 OI↑=신규롱 / 되돌림 중 OI↓=롱청산
            F['oi|pump_up_retr_down'] = float(F['oi|chg_pump'] > 0 and F['oi|chg_retr'] < 0)
    # ===== 펀딩 =====
    fd = OI.get(o['id'], {}).get('fund') or []
    if len(fd) >= 1:
        fr = np.array([x[1] for x in fd], float) * 100
        F['fund|last'] = fr[-1]; F['fund|mean3'] = fr[-3:].mean(); F['fund|mean9'] = fr[-9:].mean() if len(fr) >= 9 else np.nan
        F['fund|z'] = (fr[-1] - fr.mean()) / (fr.std() + 1e-12) if len(fr) >= 4 else np.nan
        F['fund|delta'] = fr[-1] - fr[-2] if len(fr) >= 2 else np.nan
        F['fund|hours_since'] = (ets - fd[-1][0]) / 3600000.0
    # ===== 횡단면 (전일까지) =====
    F['xs|breadth7'], F['xs|mkt7'], F['xs|btc7'] = xsec(int(ets // 86400000))
    # ===== 시간 =====
    import datetime as dt
    d = dt.datetime.utcfromtimestamp(ets / 1000)
    F['t|hour_utc'] = d.hour; F['t|dow'] = d.weekday(); F['t|min_of_hour'] = d.minute
    F['t|us_session'] = float(13 <= d.hour < 21); F['t|asia_session'] = float(0 <= d.hour < 8)
    # ===== 집계 =====
    for name, ser in S.items():
        agg(ser.values if hasattr(ser, 'values') else ser, ls, top, E, name, F)
    F['id'] = o['id']; F['sym'] = o['sym']; F['pnl'] = o['pnl']; F['pct'] = o['pct']; F['reason'] = o['reason']; F['win'] = int(o['pnl'] > 0)
    F['period'] = o['period']
    return F

rows = []
for i, o in enumerate(T):
    try:
        f = features(o)
        if f: rows.append(f)
    except Exception as e:
        print("  #%s %s 실패: %s" % (o.get('id'), o.get('sym'), str(e)[:60]))
    if (i + 1) % 100 == 0: print("  %d/%d" % (i + 1, len(T)), flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, 'exports', 'FEATURE_SCREEN_409_MATRIX.csv'), index=False, encoding='utf-8-sig')
meta = {'id', 'sym', 'pnl', 'pct', 'reason', 'win', 'period'}
nf = [c for c in df.columns if c not in meta]
print("거래 %d건 · 특징 %d개 · 수익 %d / 손실 %d" % (len(df), len(nf), df.win.sum(), (df.win == 0).sum()))
print("  결측 50%% 초과 특징 %d개" % sum(df[c].isna().mean() > 0.5 for c in nf))
