# AIS0.2 — post-mortem of a crypto pump-fade short strategy

**This strategy has no edge. This repository documents how I found that out.**

I built a bot that shorts crypto perpetuals after a 7%+ pump fully retraces, ran it live on
Bybit and OKX for 37 days (500 trades), and then spent a month trying to falsify it with
everything I had: a 24-month backtest, a 52,199-leg outcome study, an 867-feature screen with
a permutation control, and a dozen entry/exit variants. Everything points the same way.

Full write-up with charts: [`report/post-mortem.html`](report/post-mortem.html)

## Live result

| | |
|---|---|
| Period | 2026-08-03 → 2026-09-08 (37 days) |
| Trades | 500 (OKX 91, Bybit 409) |
| Win rate | **50.0%** (250 / 250) |
| Avg win / avg loss (on margin) | +26.4% / −26.3% |
| Net P&L | +$73.22 (profit factor 1.01) |
| t-statistic | **0.05** |
| Max drawdown | $1,287 (from peak +$1,360) |

All P&L is exchange-confirmed (`closed-pnl` API), fees and funding included.
`exports/LIVE_TRADES_500.csv` has every trade.

## The strategy (`signal.py`)

On 1-minute closes, scanned once per minute over a 400-bar trailing window:

1. **Pivots** — 4-bar fractal highs/lows on log closes, same-kind pivots merged to the extreme.
2. **Pump** — an up-leg (low pivot → high pivot) with rise ≥ 7% (log-space).
3. **Entry** — the first bar after the top (top+4 … top+120) whose close is at or below the
   leg's starting price, i.e. a 100% retrace. Short at market.
4. **Exit** — take-profit at 0.6 × pump size below entry; stop at min(1.0 × pump size,
   70% of the distance to liquidation computed at 9× leverage); forced exit after 30 bars.
   TP/SL are attached to the exchange order.

`signal.py` is the detection and exit logic lifted verbatim from the production bot
(the bot itself — exchange auth, order routing, DB, health checks — is not included).
`python signal.py` runs a synthetic smoke test.

## What was tested, and what it showed

| Test | Data | Result | Script |
|---|---|---|---|
| Live config backtest | Binance 1m, 805 symbols, Sep 2024 – Aug 2026 | 15,733 trades, −$4,921, win 44.5%, daily t −2.82, DSR 0 | `backtest_ais02_live_2y.py` |
| Directional information | 1,739 signals, 30-min MAE/MFE | adverse 2.96% vs favorable 3.08%; down-first 51% | (in write-up) |
| Market regime | 24-month trades × 7-day market return | negative in every regime; worst in down markets | `ais02_regime_check.py` |
| 2-day momentum filter | 24 months | removes losses, no positive region | `ais02_2d_rise_filter.py` |
| Taker buy-ratio (15 min) | 24 months | sign flips vs. live sample | `ais02_buyratio15_check.py` |
| Entry timing | 14 variants: delay 1–10 min, wait for 0.5–2% bounce | only "wait for 2% bounce" positive overall; negative in first 18 months | `backtest_ais02_entry_timing.py` |
| Leg completion | 52,199 pumps ≥ 7% | 18.6% reach 100%; P(complete \| at X%) ≈ X%; predictable cases have the worst P&L | `leg_outcomes_2y.py` |
| Feature screen | 867 features over the full watch window, 404 live trades | 26 at p<0.05 vs. 38 under shuffled labels; FDR survivors: 0 | `feature_screen_extract.py`, `feature_screen_test.py` |
| Screen replication | top 12 → 15,417 backtest trades | 6 replicate (ρ 0.03–0.06), all one thing: "violent retrace = bad"; best quintile 48% win vs. 51.5% breakeven | `screen_validate_2y.py`, `screen_econ_2y.py` |

The one robust finding — a retrace that dips violently below its moving average is a
bad short — can trim losses. It cannot create a positive region. The signal is a coin flip
with fees.

## Reproducing

```bash
pip install numpy pandas scipy statsmodels ta

# 1. Data (~10 GB). Pulls 1-minute klines from data.binance.vision for the symbol
#    list in data/binance_syms.txt and writes data/binance_1m/{SYMBOL}.npz
python scripts/dl_binance_2y.py --months 24

# 2. Core backtest — reproduces exports/AIS02_LIVE_2Y_TRADES.csv
python scripts/backtest_ais02_live_2y.py

# 3. Any of the validation scripts (each reads exports/ and data/, writes exports/)
python scripts/leg_outcomes_2y.py
python scripts/feature_screen_extract.py && python scripts/feature_screen_test.py
```

`data/live_windows/` ships the 200-minute 1-minute windows around each of the 409 Bybit
trades plus their open-interest / funding history, so the feature screen runs without the
full 24-month download.

## Layout

```
signal.py                     the strategy, self-contained
scripts/                      backtest + every validation script listed above
exports/                      results: live trades, backtest trades, feature matrix, screens
data/live_windows/            per-trade 1m windows, OI, funding (Bybit) — 9 MB
data/binance_daily_1m/        Binance 1m with taker-buy volume for the Sep-2026 trades
data/binance_1m/              (gitignored) 24-month 1m klines — download with dl_binance_2y.py
report/post-mortem.html       the write-up
```

## Notes for a reader

- Every signal function is causal: pivots are used only after their 4-bar confirmation, and
  no feature uses data past the entry bar.
- The stop is bounded by the maintenance margin rate, not by the strategy. That means the
  reward/risk ratio is set by the exchange's liquidation math — a structural flaw.
- Comments inside the scripts are in Korean (my working language). Function and variable
  names are English.
- Symbols in `data/binance_syms.txt` include ~130 tokenized-stock perps that Binance listed
  in 2026; they were not excluded from the 24-month backtest.

## License

MIT
