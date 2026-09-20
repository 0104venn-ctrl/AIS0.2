# Data

| Path | What | Size | In repo |
|---|---|---|---|
| `binance_1m/{SYMBOL}.npz` | 24 months of 1-minute klines, Binance USDT-M perps. Arrays: `ts, o, h, l, c, qv, tbqv` (quote volume and taker-buy quote volume) | ~10 GB | no — run `scripts/dl_binance_2y.py` |
| `binance_syms.txt` | symbol list the downloader uses | | yes |
| `binance_daily_1m.json` | daily 1m bars (with taker-buy volume), one JSON keyed `SYMBOL_DATE`, for trades after 2026-08-31, where the npz set ends | 3 MB | yes |
| `live_windows/ais02b_k1.json`, `archive_k1.json` | for each of the 409 Bybit live trades: the trade record plus 231 one-minute bars (200 before entry, entry, 30 after) from Bybit | 6 MB | yes |
| `live_windows/ais02b_oi.json` | per-trade open interest (5-min, prior 24 h) and funding history (prior 3 days) from Bybit | 2.6 MB | yes |
| `live_windows/bybit_daily.json` | Bybit daily klines, 479 symbols, used for cross-sectional (breadth) features | 20 MB | yes |

Source for `binance_1m`: https://data.binance.vision (public archive, no key needed).
Bybit data was pulled from the public v5 REST API at the time of the study.
