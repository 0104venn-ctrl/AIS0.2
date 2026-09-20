# -*- coding: utf-8 -*-
"""Binance UM 선물 2년치 1분봉 + 펀딩비 아카이브 다운로드.

Bybit REST API 는 국가 차단(403)이고 Bybit 벌크 아카이브는 틱 데이터라
667종목 2년이면 수백 GB 가 된다. Binance 아카이브만 1분봉을 직접 준다.

저장: data/binance_1m/{SYM}.npz   (ts, o, h, l, c, qv, tbqv)  ← 09-03 연구와 동일 포맷
      data/binance_funding/{SYM}.npz (ts, rate)
재실행 안전: 완성된 종목은 건너뛴다.

사용: python scripts/dl_binance_2y.py [--months 24] [--workers 10]
"""
import argparse
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYMS_TXT = os.path.join(ROOT, "data", "binance_syms.txt")
K_URL = ("https://data.binance.vision/data/futures/um/monthly/klines/"
         "{sym}/1m/{sym}-1m-{m}.zip")
F_URL = ("https://data.binance.vision/data/futures/um/monthly/fundingRate/"
         "{sym}/{sym}-fundingRate-{m}.zip")


def months_back(n):
    y, m, out = date.today().year, date.today().month, []
    for _ in range(n):                       # 이번 달은 미완성이라 지난달부터
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        out.append(f"{y:04d}-{m:02d}")
    return sorted(out)


def get(url):
    """(bytes | None) — 404 는 '그 달엔 상장 전' 이라 정상, None 은 실패."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for _ in range(4):
        try:
            return urllib.request.urlopen(req, timeout=180).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            time.sleep(3)
        except Exception:
            time.sleep(3)
    return None


def rows_klines(raw):
    zf = zipfile.ZipFile(io.BytesIO(raw))
    out = []
    for line in zf.read(zf.namelist()[0]).decode().splitlines():
        p = line.split(",")
        if len(p) < 11 or not p[0].isdigit():
            continue
        out.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]),
                    float(p[4]), float(p[7]), float(p[10])))
    return out


def rows_funding(raw):
    zf = zipfile.ZipFile(io.BytesIO(raw))
    out = []
    for line in zf.read(zf.namelist()[0]).decode().splitlines():
        p = line.split(",")
        if len(p) < 3 or not p[0].isdigit():
            continue
        out.append((int(p[0]), float(p[2])))
    return out


def one(sym, months, kdir, fdir):
    kpath, fpath = os.path.join(kdir, f"{sym}.npz"), os.path.join(fdir, f"{sym}.npz")
    note = []

    if not os.path.exists(kpath):
        parts, n = [], 0
        for m in months:
            raw = get(K_URL.format(sym=sym, m=m))
            if raw is None:
                return sym, "ERROR klines"
            if raw:
                r = rows_klines(raw)
                if r:
                    parts.append(np.array(r, dtype=np.float64))
                    n += len(r)
        if n < 5000:
            note.append(f"thin({n})")
        else:
            a = np.concatenate(parts)
            a = a[np.argsort(a[:, 0], kind="stable")]
            np.savez_compressed(
                kpath, ts=a[:, 0].astype(np.int64), o=a[:, 1], h=a[:, 2],
                l=a[:, 3], c=a[:, 4], qv=a[:, 5], tbqv=a[:, 6])
            note.append(f"{n:,}봉")
    else:
        note.append("k:cached")

    if not os.path.exists(fpath):
        acc = []
        for m in months:
            raw = get(F_URL.format(sym=sym, m=m))
            if raw is None:
                return sym, "ERROR funding"
            if raw:
                acc.extend(rows_funding(raw))
        if acc:
            acc.sort()
            a = np.array(acc)
            np.savez_compressed(fpath, ts=a[:, 0].astype(np.int64), rate=a[:, 1])
            note.append(f"펀딩{len(acc)}")
    else:
        note.append("f:cached")
    return sym, " ".join(note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    months = months_back(a.months)
    kdir = os.path.join(ROOT, "data", "binance_1m")
    fdir = os.path.join(ROOT, "data", "binance_funding")
    os.makedirs(kdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    syms = [s.strip() for s in open(SYMS_TXT, encoding="utf-8") if s.strip()]
    print(f"{len(syms)}종목 · {months[0]} ~ {months[-1]} ({len(months)}개월) "
          f"· 워커 {a.workers}", flush=True)

    t0, done, errs = time.time(), 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        for sym, st in ex.map(lambda s: one(s, months, kdir, fdir), syms):
            done += 1
            if st.startswith("ERROR"):
                errs.append(sym)
            if done % 25 == 0 or st.startswith("ERROR"):
                gb = sum(os.path.getsize(os.path.join(kdir, f))
                         for f in os.listdir(kdir)) / 1e9
                print(f"[{time.time()-t0:6.0f}s] {done}/{len(syms)} "
                      f"{gb:.2f}GB  {sym} {st}", flush=True)
    print(f"완료 {time.time()-t0:.0f}s · 실패 {len(errs)}종목 {errs[:10]}", flush=True)


if __name__ == "__main__":
    main()
