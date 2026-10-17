"""Backfill missing pre-cache history for all gapped symbols to 2023-01.

Efficient route: whole-market per-trade-date batch calls (2 requests/day)
for 2023-01-01..2024-12-31, keeping only rows for symbols whose cache starts
later (see /tmp/gap_symbols.csv from the audit). Merges into existing raw /
factor CSVs, then deletes affected qfq caches so fix_qfq_conflicts.py
re-derives them consistently.

Checkpointed: progress survives restarts (done-days list + part files in
/tmp/backfill_ckpt). Safe to rerun after a crash.

Usage: .venv/bin/python -u scripts/backfill_2023.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, "src")
from qfree.config import Settings
from qfree.data.tushare_client import TushareProvider

WORKERS = 12
GAP_CSV = "/tmp/gap_symbols.csv"
CKPT = "/tmp/backfill_ckpt"
RAW_COLS = ["symbol", "date", "open", "high", "low", "close", "volume",
            "amount", "ts_code", "pre_close", "change", "pct_chg"]


def _merge_symbol(job: tuple[str, pd.DataFrame, pd.DataFrame]) -> tuple[str, int]:
    symbol, raw_new, fac_new = job
    raw_path = f"data/daily/{symbol}_raw.csv"
    fac_path = f"data/adj_factor/{symbol}.csv"
    n = 0
    if raw_new is not None and len(raw_new):
        old = pd.read_csv(raw_path)
        merged = (pd.concat([old, raw_new[RAW_COLS]], ignore_index=True)
                  .drop_duplicates("date").sort_values("date"))
        merged.to_csv(raw_path, index=False)
        n = len(merged) - len(old)
    if fac_new is not None and len(fac_new):
        if os.path.exists(fac_path):
            oldf = pd.read_csv(fac_path)
            fac = (pd.concat([oldf, fac_new], ignore_index=True)
                   .drop_duplicates("date").sort_values("date"))
        else:
            fac = fac_new
        fac.to_csv(fac_path, index=False)
    return symbol, n


def main() -> int:
    started = time.time()
    os.makedirs(CKPT, exist_ok=True)
    done_path = os.path.join(CKPT, "done_days.txt")
    done = set()
    if os.path.exists(done_path):
        done = set(open(done_path).read().split())
    gaps = pd.read_csv(GAP_CSV, dtype={"symbol": str})
    cached_start = {r.symbol: r.cached_start for r in gaps.itertuples()}
    gap_syms = set(gaps["symbol"])
    print(f"gapped symbols: {len(gap_syms)} | already-fetched days: {len(done)}")

    provider = TushareProvider(Settings.from_env().require_token(),
                               api_url="https://t.xiaodefa.top/", request_timeout=30)
    sessions = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2023-01-01", "2024-12-31")))
    todo = [d for d in sessions if d.strftime("%Y-%m-%d") not in done]
    print(f"trade sessions: {len(sessions)} | to fetch: {len(todo)}")

    raws: list[pd.DataFrame] = []
    facs: list[pd.DataFrame] = []
    fails = 0
    done_f = open(done_path, "a")

    def _process(day: str, d, f) -> None:
        ds = pd.Timestamp(day).strftime("%Y-%m-%d")
        d2 = None
        if d is not None and "ts_code" in d.columns and len(d):
            d = d[d["ts_code"].str[:6].isin(gap_syms)]
            if len(d):
                d = d.rename(columns={"trade_date": "date", "vol": "volume"}).copy()
                d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
                d["symbol"] = d["ts_code"].str[:6]
                keep = d["date"] < d["symbol"].map(cached_start)
                d2 = d.loc[keep, RAW_COLS]
        f2 = None
        if f is not None and "ts_code" in f.columns and len(f):
            f = f[f["ts_code"].str[:6].isin(gap_syms)]
            if len(f):
                f2 = pd.DataFrame({
                    "date": pd.to_datetime(f["trade_date"]).dt.strftime("%Y-%m-%d"),
                    "symbol": f["ts_code"].str[:6],
                    "adj_factor": f["adj_factor"]})
                keep = f2["date"] < f2["symbol"].map(cached_start)
                f2 = f2.loc[keep, ["date", "adj_factor", "symbol"]]
        if d2 is not None and len(d2):
            raws.append(d2)
        if f2 is not None and len(f2):
            facs.append(f2)
        done.add(ds)
        done_f.write(ds + "\n")
        done_f.flush()

    # parallel per-day fetch: latency-bound sequential calls wasted ~6s/day;
    # thread pool over days is the same pattern batch_fetch.py uses.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from qfree.data.batch_fetch import _fetch_session

    fails = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_session, provider, day.strftime("%Y%m%d")): day
                   for day in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            day, d, f, err = fut.result()
            if err is not None:
                fails += 1
                print(f"fetch fail {day}: {err}", flush=True)
                continue
            _process(day, d, f)
            if i % 25 == 0 or i == len(futures):
                print(f"fetched {i}/{len(futures)} (+{len(done)} done), fails={fails} "
                      f"({time.time()-started:.0f}s)", flush=True)
                if raws:
                    pd.concat(raws, ignore_index=True).to_csv(
                        os.path.join(CKPT, f"raw_{len(done)}.csv"), index=False)
                    raws = []
                if facs:
                    pd.concat(facs, ignore_index=True).to_csv(
                        os.path.join(CKPT, f"fac_{len(done)}.csv"), index=False)
                    facs = []
    done_f.close()
    if fails > 3:
        print(f"WARNING: {fails} days failed — rerun to fill those")
        return 1

    parts_r, parts_f = [], []
    for name in sorted(os.listdir(CKPT)):
        if name.startswith("raw_"):
            parts_r.append(pd.read_csv(os.path.join(CKPT, name), dtype={"symbol": str}))
        elif name.startswith("fac_"):
            parts_f.append(pd.read_csv(os.path.join(CKPT, name), dtype={"symbol": str}))
    raw_all = pd.concat(parts_r, ignore_index=True).drop_duplicates(
        ["symbol", "date"])
    fac_all = pd.concat(parts_f, ignore_index=True).drop_duplicates(
        ["symbol", "date"])
    print(f"collected backfill rows: raw {len(raw_all)}, factor {len(fac_all)} "
          f"({time.time()-started:.0f}s)")

    jobs = []
    for sym, grp in raw_all.groupby("symbol"):
        fac = fac_all[fac_all["symbol"] == sym]
        jobs.append((sym, grp, fac[["date", "adj_factor"]] if len(fac) else None))
    added = 0
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        for sym, n in pool.map(_merge_symbol, jobs, chunksize=8):
            added += n
    print(f"merged {len(jobs)} symbols, +{added} raw rows ({time.time()-started:.0f}s)")

    for sym, *_ in jobs:
        p = f"data/daily/{sym}_qfq.csv"
        if os.path.exists(p):
            os.remove(p)
    print("stale qfq removed; run: .venv/bin/python scripts/fix_qfq_conflicts.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
