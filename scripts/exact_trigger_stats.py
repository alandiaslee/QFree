"""Exact trigger-date statistics: scan EVERY trading day (not every-5-sessions)
for startup breadth, list all gate-pass dates, then evaluate the final
strategy on exactly those dates.

Pass 1 (fast): per-symbol features only, count startup signals per session.
Pass 2: full daily-scan pipeline (fast_walk_forward_scan) on gate-pass dates,
final-config filters, 5%/10d profit-target event study, per-date breakdown.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from qfree.backtest.event_study import evaluate_events, summarize_events
from qfree.config import Settings
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider
from qfree.features.technical import calculate_technical, technical_score, TechnicalConfig
from qfree.strategy.batch_scan import BatchScanConfig
from qfree.strategy.fast_scan import fast_walk_forward_scan
from qfree.universe import filter_universe

DATA_DIR = "data"
BREADTH_GATE = 0.01
NEAR_MISS = 0.008
WORKERS = 12
TARGET = 0.05

_SESSION_DAYS: np.ndarray | None = None


def _count_startups(ts_code: str) -> pd.DataFrame:
    """Per-session rows: startup count and has-bar flag (empty if no history)."""
    store = BarStore(DATA_DIR)
    try:
        bars = store.load(Instrument(ts_code), "2024-06-01", "2100-01-01")
    except FileNotFoundError:
        return pd.DataFrame()
    if len(bars) < 130:
        return pd.DataFrame()
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    bars = bars.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    feat = calculate_technical(bars, TechnicalConfig(max_volume_ratio=2.5))
    all_days = pd.DatetimeIndex(bars["date"].unique())
    frame = pd.DataFrame({"has_bar": 1}, index=all_days)
    counts = feat.loc[feat["startup_signal"].fillna(False).astype(bool),
                      "date"].value_counts()
    frame["n"] = pd.Series(counts, index=pd.DatetimeIndex(counts.index)).reindex(
        all_days, fill_value=0)
    return frame


def main() -> int:
    global _SESSION_DAYS
    started = time.time()
    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    sessions = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2025-01-01", "2026-10-16")))
    _SESSION_DAYS = sessions.values.astype("datetime64[D]").astype(np.int64)
    print(f"sessions: {len(sessions)} ({sessions[0].date()} .. {sessions[-1].date()})")

    universe = pd.read_csv("data/universe.csv")
    filtered = filter_universe(universe, as_of="2026-10-16")
    codes = filtered["ts_code"].astype(str).tolist()
    print(f"filtered universe: {len(codes)} symbols")

    ctx = mp.get_context("fork")
    frames = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        for index, frame in enumerate(pool.map(_count_startups, codes, chunksize=32), 1):
            if len(frame):
                frames.append(frame)
            if index % 1000 == 0:
                print(f"[{index}/{len(codes)}] {time.time() - t0:.0f}s", flush=True)
    if not frames:
        print("no startup signals found at all")
        return 0
    big = pd.concat(frames)
    by_date = big.groupby(level=0).sum().reindex(sessions, fill_value=0)
    n_dates = by_date["has_bar"].replace(0, np.nan)
    breadth = (by_date["n"] / n_dates).dropna()
    breadth = breadth.sort_index()
    print(f"pass 1 done in {time.time() - t0:.0f}s")

    out = pd.DataFrame({"breadth": breadth})
    out["startup_n"] = by_date["n"]
    out["n_bars"] = by_date["has_bar"]
    out.to_csv("outputs/full_breadth_history.csv")
    print(f"\nbreadth distribution over {len(out)} sessions:")
    print(out["breadth"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(4).to_string())

    trigger = out[out["breadth"] >= BREADTH_GATE]
    near = out[(out["breadth"] >= NEAR_MISS) & (out["breadth"] < BREADTH_GATE)]
    print(f"\n=== gate-pass dates (breadth>={BREADTH_GATE:.0%}): {len(trigger)} ===")
    for day, row in trigger.iterrows():
        print(f"  {day.date()}  breadth={row['breadth']:.2%}  ({int(row['startup_n'])} signals)")
    print(f"\nnear-miss dates (0.8%~1%): {len(near)}")
    for day, row in near.iterrows():
        print(f"  {day.date()}  breadth={row['breadth']:.2%}")

    # ---------- Pass 2: full pipeline + event study on trigger dates ----------
    if trigger.empty:
        print("no trigger dates; nothing to evaluate")
        return 0
    dates = [str(d.date()) for d in trigger.index]
    cfg = BatchScanConfig(as_of=dates[-1], start="2025-01-01", end=dates[-1],
                          max_volume_ratio=2.5, min_quality_score=55)
    t0 = time.time()
    report = fast_walk_forward_scan(universe, cfg, dates, trade_dates=sessions,
                                    workers=WORKERS, data_dir=DATA_DIR)
    print(f"\npass 2 scan done in {time.time() - t0:.0f}s")
    cand = report.candidates.copy()
    startup = cand["startup_signal"].astype(str).str.lower().isin(("true", "1"))
    sel = (startup
           & (cand["startup_signal_breadth"] >= BREADTH_GATE)
           & (cand["base_total_score"] >= 60)
           & (cand["quality_score"] >= 55)
           & (cand["financial_field_count"] >= 3))
    signals = cand.loc[sel, ["ts_code", "signal_date", "name"]].rename(
        columns={"signal_date": "date"}).copy()
    signals["selected"] = True
    print(f"final-config signals on exact trigger dates: {len(signals)}")

    store = BarStore(DATA_DIR)
    bars_map = {}
    for sym in signals["ts_code"].dropna().unique():
        try:
            b = store.load(Instrument(sym), "2024-06-01", "2100-01-01")
        except FileNotFoundError:
            continue
        b = b.copy()
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        bars_map[sym] = b.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    events = evaluate_events(signals, bars_map, (5, 10),
                             exit_modes=("profit_target_close",),
                             trade_dates=sessions, profit_target=TARGET)
    summary = summarize_events(events)
    print("\n=== final strategy on EXACT trigger dates (5% target) ===")
    print(summary.to_string(index=False,
                            formatters={"win_rate": "{:.1%}".format,
                                        "mean_net_return": "{:+.2%}".format,
                                        "median_net_return": "{:+.2%}".format}))
    per_date = (events[events["status"].isin(
        ("ok", "window_end_exit", "profit_target_exit", "max_holding_exit"))]
        .assign(net_return=lambda d: pd.to_numeric(d["net_return"], errors="coerce"))
        .dropna(subset=["net_return"])
        .groupby(["holding_days", "signal_date"])
        .agg(n=("ts_code", "size"), win=("net_return", lambda s: (s > 0).mean()),
             mean=("net_return", "mean"))
        .reset_index())
    per_date["win"] = per_date["win"].round(3)
    per_date["mean"] = per_date["mean"].round(4)
    per_date.to_csv("outputs/exact_trigger_per_date.csv", index=False)
    print("\nper-date (10d):")
    print(per_date[per_date.holding_days == 10].drop(columns="holding_days").to_string(
        index=False, formatters={"win": "{:.1%}".format, "mean": "{:+.2%}".format}))
    print(f"\ntotal: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
