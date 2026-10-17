"""Market-gate threshold sensitivity on the exact 431-session breadth history.

For thresholds 1.0%% .. 0.80%% (step 0.05), evaluate the final strategy on
every date whose startup breadth passes. One full scan over the union of
candidate dates, then per-threshold signal filtering + event study.
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from qfree.backtest.event_study import evaluate_events, summarize_events
from qfree.config import Settings
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider
from qfree.strategy.batch_scan import BatchScanConfig
from qfree.strategy.fast_scan import fast_walk_forward_scan

THRESHOLDS = (0.010, 0.0095, 0.009, 0.0085, 0.008)
WORKERS = 12
TARGET = 0.05


def main() -> int:
    started = time.time()
    hist = pd.read_csv("outputs/full_breadth_history.csv", parse_dates=["cal_date"],
                       index_col="cal_date")
    hist = hist.sort_index()
    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    sessions = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2025-01-01", "2026-10-16")))

    union_dates = hist.index[hist["breadth"] >= min(THRESHOLDS)]
    dates = [str(d.date()) for d in union_dates]
    print(f"union of candidate dates (breadth>={min(THRESHOLDS):.2%}): {len(dates)}")

    universe = pd.read_csv("data/universe.csv")
    cfg = BatchScanConfig(as_of=dates[-1], start="2025-01-01", end=dates[-1],
                          max_volume_ratio=2.5, min_quality_score=55)
    t0 = time.time()
    report = fast_walk_forward_scan(universe, cfg, dates, trade_dates=sessions,
                                    workers=WORKERS, data_dir="data")
    print(f"scan done in {time.time() - t0:.0f}s")
    cand = report.candidates.copy()
    cand["signal_date"] = pd.to_datetime(cand["signal_date"]).dt.normalize()
    startup = cand["startup_signal"].astype(str).str.lower().isin(("true", "1"))

    store = BarStore("data")
    all_syms = cand["ts_code"].dropna().unique()
    bars_map = {}
    for sym in all_syms:
        try:
            b = store.load(Instrument(sym), "2024-06-01", "2100-01-01")
        except FileNotFoundError:
            continue
        b = b.copy()
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        bars_map[sym] = b.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")

    rows = []
    prev_dates: set = set()
    for thr in THRESHOLDS:
        pass_dates = set(hist.index[hist["breadth"] >= thr].normalize())
        new_dates = sorted(pass_dates - prev_dates)
        prev_dates = pass_dates
        mask = (startup & cand["signal_date"].isin(pass_dates)
                & (cand["base_total_score"] >= 60)
                & (cand["quality_score"] >= 55)
                & (cand["financial_field_count"] >= 3))
        signals = cand.loc[mask, ["ts_code", "signal_date"]].rename(
            columns={"signal_date": "date"}).copy()
        signals["selected"] = True
        events = evaluate_events(signals, bars_map, (5, 10),
                                 exit_modes=("profit_target_close",),
                                 trade_dates=sessions, profit_target=TARGET)
        summary = summarize_events(events).set_index("holding_days")
        for holding in (5, 10):
            record = summary.loc[holding]
            rows.append({
                "threshold": f"{thr:.2%}", "dates": len(pass_dates),
                "signals": int(record["samples"]),
                "win_rate": record["win_rate"],
                "mean": record["mean_net_return"],
                "median": record["median_net_return"],
                "holding": holding,
            })
        print(f"thr={thr:.2%}: {len(pass_dates)} dates, {len(signals)} signals | "
              f"new dates at this step: {[str(d.date()) for d in new_dates]}")

    out = pd.DataFrame(rows)
    out.to_csv("outputs/gate_threshold_sensitivity.csv", index=False)
    for holding in (5, 10):
        view = out[out.holding == holding][["threshold", "dates", "signals",
                                            "win_rate", "mean", "median"]]
        print(f"\n=== threshold grid, holding={holding}d, 5% target ===")
        print(view.to_string(index=False,
                             formatters={"win_rate": "{:.1%}".format,
                                         "mean": "{:+.2%}".format,
                                         "median": "{:+.2%}".format}))

    # Per-date P&L at the loosest threshold, to see which marginal days hurt.
    thr = min(THRESHOLDS)
    pass_dates = set(hist.index[hist["breadth"] >= thr].normalize())
    mask = (startup & cand["signal_date"].isin(pass_dates)
            & (cand["base_total_score"] >= 60) & (cand["quality_score"] >= 55)
            & (cand["financial_field_count"] >= 3))
    signals = cand.loc[mask, ["ts_code", "signal_date"]].rename(
        columns={"signal_date": "date"}).copy()
    signals["selected"] = True
    events = evaluate_events(signals, bars_map, (10,),
                             exit_modes=("profit_target_close",),
                             trade_dates=sessions, profit_target=TARGET)
    valid = events[events["status"].isin(
        ("ok", "window_end_exit", "profit_target_exit", "max_holding_exit"))].copy()
    valid["net_return"] = pd.to_numeric(valid["net_return"], errors="coerce")
    per = (valid.dropna(subset=["net_return"])
           .groupby("signal_date")["net_return"]
           .agg(["size", "mean", lambda s: (s > 0).mean()])
           .reset_index())
    per.columns = ["date", "n", "mean", "win"]
    per = per.sort_values("date")
    per["breadth"] = per["date"].map(hist["breadth"])
    per["mean"] = per["mean"].round(4)
    per["win"] = per["win"].round(3)
    per.to_csv("outputs/gate_threshold_per_date.csv", index=False)
    print("\n=== per-date at loosest threshold (0.80%), 10d ===")
    print(per.to_string(index=False,
                        formatters={"breadth": "{:.2%}".format,
                                    "win": "{:.1%}".format, "mean": "{:+.2%}".format}))
    print(f"\ntotal: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
