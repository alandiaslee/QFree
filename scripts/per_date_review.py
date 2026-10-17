"""Per-date review of the final config: picks and forward returns by as-of date.

Final config: AND gate (breadth>=1%, market_return_5d>=1.7%), no boost,
base_total_score>=60, quality_score>=55, 5% profit target, 5d holding.
Also shows the no-quality-filter variant per date so the filter's per-date
effect is visible, and a fixed-open 5d exit as a target-free robustness view.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from qfree.backtest.event_study import evaluate_events
from qfree.config import Settings
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider

CANDIDATES_CSV = "outputs/walk_exp_candidates.csv"
OUT_CSV = "outputs/per_date_final_config.csv"
BREADTH = 0.01
MARKET_RET5 = 0.017
MIN_FINANCIAL_FIELDS = 3
TARGET = 0.05
HOLDING = 5
WORKERS = 12

_BARS: dict[str, pd.DataFrame] = {}
_TRADE_DATES: pd.DatetimeIndex | None = None


def _preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result = result.dropna(subset=["date"]).sort_values("date")
    return result.drop_duplicates("date").reset_index(drop=True)


def _evaluate_job(job: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    _, signals = job
    return evaluate_events(
        signals, _BARS, (HOLDING,),
        exit_modes=("profit_target_close", "fixed_open"),
        trade_dates=_TRADE_DATES, profit_target=TARGET,
    )


def main() -> int:
    global _TRADE_DATES
    started = time.time()
    cand = pd.read_csv(CANDIDATES_CSV, parse_dates=["signal_date", "as_of"])

    startup = cand["startup_signal"].astype(str).str.lower().isin(("true", "1"))
    base = (startup
            & (cand["startup_signal_breadth"] >= BREADTH)
            & (cand["market_return_5d"] >= MARKET_RET5)
            & (cand["financial_field_count"] >= MIN_FINANCIAL_FIELDS)
            & (cand["base_total_score"] >= 60))
    quality = pd.to_numeric(cand["quality_score"], errors="coerce")
    mask_q = base & (quality >= 55)
    base_or = (startup
               & (cand["startup_signal_breadth"] >= BREADTH)
               & (cand["financial_field_count"] >= MIN_FINANCIAL_FIELDS)
               & (cand["base_total_score"] >= 60))
    mask_or_q = base_or & (quality >= 55)
    print(f"AND pool {int(base.sum())}, AND+q55 {int(mask_q.sum())}, "
          f"OR pool {int(base_or.sum())}, OR+q55 {int(mask_or_q.sum())} "
          f"in {time.time() - started:.1f}s")

    symbols = cand.loc[mask_q | base | mask_or_q | base_or, "ts_code"].dropna().unique()
    t0 = time.time()
    store = BarStore("data")
    for symbol in symbols:
        try:
            _BARS[symbol] = _preprocess(
                store.load(Instrument(symbol), "1900-01-01", "2100-01-01")
            )
        except FileNotFoundError:
            continue
    print(f"loaded {len(_BARS)}/{len(symbols)} symbols in {time.time() - t0:.1f}s")

    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    _TRADE_DATES = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2025-01-01", "2026-10-09")))

    jobs = []
    variants = (("AND+q55", mask_q), ("AND", base), ("OR+q55", mask_or_q), ("OR", base_or))
    for label, mask in variants:
        signals = cand.loc[mask, ["ts_code", "signal_date", "name"]].rename(
            columns={"signal_date": "date"}).copy()
        signals["selected"] = True
        jobs.append((label, signals))

    ctx = mp.get_context("fork")
    frames = []
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
        for label, events in zip([v[0] for v in variants], pool.map(_evaluate_job, jobs)):
            events = events.copy()
            events["variant"] = label
            frames.append(events)
    events = pd.concat(frames, ignore_index=True)

    valid = events[events["status"].isin(
        ("ok", "window_end_exit", "profit_target_exit", "max_holding_exit"))].copy()
    valid["net_return"] = pd.to_numeric(valid["net_return"], errors="coerce")
    valid = valid.dropna(subset=["net_return"])

    per_date = (valid.groupby(["variant", "exit_mode", "signal_date"])
                .agg(signals=("ts_code", "size"),
                     win_rate=("net_return", lambda s: (s > 0).mean()),
                     mean_ret=("net_return", "mean"),
                     median_ret=("net_return", "median"))
                .reset_index())
    per_date.to_csv(OUT_CSV, index=False)

    fmt = {"win_rate": "{:.0%}".format, "mean_ret": "{:+.2%}".format,
           "median_ret": "{:+.2%}".format}
    for mode in ("profit_target_close", "fixed_open"):
        for label in ("AND+q55", "OR+q55"):
            view = per_date[(per_date["variant"] == label)
                            & (per_date["exit_mode"] == mode)].copy()
            print(f"\n=== per date: {label}, {mode}, {HOLDING}d ===")
            print(view.drop(columns=["variant", "exit_mode"]).to_string(
                index=False, formatters=fmt))
            pos = (view["mean_ret"] > 0).mean()
            print(f"dates with positive mean: {pos:.0%} of {len(view)}")

    pivot = (valid[(valid["exit_mode"] == "profit_target_close")]
             .groupby(["signal_date", "variant"])["net_return"].mean()
             .unstack("variant"))
    pivot = pivot.dropna(subset=["AND", "AND+q55"]).sort_values("AND")
    print(f"\n=== per-date mean return: no_q vs q>=55 (paired, {len(pivot)} dates) ===")
    print(pivot.to_string(formatters={c: "{:+.2%}".format for c in pivot.columns}))
    print(f"\ntotal: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
