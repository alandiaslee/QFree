"""Post-filter sensitivity: min total score x 20d-return floor x quality floor.

Takes the recommended AND-gate base pool (startup_signal, breadth>=1%,
market_return_5d>=1.7%, no boost) from the expanded walk-forward candidates
CSV and evaluates a grid of selection filters with the 5% profit-target /
5-day holding event study. No rescan needed: all filter inputs are persisted
per candidate row, and bars are loaded once per symbol, shared via fork COW.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from qfree.backtest.event_study import evaluate_events, summarize_events
from qfree.config import Settings
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider

CANDIDATES_CSV = "outputs/walk_exp_candidates.csv"
SUMMARY_CSV = "outputs/filter_sensitivity_summary.csv"
MIN_SCORES = (60.0, 65.0, 70.0, 75.0)
MIN_R20 = (None, -0.08, -0.05, -0.03)
MIN_QUALITY = (None, 40.0, 50.0, 55.0)
BREADTH = 0.01
MARKET_RET5 = 0.017
MIN_FINANCIAL_FIELDS = 3
TARGET = 0.05
HOLDING = 5
WORKERS = 12

# Populated in the parent before forking; inherited copy-on-write by workers.
_BARS: dict[str, pd.DataFrame] = {}
_TRADE_DATES: pd.DatetimeIndex | None = None


def _preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result = result.dropna(subset=["date"]).sort_values("date")
    return result.drop_duplicates("date").reset_index(drop=True)


def _evaluate_job(job: tuple[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    label, signals = job
    events = evaluate_events(
        signals, _BARS, (HOLDING,),
        exit_modes=("profit_target_close",),
        trade_dates=_TRADE_DATES, profit_target=TARGET,
    )
    return label, summarize_events(events)


def main() -> int:
    global _TRADE_DATES
    started = time.time()
    cand = pd.read_csv(CANDIDATES_CSV, parse_dates=["signal_date", "as_of"])
    print(f"loaded {len(cand)} candidate rows in {time.time() - started:.1f}s")

    startup = cand["startup_signal"].astype(str).str.lower().isin(("true", "1"))
    fin_ok = cand["financial_field_count"] >= MIN_FINANCIAL_FIELDS
    base = (startup
            & (cand["startup_signal_breadth"] >= BREADTH)
            & (cand["market_return_5d"] >= MARKET_RET5)
            & fin_ok)
    base_pool = cand.loc[base].copy()
    base_pool["return_20d"] = pd.to_numeric(base_pool["return_20d"], errors="coerce")
    base_pool["quality_score"] = pd.to_numeric(base_pool["quality_score"], errors="coerce")
    print(f"AND-gate base pool: {len(base_pool)} signals "
          f"({base_pool['ts_code'].nunique()} symbols, "
          f"{base_pool['signal_date'].nunique()} dates)")

    symbols = base_pool["ts_code"].dropna().unique()
    t0 = time.time()
    store = BarStore("data")
    for symbol in symbols:
        try:
            _BARS[symbol] = _preprocess(
                store.load(Instrument(symbol), "1900-01-01", "2100-01-01")
            )
        except FileNotFoundError:
            continue
    print(f"loaded+preprocessed {len(_BARS)}/{len(symbols)} symbols in {time.time() - t0:.1f}s")

    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    _TRADE_DATES = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2025-01-01", "2026-10-09")))

    jobs: list[tuple[str, pd.DataFrame]] = []
    for min_score in MIN_SCORES:
        score_ok = base_pool["base_total_score"] >= min_score
        for min_r20 in MIN_R20:
            r20_ok = (pd.Series(True, index=base_pool.index) if min_r20 is None
                      else base_pool["return_20d"] >= min_r20)
            for min_quality in MIN_QUALITY:
                q_ok = (pd.Series(True, index=base_pool.index) if min_quality is None
                        else base_pool["quality_score"] >= min_quality)
                mask = score_ok & r20_ok & q_ok
                label = f"score>={min_score:.0f},r20>={'all' if min_r20 is None else f'{min_r20:.0%}'},q>={'all' if min_quality is None else f'{min_quality:.0f}'}"
                signals = base_pool.loc[mask, ["ts_code", "signal_date"]].rename(
                    columns={"signal_date": "date"}).copy()
                signals["selected"] = True
                jobs.append((label, signals))
    print(f"{len(jobs)} filter combos on {WORKERS} workers")

    rows: list[dict[str, object]] = []
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        for index, (label, summary) in enumerate(pool.map(_evaluate_job, jobs), start=1):
            for record in summary.to_dict("records"):
                rows.append({"filters": label, **record})
            print(f"[{index}/{len(jobs)}] {label} ({time.time() - t0:.0f}s)", flush=True)
    print(f"all jobs done in {time.time() - t0:.1f}s")

    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_CSV, index=False)

    print("\n=== win_rate / mean_net_return (holding=5, target=5%) ===")
    view = out[["filters", "samples", "win_rate", "mean_net_return"]].copy()
    view = view.sort_values("mean_net_return", ascending=False)
    print(view.to_string(index=False,
                         formatters={"win_rate": "{:.1%}".format,
                                     "mean_net_return": "{:+.2%}".format}))

    print("\n=== heatmaps at quality>=50: rows=min_score, cols=min_r20 ===")
    withq = out[out["filters"].str.contains("q>=50")]
    for metric, fmt in (("win_rate", "{:.1%}"), ("mean_net_return", "{:+.2%}")):
        grid = withq.copy()
        grid["score"] = grid["filters"].str.extract(r"score>=(\d+)").astype(float)
        grid["r20"] = grid["filters"].str.extract(r"r20>=(all|-\d+%)")
        pivot = grid.pivot_table(index="score", columns="r20", values=metric)
        order = [c for c in ("all", "-8%", "-5%", "-3%") if c in pivot.columns]
        print(f"\n{metric}:")
        print(pivot[order].to_string(float_format=lambda v: fmt.format(v)))
    print(f"\ntotal: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
