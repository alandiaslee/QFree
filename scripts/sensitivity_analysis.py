"""Parameter sensitivity analysis for strategy A on an expanded walk-forward run.

Recomputes date-level selections under gate/boost variants from the candidate
CSV (no rescan needed: startup_signal, breadth, market_return_5d, base/total
score and financial completeness are all persisted per row), then re-runs the
profit-target event study for each (variant, target, holding) combination.

Performance: bars are loaded and preprocessed once in the parent process and
shared with workers via fork copy-on-write; the (variant, target) grid runs in
parallel. Results are identical to serial evaluation because the event-study
transforms are idempotent on preprocessed frames.
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
SUMMARY_CSV = "outputs/sensitivity_summary.csv"
HOLDINGS = (3, 5, 8)
TARGETS = (0.05, 0.06, 0.08)
MIN_SCORE = 60.0
MIN_FINANCIAL_FIELDS = 3
BREADTH = 0.01
WORKERS = 12

# Populated in the parent before forking; inherited copy-on-write by workers.
_BARS: dict[str, pd.DataFrame] = {}
_TRADE_DATES: pd.DatetimeIndex | None = None


def _preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same normalization evaluate_events performs per signal."""
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result = result.dropna(subset=["date"]).sort_values("date")
    return result.drop_duplicates("date").reset_index(drop=True)


def _evaluate_job(job: tuple) -> tuple[str, float, pd.DataFrame]:
    label, target, signals = job
    events = evaluate_events(
        signals, _BARS, HOLDINGS,
        exit_modes=("profit_target_close",),
        trade_dates=_TRADE_DATES, profit_target=target,
    )
    return label, target, summarize_events(events)


def build_variants() -> tuple[pd.DataFrame, list[tuple[str, pd.Series]]]:
    """Return (candidates, (label, selected_mask)) pairs recomputed from the CSV."""
    cand = pd.read_csv(CANDIDATES_CSV, parse_dates=["signal_date", "as_of"])
    startup = cand["startup_signal"].astype(str).str.lower().isin(("true", "1"))
    breadth = cand["startup_signal_breadth"]
    ret5 = cand["market_return_5d"]
    fin_ok = cand["financial_field_count"] >= MIN_FINANCIAL_FIELDS

    def gate(mode: str, threshold: float) -> pd.Series:
        if mode == "off":
            return pd.Series(True, index=cand.index)
        if mode == "OR":
            return (breadth >= BREADTH) | (ret5 >= threshold)
        if mode == "AND":
            return (breadth >= BREADTH) & (ret5 >= threshold)
        raise ValueError(mode)

    variants: list[tuple[str, pd.Series]] = []
    for mode, threshold in (("off", 0.0), ("OR", 0.017), ("OR", 0.025),
                            ("AND", 0.017), ("AND", 0.025)):
        gate_mask = gate(mode, threshold)
        for boost in (True, False):
            score = cand["total_score"] if boost else cand["base_total_score"]
            label = (f"gate={mode}{threshold if mode != 'off' else ''},"
                     f"boost={'on' if boost else 'off'}")
            variants.append((label, startup & gate_mask & (score >= MIN_SCORE) & fin_ok))
    return cand, variants


def main() -> int:
    global _TRADE_DATES
    started = time.time()
    cand, variants = build_variants()
    print(f"loaded {len(cand)} candidate rows in {time.time() - started:.1f}s")

    any_selected = variants[0][1].copy()
    for _, mask in variants[1:]:
        any_selected |= mask
    symbols = cand.loc[any_selected, "ts_code"].dropna().unique()
    print(f"loading bars for {len(symbols)} symbols ...")
    t0 = time.time()
    store = BarStore("data")
    for symbol in symbols:
        try:
            _BARS[symbol] = _preprocess(
                store.load(Instrument(symbol), "1900-01-01", "2100-01-01")
            )
        except FileNotFoundError:
            continue
    print(f"loaded+preprocessed {len(_BARS)} symbols in {time.time() - t0:.1f}s")

    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    _TRADE_DATES = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2025-01-01", "2026-10-09")))

    jobs: list[tuple[str, float, pd.DataFrame]] = []
    for label, mask in variants:
        signals = cand.loc[mask, ["ts_code", "signal_date"]].copy()
        signals = signals.rename(columns={"signal_date": "date"})
        signals["selected"] = True
        for target in TARGETS:
            jobs.append((label, target, signals))
    print(f"submitting {len(jobs)} (variant, target) jobs on {WORKERS} workers")

    rows: list[dict[str, object]] = []
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        for index, (label, target, summary) in enumerate(
            pool.map(_evaluate_job, jobs), start=1
        ):
            for row in summary.to_dict("records"):
                rows.append({"variant": label, "profit_target": target, **row})
            print(f"[{index}/{len(jobs)}] {label} target={target:.0%} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
    print(f"all jobs done in {time.time() - t0:.1f}s")

    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_CSV, index=False)

    for metric in ("win_rate", "mean_net_return"):
        pivot = out.pivot_table(index=["variant", "profit_target"],
                                columns="holding_days", values=metric)
        print(f"\n=== {metric} ===")
        print(pivot.round(4).to_string())
    print(f"\ntotal: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
