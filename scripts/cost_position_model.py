"""Task 3: transaction cost model + position sizing + capital efficiency.

Reuses the 42-date union scan (breadth >= 0.80%). For final-config signals
(10d / 5% profit target), evaluates:
  (a) cost drag: round-trip cost scenarios 0/10/20/30 bps (commission+stamp
      ~10 bps realistic, plus slippage),
  (b) position sizing: per-date top-N by base_total_score (N=5/10/20/all),
  (c) capital schedule simulation: real exit dates, max concurrent capital,
      deployment fraction, annualized return,
  (d) the open gate question: threshold 1.0% vs 0.9% portfolio metrics with
      realistic costs.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from qfree.backtest.event_study import evaluate_events
from qfree.config import Settings
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider
from qfree.strategy.batch_scan import BatchScanConfig
from qfree.strategy.fast_scan import fast_walk_forward_scan

WORKERS = 12
TARGET = 0.05
HOLDING = 10
COSTS = (0.0, 0.001, 0.002, 0.003)  # round-trip fractions


def main() -> int:
    started = time.time()
    hist = pd.read_csv("outputs/full_breadth_history.csv", parse_dates=["cal_date"],
                       index_col="cal_date").sort_index()
    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    sessions = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2025-01-01", "2026-10-16")))

    dates = [str(d.date()) for d in hist.index[hist["breadth"] >= 0.008]]
    universe = pd.read_csv("data/universe.csv")
    cfg = BatchScanConfig(as_of=dates[-1], start="2025-01-01", end=dates[-1],
                          max_volume_ratio=2.5, min_quality_score=55)
    t0 = time.time()
    report = fast_walk_forward_scan(universe, cfg, dates, trade_dates=sessions,
                                    workers=WORKERS, data_dir="data")
    cand = report.candidates.copy()
    cand["signal_date"] = pd.to_datetime(cand["signal_date"]).dt.normalize()
    startup = cand["startup_signal"].astype(str).str.lower().isin(("true", "1"))
    base = (startup
            & (cand["base_total_score"] >= 60)
            & (cand["quality_score"] >= 55)
            & (cand["financial_field_count"] >= 3))
    print(f"scan {time.time()-t0:.0f}s; base signals: {int(base.sum())}")

    store = BarStore("data")
    syms = cand.loc[base, "ts_code"].dropna().unique()
    bars_map = {}
    for sym in syms:
        try:
            b = store.load(Instrument(sym), "2024-06-01", "2100-01-01")
        except FileNotFoundError:
            continue
        b = b.copy()
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        bars_map[sym] = b.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")

    signals = cand.loc[base, ["ts_code", "signal_date", "base_total_score"]].rename(
        columns={"signal_date": "date"}).copy()
    signals["selected"] = True
    events = evaluate_events(signals, bars_map, (HOLDING,),
                             exit_modes=("profit_target_close",),
                             trade_dates=sessions, profit_target=TARGET)
    done = events[events["status"].isin(
        ("ok", "window_end_exit", "profit_target_exit", "max_holding_exit"))].copy()
    done["net_return"] = pd.to_numeric(done["net_return"], errors="coerce")
    done = done.dropna(subset=["net_return"])
    scores = cand.set_index(["ts_code", "signal_date"])["base_total_score"]
    done["score"] = [scores.get((r.ts_code, r.signal_date), np.nan)
                     for r in done.itertuples()]
    print(f"completed events: {len(done)} ({time.time()-started:.0f}s)")

    # (a) cost drag on all signals
    print("\n=== (a) cost drag, all signals, 10d/5% target ===")
    for cost in COSTS:
        r = (1 + done["net_return"]) * (1 - cost) - 1
        print(f"cost={cost*100:.1f}% round-trip: win={(r>0).mean():.1%}  "
              f"mean={r.mean():+.2%}  median={r.median():+.2%}")

    # (b)+(d) per-date top-N portfolios, threshold 1.0% vs 0.9%, cost 10bp
    COST = 0.001
    hist_idx = hist.index.normalize()
    breadth = hist["breadth"]

    def portfolio(threshold: float, n_top: int | None) -> pd.DataFrame:
        keep_dates = set(hist_idx[breadth >= threshold])
        g = done[done["signal_date"].isin(keep_dates)]
        if n_top is not None:
            g = (g.sort_values("score", ascending=False)
                 .groupby("signal_date").head(n_top))
        per = (g.assign(r=lambda d: (1 + d["net_return"]) * (1 - COST) - 1)
               .groupby("signal_date")["r"].mean().reset_index()
               .rename(columns={"r": "ret"}))
        return per

    print("\n=== (b) per-date portfolio (10bp cost), by top-N ===")
    rows = []
    for n in (5, 10, 20, None):
        per = portfolio(0.01, n)
        rows.append({"topN": n or "all", "dates": len(per),
                     "win": (per["ret"] > 0).mean(), "mean": per["ret"].mean(),
                     "median": per["ret"].median(), "total": (1 + per["ret"]).prod() - 1})
    print(pd.DataFrame(rows).to_string(index=False,
          formatters={"win": "{:.1%}".format, "mean": "{:+.2%}".format,
                      "median": "{:+.2%}".format, "total": "{:+.2%}".format}))

    print("\n=== (d) gate question: 1.0% vs 0.9% vs 0.8% (top10, 10bp cost) ===")
    for thr in (0.01, 0.009, 0.008):
        per = portfolio(thr, 10)
        print(f"thr={thr:.1%}: {len(per)} dates | date-win {(per['ret']>0).mean():.1%} | "
              f"mean {per['ret'].mean():+.2%} | compounded {(1+per['ret']).prod()-1:+.2%}")

    # (c) capital schedule with real exit dates, top10 per date, 10bp cost
    per10 = portfolio(0.01, 10)
    pos = done[done["signal_date"].isin(set(per10["signal_date"]))].copy()
    pos = (pos.sort_values("score", ascending=False)
           .groupby("signal_date").head(10))
    pos["r_net"] = (1 + pos["net_return"]) * (1 - COST) - 1
    pos["entry_date"] = pos["exit_date"] - pd.to_timedelta(0, unit="D")  # placeholder
    # entry day = first session after signal_date
    session_list = list(sessions)
    pos["entry_d"] = pos["signal_date"].map(
        lambda d: min((s for s in session_list if s > d), default=None))
    pos["exit_d"] = pd.to_datetime(pos["exit_date"])
    pos = pos.dropna(subset=["entry_d"])
    span_days = (sessions[-1] - sessions[0]).days
    invested_days = 0.0
    max_concurrent = 0
    cur = 0.0
    timeline = sorted([(d, 1) for d in pos["entry_d"]] + [(d, -1) for d in pos["exit_d"]])
    for d, delta in timeline:
        cur += delta
        max_concurrent = max(max_concurrent, cur)
    active_session_days = sum(
        1 for s in sessions
        if ((pos["entry_d"] <= s) & (pos["exit_d"] >= s)).any())
    deployment = active_session_days / len(sessions)
    # annualized: compounded per-date portfolio return scaled by deployment
    compounded = (1 + per10["ret"]).prod() - 1
    years = span_days / 365.25
    cagr = (1 + compounded) ** (1 / years) - 1 if compounded > -1 else -1
    print("\n=== (c) capital schedule (top10, 10bp, threshold 1.0%) ===")
    print(f"span: {sessions[0].date()} .. {sessions[-1].date()} ({span_days} days, "
          f"{len(sessions)} sessions)")
    print(f"positions: {len(pos)}, max concurrent: {max_concurrent}")
    print(f"active sessions: {active_session_days}/{len(sessions)} "
          f"(deployment {deployment:.1%})")
    print(f"compounded portfolio return over full span: {compounded:+.2%}")
    print(f"naive CAGR (compounded over {years:.2f}y): {cagr:+.2%}")
    # capital-efficiency-adjusted: only deployed fraction earns
    eff_cagr = (1 + compounded) ** (deployment / years) - 1 if compounded > -1 else -1
    print(f"deployment-adjusted CAGR: {eff_cagr:+.2%}")
    print(f"\ntotal: {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
