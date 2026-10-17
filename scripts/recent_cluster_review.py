"""Realized review of the most recent trigger cluster under the frozen v2 config.

Frozen config: V0 startup + volume cap 2.5 + base_total_score>=60 +
quality_score>=55, breadth gate >=0.9% (dates taken from
outputs/full_breadth_history.csv), top-10 equal weight per date,
10d holding / 5% profit target, returns net of 0.10% round-trip cost.
All events are fully realized (signals 2026-07-15..08-13, data to 10-16).
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from qfree.backtest.event_study import evaluate_events
from qfree.config import Settings
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider
from qfree.strategy.batch_scan import BatchScanConfig
from qfree.strategy.fast_scan import fast_walk_forward_scan

WORKERS = 12
COST = 0.001  # round-trip
TARGET = 0.05
HOLDING = 10
TOPN = 10


def main() -> int:
    started = time.time()
    hist = pd.read_csv("outputs/full_breadth_history.csv", parse_dates=["cal_date"])
    dates = sorted(hist.loc[hist["breadth"] >= 0.009, "cal_date"]
                   .dt.strftime("%Y-%m-%d").tolist())[-13:]
    breadth_map = dict(zip(hist["cal_date"].dt.strftime("%Y-%m-%d"), hist["breadth"]))

    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    sessions = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2025-01-01", "2026-10-16")))

    universe = pd.read_csv("data/universe.csv")
    last = dates[-1]
    cfg = BatchScanConfig(as_of=last, start="2025-01-01", end=last,
                          max_volume_ratio=2.5, min_quality_score=55)
    report = fast_walk_forward_scan(universe, cfg, dates, trade_dates=sessions,
                                    workers=WORKERS, data_dir="data")
    cand = report.candidates.copy()
    cand["signal_date"] = pd.to_datetime(cand["signal_date"]).dt.normalize()
    startup = cand["startup_signal"].astype(str).str.lower().isin(("true", "1"))
    sig = (startup
           & (cand["base_total_score"] >= 60)
           & (cand["quality_score"] >= 55))
    print(f"dates: {len(dates)} | raw candidates: {len(cand)} | "
          f"final-config signals: {int(sig.sum())} ({time.time()-started:.0f}s)")

    store = BarStore("data")
    syms = cand.loc[sig, "ts_code"].dropna().unique()
    bars_map = {}
    for sym in syms:
        try:
            b = store.load(Instrument(sym), "2024-06-01", "2100-01-01").copy()
        except FileNotFoundError:
            continue
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        bars_map[sym] = (b.dropna(subset=["date"]).drop_duplicates("date")
                         .sort_values("date"))
    print(f"loaded bars for {len(bars_map)}/{len(syms)} symbols")

    signals = (cand.loc[sig, ["ts_code", "name", "signal_date", "base_total_score",
                              "quality_score"]]
               .rename(columns={"signal_date": "date"}).copy())
    signals["selected"] = True
    events = evaluate_events(signals, bars_map, (HOLDING,),
                             exit_modes=("profit_target_close",),
                             trade_dates=sessions, profit_target=TARGET)
    done = events[events["status"].isin(
        ("ok", "window_end_exit", "profit_target_exit", "max_holding_exit"))].copy()
    done["net_return"] = pd.to_numeric(done["net_return"], errors="coerce")
    done = done.dropna(subset=["net_return"])
    done["r_net"] = (1 + done["net_return"]) * (1 - COST) - 1
    done["date"] = pd.to_datetime(done["signal_date"]).dt.strftime("%Y-%m-%d")
    score_map = signals.set_index(["ts_code", "date"])["base_total_score"]
    done["score"] = [score_map.get((r.ts_code, r.date), pd.NA)
                     for r in done.itertuples()]

    # top-10 by score per date
    top = (done.sort_values("score", ascending=False)
           .groupby("signal_date").head(TOPN))

    print("\n=== per-date top-10 portfolio (realized, net of 0.10% cost) ===")
    per = (top.groupby("date")
           .agg(picks=("ts_code", "size"), win=("r_net", lambda s: (s > 0).mean()),
                mean=("r_net", "mean"), median=("r_net", "median"),
                total=("r_net", "sum"))
           .reset_index())
    per["breadth"] = per["date"].map(lambda d: f"{breadth_map.get(d, float('nan')):.2%}")
    per["day_ret"] = per["total"] / TOPN
    print(per.to_string(index=False,
          formatters={"win": "{:.0%}".format, "mean": "{:+.2%}".format,
                      "median": "{:+.2%}".format, "day_ret": "{:+.2%}".format}))
    day_win = (per["day_ret"] > 0).mean()
    comp = (1 + per["day_ret"]).prod() - 1
    print(f"\nportfolio over {len(per)} dates: date-win {day_win:.0%} | "
          f"mean {per['day_ret'].mean():+.2%} | median {per['day_ret'].median():+.2%} | "
          f"equal-date compounded {comp:+.2%}")

    print("\n=== every pick (top10/date, sorted by date, score) ===")
    cols = ["date", "ts_code", "name", "score", "r_net", "exit_date", "holding_days"]
    show = top.sort_values(["date", "score"], ascending=[True, False])
    print(show[cols].to_string(index=False, formatters={"r_net": "{:+.2%}".format,
                                                        "score": "{:.1f}".format}))
    n = len(show)
    print(f"\npicks: {n} | win {(show['r_net'] > 0).mean():.0%} | "
          f"mean {show['r_net'].mean():+.2%} | "
          f"total P&L {show['r_net'].sum():+.2%} (equal 1x per pick)")
    print(f"total: {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
