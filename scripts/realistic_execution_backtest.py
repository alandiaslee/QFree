"""Realistic execution backtest: apply the actual order-fill rules to the
official v3 picks and recompute the compounded portfolio return.

Fill rules (validated in scripts/gap_entry_analysis.py, frozen in memory
final-strategy-config):
  - Pre-open limit order at signal_close * 1.02; fills at next session open
    iff open <= limit (i.e. overnight gap <= +2%).
  - Manual cancel if open gap <= -5% (overnight crash).
  - Cash of skipped names is redistributed -> day return = mean over FILLED
    names only (equal weight per trigger date, same as official).
  - Exit: first close >= entry*1.05 within 10 sessions, else 10th close.
  - Cost: 10bp round trip, applied as (1+gross)*(1-0.001)-1.

Compares against the official +236.36% (fills-at-next-open for all names).

Usage: .venv/bin/python scripts/realistic_execution_backtest.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_dashboard_data import load_official_picks  # noqa: E402
from qfree.data.models import Instrument  # noqa: E402
from qfree.data.store import BarStore  # noqa: E402

GAP_LIMIT = 0.02      # limit price = close * 1.02
GAP_CANCEL = -0.05    # manual cancel at open gap <= -5%
TARGET = 0.05
HOLDING = 10
COST = 0.001


def main() -> int:
    started = time.time()
    sessions, picks, _ = load_official_picks()
    session_list = list(sessions)
    print(f"picks: {len(picks)} on {picks['date'].nunique()} dates")

    store = BarStore(str(ROOT / "data"))
    bars_map = {}
    for sym in picks["ts_code"].unique():
        try:
            b = store.load(Instrument(sym), "2023-01-01", "2100-01-01").copy()
        except FileNotFoundError:
            continue
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        bars_map[sym] = (b.dropna(subset=["date"]).drop_duplicates("date")
                         .sort_values("date").reset_index(drop=True))
    print(f"bars for {len(bars_map)} symbols ({time.time()-started:.0f}s)")

    events = []
    for r in picks.itertuples(index=False):
        sym, sdate = r.ts_code, r.date
        bars = bars_map.get(sym)
        if bars is None:
            events.append({"date": sdate, "ts_code": sym, "status": "missing_bars"})
            continue
        row = bars[bars["date"] == sdate]
        if row.empty:
            events.append({"date": sdate, "ts_code": sym, "status": "no_signal_bar"})
            continue
        signal_close = float(row["close"].iloc[0])
        fut = bars[bars["date"] > sdate].reset_index(drop=True)
        if fut.empty:
            events.append({"date": sdate, "ts_code": sym, "status": "no_future"})
            continue
        entry_open = float(fut["open"].iloc[0])
        entry_date = fut["date"].iloc[0]
        gap = entry_open / signal_close - 1
        # ---- fill rules
        if entry_open > signal_close * (1 + GAP_LIMIT) + 1e-9:
            events.append({"date": sdate, "ts_code": sym, "status": "skip_gap_up",
                           "gap": gap})
            continue
        if gap <= GAP_CANCEL:
            events.append({"date": sdate, "ts_code": sym, "status": "skip_crash_cancel",
                           "gap": gap})
            continue
        # ---- exit: profit target close within 10 sessions, else 10th close
        win = fut.iloc[:HOLDING + 1]  # entry row + up to 10 more
        exit_idx = None
        exit_status = None
        closes = pd.to_numeric(win["close"], errors="coerce")
        for i in range(1, len(win)):
            c = closes.iloc[i]
            if pd.notna(c) and float(c) / entry_open - 1 >= TARGET:
                exit_idx = i
                exit_status = "profit_target_exit"
                break
        if exit_idx is None:
            if len(win) <= HOLDING:
                events.append({"date": sdate, "ts_code": sym,
                               "status": "insufficient_future_data", "gap": gap})
                continue
            exit_idx = HOLDING
            exit_status = "max_holding_exit"
        exit_price = float(closes.iloc[exit_idx])
        exit_date = win["date"].iloc[exit_idx]
        gross = exit_price / entry_open - 1
        r_net = (1 + gross) * (1 - COST) - 1
        events.append({"date": sdate, "ts_code": sym, "status": exit_status,
                       "gap": gap, "r_net": r_net,
                       "entry_date": entry_date, "exit_date": exit_date,
                       "days_held": int(exit_idx)})

    ev = pd.DataFrame(events)
    filled = ev.dropna(subset=["r_net"]).copy()
    skipped = ev[ev["r_net"].isna()]
    print(f"\nfilled: {len(filled)} | skipped: {len(skipped)} "
          f"({time.time()-started:.0f}s)")
    print("skip reasons:", skipped["status"].value_counts().to_dict())
    print(f"gap stats of fills: median {filled['gap'].median():+.2%} | "
          f"min {filled['gap'].min():+.2%}")

    # ---- per-date portfolio return: equal weight among filled names
    per = (filled.groupby("date")
           .agg(day_ret=("r_net", "mean"), n=("ts_code", "size"),
                win=("r_net", lambda s: (s > 0).mean()))
           .reset_index())
    per["equity"] = (1 + per["day_ret"]).cumprod()
    compounded = float(per["equity"].iloc[-1] - 1)
    print(f"\n=== realistic execution (limit 1.02 / cancel -5%) ===")
    print(f"dates with fills: {len(per)} | compounded {compounded:+.2%} | "
          f"date-win {(per['day_ret']>0).mean():.0%} | mean {per['day_ret'].mean():+.2%}")
    print(f"single: n={len(filled)} win {(filled['r_net']>0).mean():.0%} "
          f"mean {filled['r_net'].mean():+.2%} worst {filled['r_net'].min():+.2%}")

    # official reference on the same pick set (all filled, next-open entry)
    print("\nofficial reference: +236.36% | 55 dates | 541 signals")

    # ---- capital schedule with realistic fills
    pos = filled.copy()
    active = int(sum(1 for s in session_list
                     if ((pos["entry_date"] <= s) & (pos["exit_date"] >= s)).any()))
    deployment = active / len(sessions)
    years = (sessions[-1] - sessions[0]).days / 365.25
    naive_cagr = (1 + compounded) ** (1 / years) - 1
    adj_cagr = (1 + compounded) ** (deployment / years) - 1
    print(f"\ncapital schedule: active {active}/{len(sessions)} sessions "
          f"({deployment:.1%}) | naive CAGR {naive_cagr:+.2%} | "
          f"deployment-adjusted CAGR {adj_cagr:+.2%}")

    # ---- per-cluster view
    from build_dashboard_data import THR  # noqa
    hist = pd.read_csv(ROOT / "outputs" / "full_breadth_history.csv",
                       parse_dates=["cal_date"]).sort_values("cal_date")
    bmap = hist.set_index("cal_date")["breadth"]
    per["breadth"] = per["date"].map(bmap)
    print("\nper-date (realistic):")
    for r in per.itertuples(index=False):
        print(f"  {r.date:%Y-%m-%d}  n={r.n:>2}  ret {r.day_ret:+.2%}  "
              f"win {r.win:.0%}  breadth {r.breadth:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
