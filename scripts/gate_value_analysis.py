"""Gate value analysis: what do the market gates actually filter?

Runs the final-config signal pipeline (V0 + vol cap 2.5 + score>=60 + q>=55)
on EVERY session 2023-04..2026-10 with NO gates, takes top-10 per date, and
evaluates 10d/5% events net of 10bp cost. Then classifies each date by the
v3 gates (in-scan breadth >=0.9%, CSI300>MA20) and compares:

  - no gate at all (trade every day there are signals)
  - breadth gate only / regime gate only / both (official v3)
  - "best filter" reference: hindsight oracle (only positive-mean dates)

Answers: how much return the gates destroy vs protect, and how close v3
gets to the un-achievable oracle. Chunked by year to bound memory.

Usage: .venv/bin/python -u scripts/gate_value_analysis.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from qfree.backtest.event_study import evaluate_events
from qfree.config import Settings
from qfree.data.index_cache import load_index, regime_flags
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider
from qfree.strategy.batch_scan import BatchScanConfig
from qfree.strategy.fast_scan import fast_walk_forward_scan

DATA_DIR = "data"
WORKERS = 12
COST = 0.001
TARGET = 0.05
HOLDING = 10
KEEP_COLS = ["ts_code", "signal_date", "base_total_score", "startup_signal",
             "quality_score", "startup_signal_breadth", "selected"]


def main() -> int:
    started = time.time()
    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    sessions = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2023-04-01", "2026-10-16")))

    universe = pd.read_csv("data/universe.csv")
    rf = regime_flags(load_index(DATA_DIR), 20)

    all_signals = []
    import os
    for year in sorted(set(sessions.year)):
        cache = f"outputs/gate_value_signals_{year}.csv"
        if os.path.exists(cache):
            sig = pd.read_csv(cache)
            all_signals.append(sig)
            print(f"{year}: cached signals {len(sig)}", flush=True)
            continue
        year_sessions = sessions[sessions.year == year]
        dates = [str(d.date()) for d in year_sessions]
        cfg = BatchScanConfig(as_of=dates[-1], start="2023-01-01", end=dates[-1],
                              max_volume_ratio=2.5, min_quality_score=55)
        t0 = time.time()
        report = fast_walk_forward_scan(universe, cfg, dates,
                                        trade_dates=year_sessions,
                                        workers=WORKERS, data_dir=DATA_DIR)
        cand = report.candidates
        if cand.empty:
            continue
        startup = cand["startup_signal"].astype(str).str.lower().isin(("true", "1"))
        sig = cand.loc[startup
                       & (cand["base_total_score"] >= 60)
                       & (pd.to_numeric(cand["quality_score"], errors="coerce") >= 55),
                       [c for c in KEEP_COLS if c in cand.columns]]
        sig.to_csv(cache, index=False)
        all_signals.append(sig)
        print(f"{year}: scan {time.time()-t0:.0f}s | candidates {len(cand)} | "
              f"signals {len(sig)} | dates with signals: {sig['signal_date'].nunique()}",
              flush=True)
    signals = pd.concat(all_signals, ignore_index=True)
    signals["signal_date"] = pd.to_datetime(signals["signal_date"]).dt.normalize()
    print(f"total signals (no gate): {len(signals)} on "
          f"{signals['signal_date'].nunique()} dates ({time.time()-started:.0f}s)")

    # top10 per date by score
    top = (signals.sort_values("base_total_score", ascending=False)
           .groupby("signal_date").head(10))
    top = top.copy()
    top["breadth_gate"] = pd.to_numeric(top["startup_signal_breadth"],
                                        errors="coerce") >= 0.009
    top["regime_gate"] = top["signal_date"].map(rf).fillna(False).astype(bool)
    top["v3_gate"] = top["breadth_gate"] & top["regime_gate"]
    top.to_csv("outputs/gate_value_top.csv", index=False)

    store = BarStore(DATA_DIR)
    bars_map = {}
    for sym in top["ts_code"].unique():
        try:
            b = store.load(Instrument(sym), "2023-01-01", "2100-01-01").copy()
        except FileNotFoundError:
            continue
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        bars_map[sym] = (b.dropna(subset=["date"]).drop_duplicates("date")
                         .sort_values("date"))
    print(f"bars for {len(bars_map)} symbols", flush=True)

    ev = evaluate_events(
        top.rename(columns={"signal_date": "date"})
            .assign(selected=True)[["ts_code", "date", "selected"]],
        bars_map, (HOLDING,), exit_modes=("profit_target_close",),
        trade_dates=sessions, profit_target=TARGET)
    done = ev[ev["status"].isin(
        ("ok", "window_end_exit", "profit_target_exit", "max_holding_exit"))].copy()
    done["net_return"] = pd.to_numeric(done["net_return"], errors="coerce")
    done = done.dropna(subset=["net_return"])
    done["r_net"] = (1 + done["net_return"]) * (1 - COST) - 1
    keys = top.set_index(["ts_code", "signal_date"])[["breadth_gate", "regime_gate", "v3_gate"]]
    done = done.join(keys, on=["ts_code", "signal_date"])
    print(f"completed events: {len(done)} ({time.time()-started:.0f}s)")

    def block(label: str, mask: pd.Series) -> None:
        g = done[mask].copy()
        per = g.groupby("signal_date")["r_net"].mean()
        if per.empty:
            print(f"{label:<34} EMPTY")
            return
        comp = (1 + per).prod() - 1
        active = g.groupby("signal_date").size().clip(upper=10)
        deploy_days = int((active.groupby(level=0).first()).sum())
        years = (sessions[-1] - sessions[0]).days / 365.25
        adj = (1 + comp) ** ((deploy_days / 10 / len(sessions)) / years) - 1 if comp > -1 else -1
        print(f"{label:<34} {len(per):>3} dates | win {(per>0).mean():>4.0%} | "
              f"mean {per.mean():>+6.2%} | comp {comp:>+8.2%} | adjCAGR {adj:>+6.2%}")

    print("\n=== per-date top-10 portfolio by gate group (10d/5%, 10bp) ===")
    block("no gate (every signal day)", pd.Series(True, index=done.index))
    block("breadth gate only", done["breadth_gate"])
    block("regime gate only", done["regime_gate"])
    block("v3 = breadth AND regime (official)", done["v3_gate"])
    block("breadth pass but regime BLOCK", done["breadth_gate"] & ~done["regime_gate"])
    block("regime pass but breadth fail", done["regime_gate"] & ~done["breadth_gate"])
    block("both gates fail", ~done["breadth_gate"] & ~done["regime_gate"])

    # oracle reference: dates with positive realized mean (hindsight upper bound)
    per_all = done.groupby("signal_date")["r_net"].mean()
    oracle_dates = set(per_all[per_all > 0].index)
    block("ORACLE (only positive dates)", done["signal_date"].isin(oracle_dates))

    print("\n=== single-name stats by group ===")
    for label, mask in [("no gate", pd.Series(True, index=done.index)),
                        ("v3 gate", done["v3_gate"]),
                        ("both gates fail", ~done["breadth_gate"] & ~done["regime_gate"])]:
        g = done[mask]
        print(f"{label:<12} n={len(g):>4} | win {(g['r_net']>0).mean():.0%} | "
              f"mean {g['r_net'].mean():+.2%} | median {g['r_net'].median():+.2%} | "
              f"worst {g['r_net'].min():+.2%}")

    # capture ratios
    per_v3 = done[done["v3_gate"]].groupby("signal_date")["r_net"].mean()
    per_none = per_all
    comp_v3 = (1 + per_v3).prod() - 1
    comp_none = (1 + per_none).prod() - 1
    comp_oracle = (1 + per_all[per_all.index.isin(oracle_dates)]).prod() - 1
    print(f"\nv3 compounded {comp_v3:+.2%} | no-gate {comp_none:+.2%} | "
          f"oracle {comp_oracle:+.2%}")
    print(f"v3 captures {comp_v3/comp_oracle:.0%} of the hindsight oracle and "
          f"{comp_v3/max(comp_none, 1e-9):.0%} of the no-gate total while trading "
          f"on {len(per_v3)}/{len(per_none)} of signal days")
    print(f"\ntotal: {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
