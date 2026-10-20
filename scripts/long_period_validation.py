"""Long-period validation of the frozen v3 strategy over 2023-04..2026-10.

v3 = v2 + regime gate (CSI300 close > MA20 on the signal date; adopted
2026-10-17 after scripts/regime_filter_experiment.py showed it blocks the
2024 microcap-crash cluster and lifts full-period compounded +105% -> +186%).

Pass 1 rebuilds the startup-breadth history from 2023 (60-session indicator
warm-up; all caches now start 2023-01-03) and overwrites
outputs/full_breadth_history.csv (backup of the 2025-based file first).
Pass 2 runs the final config (V0 + vol cap 2.5 + score>=60 + q>=55) on the
breadth>=0.8% union dates, evaluates 10d/5% events net of 10bp cost, and
reports per-date top-10 portfolios: full-period, per-era split, gate
threshold comparison, and the capital schedule (deployment + honest CAGR).

Usage: .venv/bin/python -u scripts/long_period_validation.py [--no-regime]
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from qfree.backtest.event_study import evaluate_events
from qfree.config import Settings
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider
from qfree.features.technical import TechnicalConfig, calculate_technical
from qfree.strategy.batch_scan import BatchScanConfig
from qfree.strategy.fast_scan import fast_walk_forward_scan
from qfree.universe import filter_universe

DATA_DIR = "data"
WORKERS = 12
COST = 0.001
TARGET = 0.05
HOLDING = 10
HIST_CSV = "outputs/full_breadth_history.csv"
HIST_BAK = "outputs/full_breadth_history_2025.bak.csv"
SIGNALS_CSV = "outputs/long_period_signals.csv"


def _data_end() -> str:
    """Last session available locally (index cache tail) — the official
    backtest window extends as daily syncs add new sessions."""
    from qfree.data.index_cache import load_index
    idx = load_index(DATA_DIR)
    return str(pd.to_datetime(idx["date"].iloc[-1]).date())


def _count_startups(ts_code: str) -> pd.DataFrame:
    store = BarStore(DATA_DIR)
    try:
        bars = store.load(Instrument(ts_code), "2023-01-01", "2100-01-01")
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
    counts = feat.loc[feat["startup_signal"].fillna(False).astype(bool), "date"].value_counts()
    frame["n"] = pd.Series(counts, index=pd.DatetimeIndex(counts.index)).reindex(
        all_days, fill_value=0)
    return frame


def pass1_breadth(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    universe = pd.read_csv("data/universe.csv")
    filtered = filter_universe(universe, as_of=_data_end())
    codes = filtered["ts_code"].astype(str).tolist()
    print(f"pass1: {len(codes)} symbols, {len(sessions)} sessions", flush=True)
    ctx = mp.get_context("fork")
    frames = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        for i, frame in enumerate(pool.map(_count_startups, codes, chunksize=32), 1):
            if len(frame):
                frames.append(frame)
            if i % 1000 == 0:
                print(f"  [{i}/{len(codes)}] {time.time()-t0:.0f}s", flush=True)
    big = pd.concat(frames)
    by_date = big.groupby(level=0).sum().reindex(sessions, fill_value=0)
    n_dates = by_date["has_bar"].replace(0, np.nan)
    breadth = (by_date["n"] / n_dates).dropna().sort_index()
    out = pd.DataFrame({"breadth": breadth,
                        "startup_n": by_date["n"],
                        "n_bars": by_date["has_bar"]})
    if os.path.exists(HIST_CSV) and not os.path.exists(HIST_BAK):
        shutil.copy(HIST_CSV, HIST_BAK)
    out.to_csv(HIST_CSV)
    print(f"pass1 done {time.time()-t0:.0f}s | sessions with bars: {len(out)}", flush=True)
    print(f"breadth>=0.9%: {int((out['breadth']>=0.009).sum())} days | "
          f">=1.0%: {int((out['breadth']>=0.01).sum())} | "
          f">=0.8%: {int((out['breadth']>=0.008).sum())}")
    return out


def pass2_eval(hist: pd.DataFrame, sessions: pd.DatetimeIndex,
               universe: pd.DataFrame) -> None:
    dates = sorted(hist.index[hist["breadth"] >= 0.008].strftime("%Y-%m-%d"))
    print(f"pass2: scanning {len(dates)} union dates >=0.8%", flush=True)

    if os.path.exists(SIGNALS_CSV):
        signals = pd.read_csv(SIGNALS_CSV, parse_dates=["date"])
        print(f"cached signals: {len(signals)}")
    else:
        cfg = BatchScanConfig(as_of=dates[-1], start="2023-01-01", end=dates[-1],
                              max_volume_ratio=2.5, min_quality_score=55)
        t0 = time.time()
        report = fast_walk_forward_scan(universe, cfg, dates, trade_dates=sessions,
                                        workers=WORKERS, data_dir=DATA_DIR)
        print(f"scan {time.time()-t0:.0f}s", flush=True)
        cand = report.candidates.copy()
        cand["signal_date"] = pd.to_datetime(cand["signal_date"]).dt.normalize()
        startup = cand["startup_signal"].astype(str).str.lower().isin(("true", "1"))
        sig = (startup & (cand["base_total_score"] >= 60)
               & (cand["quality_score"] >= 55))
        signals = (cand.loc[sig, ["ts_code", "signal_date", "base_total_score"]]
                   .rename(columns={"signal_date": "date"}).copy())
        signals.to_csv(SIGNALS_CSV, index=False)
    print(f"final-config signals: {len(signals)} on union dates", flush=True)

    # v3 regime gate: CSI300 close > MA20 on the signal date
    if "--no-regime" not in sys.argv:
        from qfree.data.index_cache import load_index, regime_flags as _mk_flags
        rf = _mk_flags(load_index(DATA_DIR), 20)
        keep = signals["date"].map(rf)
        keep = keep.fillna(False).astype(bool)
        print(f"regime gate (CSI300>MA20): {int(keep.sum())}/{len(signals)} signals kept",
              flush=True)
        signals = signals[keep].copy()

    # map each signal to the breadth of its date for threshold gating
    bmap = hist["breadth"]
    signals = signals.copy()
    signals["breadth"] = signals["date"].map(bmap)

    store = BarStore(DATA_DIR)
    bars_map = {}
    for sym in signals["ts_code"].unique():
        try:
            b = store.load(Instrument(sym), "2023-01-01", "2100-01-01").copy()
        except FileNotFoundError:
            continue
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        bars_map[sym] = (b.dropna(subset=["date"]).drop_duplicates("date")
                         .sort_values("date"))
    print(f"bars for {len(bars_map)} symbols", flush=True)

    ev = evaluate_events(signals[["ts_code", "date", "selected"]] if "selected" in signals
                         else signals.assign(selected=True)[["ts_code", "date", "selected"]],
                         bars_map, (HOLDING,), exit_modes=("profit_target_close",),
                         trade_dates=sessions, profit_target=TARGET)
    done = ev[ev["status"].isin(
        ("ok", "window_end_exit", "profit_target_exit", "max_holding_exit"))].copy()
    done["net_return"] = pd.to_numeric(done["net_return"], errors="coerce")
    done = done.dropna(subset=["net_return"])
    done["r_net"] = (1 + done["net_return"]) * (1 - COST) - 1
    done["breadth"] = done["signal_date"].map(bmap)
    scores = signals.set_index(["ts_code", "date"])["base_total_score"]
    done["score"] = [scores.get((r.ts_code, r.signal_date), np.nan)
                     for r in done.itertuples()]
    print(f"completed events: {len(done)}", flush=True)

    def portfolio(thr: float, n_top: int = 10) -> pd.DataFrame:
        g = done[done["breadth"] >= thr]
        if n_top:
            g = (g.sort_values("score", ascending=False)
                 .groupby("signal_date").head(n_top))
        return g.groupby("signal_date")["r_net"].mean().reset_index(name="day_ret")

    def report_block(label: str, per: pd.DataFrame) -> None:
        if per.empty:
            print(f"{label}: empty")
            return
        comp = (1 + per["day_ret"]).prod() - 1
        print(f"{label}: {len(per)} dates | date-win {(per['day_ret']>0).mean():.0%} | "
              f"mean {per['day_ret'].mean():+.2%} | median {per['day_ret'].median():+.2%} | "
              f"equal-date compounded {comp:+.2%}")

    print("\n=== full period 2023-04..2026-10, top10 per date, 10bp cost ===")
    for thr in (0.01, 0.009, 0.008):
        report_block(f"thr={thr:.1%}", portfolio(thr))

    eras = [("2023-04~2024-12", "2023-04-01", "2024-12-31"),
            ("2025-01~now", "2025-01-01", _data_end())]
    print("\n=== era split (thr=0.9%) ===")
    for label, s0, s1 in eras:
        per = portfolio(0.009)
        per = per[(per["signal_date"] >= s0) & (per["signal_date"] <= s1)]
        report_block(label, per)

    print("\n=== per-date detail at thr=0.9% (top10, 10d/5%, net 10bp) ===")
    per = portfolio(0.009)
    detail = per.copy()
    detail["date"] = pd.to_datetime(detail["signal_date"]).dt.strftime("%Y-%m-%d")
    detail["breadth"] = detail["signal_date"].map(bmap).map("{:.2%}".format)
    detail["day_ret"] = detail["day_ret"].map("{:+.2%}".format)
    print(detail[["date", "breadth", "day_ret"]].to_string(index=False))

    # capital schedule on thr=0.9%
    pos = done[done["breadth"] >= 0.009].copy()
    pos = (pos.sort_values("score", ascending=False)
           .groupby("signal_date").head(10))
    session_list = list(sessions)
    pos["entry_d"] = pos["signal_date"].map(
        lambda d: min((s for s in session_list if s > d), default=None))
    pos["exit_d"] = pd.to_datetime(pos["exit_date"])
    pos = pos.dropna(subset=["entry_d"])
    max_concurrent = cur = 0
    for d, delta in sorted([(d, 1) for d in pos["entry_d"]] + [(d, -1) for d in pos["exit_d"]]):
        cur += delta
        max_concurrent = max(max_concurrent, cur)
    active = sum(1 for s in sessions
                 if ((pos["entry_d"] <= s) & (pos["exit_d"] >= s)).any())
    span_years = (sessions[-1] - sessions[0]).days / 365.25
    comp09 = (1 + per["day_ret"]).prod() - 1
    naive_cagr = (1 + comp09) ** (1 / span_years) - 1
    deploy = active / len(sessions)
    adj_cagr = (1 + comp09) ** (deploy / span_years) - 1
    print("\n=== capital schedule (thr=0.9%, top10, 10bp) ===")
    print(f"span: {sessions[0].date()} .. {sessions[-1].date()} "
          f"({len(sessions)} sessions, {span_years:.2f}y)")
    print(f"positions: {len(pos)} | max concurrent: {max_concurrent}")
    print(f"active sessions: {active}/{len(sessions)} (deployment {deploy:.1%})")
    print(f"compounded: {comp09:+.2%} | naive CAGR {naive_cagr:+.2%} | "
          f"deployment-adjusted CAGR {adj_cagr:+.2%}")

    # single-name stats within the 0.9% portfolio dates
    p = pos
    print(f"\nsingle names on portfolio dates: {len(p)} | win {(p['r_net']>0).mean():.0%} | "
          f"mean {p['r_net'].mean():+.2%} | median {p['r_net'].median():+.2%} | "
          f"worst {p['r_net'].min():+.2%}")


def main() -> int:
    started = time.time()
    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    # 60-session indicator warm-up from 2023-01-03 -> stats from 2023-04
    sessions = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2023-04-01", _data_end())))
    print(f"sessions: {len(sessions)} ({sessions[0].date()} .. {sessions[-1].date()})")

    hist = pass1_breadth(sessions)
    universe = pd.read_csv("data/universe.csv")
    pass2_eval(hist, sessions, universe)
    print(f"\ntotal: {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
