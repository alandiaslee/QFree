"""Build outputs/dashboard_data.json for the local dashboard.

Precomputes the official v3 backtest series from existing artifacts:
  outputs/full_breadth_history.csv   (sessions + breadth, 2023-04..latest)
  outputs/long_period_signals.csv    (final-config signals, all union dates)
  data/universe.csv                  (names / industries)

Picks = signals with rebuilt breadth >=0.9% AND CSI300>MA20 regime pass,
top10 per date by base_total_score. This reproduces the official v3
numbers exactly (55 dates / +236.4% / 541 signals as of 2026-10-17).

Evaluates 10d/5% profit-target events net of 10bp cost via evaluate_events,
derives the equity curve, cluster attribution, era split, capital schedule,
and a per-day calendar array for the UI month grid.

Usage: .venv/bin/python scripts/build_dashboard_data.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qfree.backtest.event_study import evaluate_events  # noqa: E402
from qfree.data.index_cache import load_index, regime_flags  # noqa: E402
from qfree.data.models import Instrument  # noqa: E402
from qfree.data.store import BarStore  # noqa: E402

DATA_DIR = ROOT / "data"
OUT = ROOT / "outputs" / "dashboard_data.json"
COST = 0.001
TARGET = 0.05
HOLDING = 10
THR = 0.009


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return clean(float(obj))
    return str(obj)


def load_official_picks() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (sessions, top10 picks per trigger date, regime_flags).

    picks columns: ts_code, date, base_total_score, name, industry.
    """
    hist = pd.read_csv(ROOT / "outputs" / "full_breadth_history.csv",
                       parse_dates=["cal_date"]).sort_values("cal_date")
    sessions = pd.DatetimeIndex(hist["cal_date"])
    rf = regime_flags(load_index(str(DATA_DIR)), 20)
    bmap = hist.set_index("cal_date")["breadth"]

    sig = pd.read_csv(ROOT / "outputs" / "long_period_signals.csv",
                      parse_dates=["date"])
    sig = sig[(sig["date"].map(bmap) >= THR)
              & sig["date"].map(rf).fillna(False).astype(bool)].copy()
    picks = (sig.sort_values("base_total_score", ascending=False)
             .groupby("date").head(10).copy())

    uni = pd.read_csv(DATA_DIR / "universe.csv",
                      usecols=["ts_code", "name", "industry"])
    picks = picks.merge(uni, on="ts_code", how="left")
    return sessions, picks, rf


def build() -> dict:
    started = time.time()
    sessions, picks, rf = load_official_picks()
    v3_dates = pd.DatetimeIndex(sorted(picks["date"].unique()))
    print(f"v3 picks: {len(picks)} on {len(v3_dates)} dates")

    store = BarStore(str(DATA_DIR))
    bars_map = {}
    for sym in picks["ts_code"].unique():
        try:
            b = store.load(Instrument(sym), "2023-01-01", "2100-01-01").copy()
        except FileNotFoundError:
            continue
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        bars_map[sym] = (b.dropna(subset=["date"]).drop_duplicates("date")
                         .sort_values("date"))
    print(f"bars for {len(bars_map)} symbols ({time.time()-started:.0f}s)")

    ev = evaluate_events(picks.assign(selected=True)[["ts_code", "date", "selected"]],
                         bars_map, (HOLDING,),
                         exit_modes=("profit_target_close",),
                         trade_dates=sessions, profit_target=TARGET)
    done = ev[ev["status"].isin(
        ("ok", "window_end_exit", "profit_target_exit", "max_holding_exit"))].copy()
    done["net_return"] = pd.to_numeric(done["net_return"], errors="coerce")
    done = done.dropna(subset=["net_return"])
    done["r_net"] = (1 + done["net_return"]) * (1 - COST) - 1
    done["signal_date"] = pd.to_datetime(done["signal_date"]).dt.normalize()

    per = (done.groupby("signal_date")
           .agg(day_ret=("r_net", "mean"), n=("ts_code", "size"),
                win=("r_net", lambda s: (s > 0).mean()))
           .reset_index())
    per["equity"] = (1 + per["day_ret"]).cumprod()
    compounded = float(per["equity"].iloc[-1] - 1)
    print(f"compounded {compounded:+.2%} | {len(per)} dates")

    # clusters: break when gap between consecutive trigger dates > 10 sessions
    pos = sessions.get_indexer(v3_dates)
    clusters = []
    cur = [0]
    for i in range(1, len(v3_dates)):
        if int(pos[i] - pos[i - 1]) > 10:
            clusters.append(cur)
            cur = [i]
        else:
            cur.append(i)
    clusters.append(cur)
    cluster_rows = []
    total_equity_factor = 1.0
    factors = []
    for idxs in clusters:
        dates = v3_dates[idxs]
        sub = per[per["signal_date"].isin(set(dates))]
        comp = float((1 + sub["day_ret"]).prod() - 1) if len(sub) else 0.0
        factors.append(1 + comp)
        cluster_rows.append({
            "start": dates[0].strftime("%Y-%m-%d"),
            "end": dates[-1].strftime("%Y-%m-%d"),
            "n_dates": len(dates),
            "n_signals": int(picks["date"].isin(set(dates)).sum()),
            "compounded": comp,
            "win": float((sub["day_ret"] > 0).mean()) if len(sub) else None,
        })
    total = float(np.prod(factors) - 1)
    for row, f in zip(cluster_rows, factors):
        row["contrib"] = (math.log(f) / math.log(1 + total)) if total > 0 else None
    print(f"clusters: {len(cluster_rows)} | chained {total:+.2%}")

    # era split
    eras = []
    for label, s0, s1 in [("2023-04~2024-12", "2023-04-01", "2024-12-31"),
                          ("2025-01~2026-10", "2025-01-01", "2026-10-16")]:
        sub = per[(per["signal_date"] >= s0) & (per["signal_date"] <= s1)]
        eras.append({
            "label": label, "n_dates": len(sub),
            "win": float((sub["day_ret"] > 0).mean()) if len(sub) else None,
            "mean": float(sub["day_ret"].mean()) if len(sub) else None,
            "compounded": float((1 + sub["day_ret"]).prod() - 1) if len(sub) else None,
        })

    # capital schedule
    pos_events = done.copy()
    session_list = list(sessions)
    pos_events["entry_d"] = pos_events["signal_date"].map(
        lambda d: min((s for s in session_list if s > d), default=None))
    pos_events["exit_d"] = pd.to_datetime(pos_events["exit_date"]).dt.normalize()
    pos_events = pos_events.dropna(subset=["entry_d"])
    cur = max_concurrent = 0
    for _, delta in sorted([(d, 1) for d in pos_events["entry_d"]]
                           + [(d, -1) for d in pos_events["exit_d"]]):
        cur += delta
        max_concurrent = max(max_concurrent, cur)
    active = int(sum(1 for s in sessions
                     if ((pos_events["entry_d"] <= s)
                         & (pos_events["exit_d"] >= s)).any()))
    deployment = active / len(sessions)
    years = (sessions[-1] - sessions[0]).days / 365.25
    naive_cagr = (1 + compounded) ** (1 / years) - 1
    adj_cagr = (1 + compounded) ** (deployment / years) - 1

    # calendar array for the UI month grid
    hist = pd.read_csv(ROOT / "outputs" / "full_breadth_history.csv",
                       parse_dates=["cal_date"]).sort_values("cal_date")
    picks_per_date = picks.groupby("date").size()
    v3_set = set(v3_dates)
    calendar = []
    for row in hist.itertuples():
        d = row.cal_date
        calendar.append({
            "date": d.strftime("%Y-%m-%d"),
            "breadth": float(row.breadth),
            "breadth_gate": bool(row.breadth >= THR),
            "regime_gate": bool(rf.get(d, False)),
            "v3_trigger": d in v3_set,
            "picks_count": int(picks_per_date.get(d, 0)),
        })

    data = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_last_date": sessions[-1].strftime("%Y-%m-%d"),
        "sessions": len(sessions),
        "summary": {
            "compounded": compounded,
            "n_dates": len(per),
            "n_signals": int(len(done)),
            "date_win": float((per["day_ret"] > 0).mean()),
            "mean": float(per["day_ret"].mean()),
            "median": float(per["day_ret"].median()),
            "worst_day": float(per["day_ret"].min()),
            "best_day": float(per["day_ret"].max()),
            "naive_cagr": float(naive_cagr),
            "deployment": float(deployment),
            "adj_cagr": float(adj_cagr),
            "max_concurrent": int(max_concurrent),
            "single_win": float((done["r_net"] > 0).mean()),
            "single_mean": float(done["r_net"].mean()),
            "single_worst": float(done["r_net"].min()),
            "cost_bp": int(COST * 10000),
            "target": TARGET,
            "holding": HOLDING,
            "breadth_thr": THR,
        },
        "eras": eras,
        "clusters": cluster_rows,
        "equity": {
            "dates": [d.strftime("%Y-%m-%d") for d in per["signal_date"]],
            "day_ret": per["day_ret"].tolist(),
            "equity": per["equity"].tolist(),
        },
        "per_date": [{
            "date": r.signal_date.strftime("%Y-%m-%d"),
            "day_ret": float(r.day_ret), "n": int(r.n), "win": float(r.win),
            "breadth": float(hist.set_index("cal_date")["breadth"].get(
                r.signal_date, float("nan"))),
        } for r in per.itertuples()],
        "calendar": calendar,
    }
    return data


def main() -> int:
    started = time.time()
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(clean(data), f, ensure_ascii=False, allow_nan=False)
    s = data["summary"]
    print(f"\nwritten {OUT} ({time.time()-started:.0f}s)")
    print(f"official check: compounded {s['compounded']:+.2%} | {s['n_dates']} dates | "
          f"adjCAGR {s['adj_cagr']:+.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
