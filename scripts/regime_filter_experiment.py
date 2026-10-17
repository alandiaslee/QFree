"""Regime-filter experiment on the long-period signal set.

Motivation: the 2023-2024 losing era (esp. the 2024-07~09 microcap
liquidity crisis) happens while the market is in a sustained downtrend
below its long MA, but the breadth-only gate stays wide open. Test
index-trend gates: allow a trigger date only if the index close is above
its MA-N on the signal date.

Baseline (no regime filter, thr=0.9%): 84 dates, date-win 56%,
mean +1.01%/date, compounded +105.5%, deployment-adjusted CAGR +5.85%.

Usage: .venv/bin/python -u scripts/regime_filter_experiment.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from qfree.backtest.event_study import evaluate_events
from qfree.config import Settings
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider

COST = 0.001
TARGET = 0.05
HOLDING = 10
THR = 0.009
INDEXES = {"hs300": "000300.SH", "csi1000": "000852.SH"}
MA_LENGTHS = (20, 60, 120, 250)


def main() -> int:
    started = time.time()
    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    sessions = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2023-04-01", "2026-10-16")))

    signals = pd.read_csv("outputs/long_period_signals.csv", parse_dates=["date"])

    # index regimes
    regimes = {}
    for label, code in INDEXES.items():
        df = provider._pro.index_daily(ts_code=code, start_date="20230101",
                                       end_date="20261016")
        df = df.sort_values("trade_date")
        df["date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("date")["close"].astype(float)
        flags = {}
        for n in MA_LENGTHS:
            ma = df.rolling(n).mean()
            flags[n] = (df > ma)
        regimes[label] = (df, flags)
        print(f"{label}: {len(df)} index bars {df.index[0].date()}..{df.index[-1].date()}")

    # events for all signals
    store = BarStore("data")
    bars_map = {}
    for sym in signals["ts_code"].unique():
        try:
            b = store.load(Instrument(sym), "2023-01-01", "2100-01-01").copy()
        except FileNotFoundError:
            continue
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        bars_map[sym] = (b.dropna(subset=["date"]).drop_duplicates("date")
                         .sort_values("date"))
    sig = signals.assign(selected=True)[["ts_code", "date", "selected"]]
    ev = evaluate_events(sig, bars_map, (HOLDING,),
                         exit_modes=("profit_target_close",),
                         trade_dates=sessions, profit_target=TARGET)
    done = ev[ev["status"].isin(
        ("ok", "window_end_exit", "profit_target_exit", "max_holding_exit"))].copy()
    done["net_return"] = pd.to_numeric(done["net_return"], errors="coerce")
    done = done.dropna(subset=["net_return"])
    done["r_net"] = (1 + done["net_return"]) * (1 - COST) - 1
    bmap = pd.read_csv("outputs/full_breadth_history.csv", parse_dates=["cal_date"]
                       ).set_index("cal_date")["breadth"]
    done["breadth"] = done["signal_date"].map(bmap)
    scores = signals.set_index(["ts_code", "date"])["base_total_score"]
    done["score"] = [scores.get((r.ts_code, r.signal_date), np.nan)
                     for r in done.itertuples()]
    done = done[done["breadth"] >= THR]
    print(f"events on thr>={THR:.1%} dates: {len(done)} ({time.time()-started:.0f}s)")

    def portfolio(mask_dates: pd.Series) -> pd.DataFrame:
        g = done[done["signal_date"].isin(set(mask_dates.index[mask_dates]))]
        g = (g.sort_values("score", ascending=False)
             .groupby("signal_date").head(10))
        return g.groupby("signal_date")["r_net"].mean().reset_index(name="day_ret")

    def report(label: str, per: pd.DataFrame) -> None:
        if per.empty:
            print(f"{label:<28} EMPTY")
            return
        comp = (1 + per["day_ret"]).prod() - 1
        years = (sessions[-1] - sessions[0]).days / 365.25
        active = len(per) * 10 / len(sessions)  # rough deployment proxy
        adj = (1 + comp) ** ((len(per) * 10 / len(sessions)) / years) - 1
        print(f"{label:<28} {len(per):>3} dates | win {(per['day_ret']>0).mean():>4.0%} | "
              f"mean {per['day_ret'].mean():>+6.2%} | comp {comp:>+7.2%} | adjCAGR {adj:>+6.2%}")

    print("\n=== regime filter comparison (thr=0.9% breadth + index regime) ===")
    all_dates = pd.DatetimeIndex(sorted(done["signal_date"].unique()))
    report("baseline (no regime)", portfolio(pd.Series(True, index=all_dates)))
    blocked_ref = {}
    for label, (close, flags) in regimes.items():
        for n in MA_LENGTHS:
            reg = flags[n].reindex(sessions).fillna(False)
            mask = pd.Series(reg.reindex(all_dates).fillna(False).values,
                             index=all_dates)
            per = portfolio(mask)
            report(f"{label} > MA{n}", per)
            blocked_ref[(label, n)] = per

    # detail: what the chosen filter blocks vs keeps (hs300 MA120)
    close, flags = regimes["hs300"]
    reg120 = flags[120].reindex(sessions).fillna(False)
    on = pd.Series(reg120.reindex(all_dates).fillna(False).values, index=all_dates)
    per_all = portfolio(pd.Series(True, index=all_dates)).set_index("signal_date")["day_ret"]
    kept, blocked = per_all[on], per_all[~on]
    print(f"\n=== hs300>MA120: blocked {len(blocked)} dates (mean {blocked.mean():+.2%}, "
          f"win {(blocked>0).mean():.0%}) | kept {len(kept)} dates "
          f"(mean {kept.mean():+.2%}, win {(kept>0).mean():.0%}) ===")
    print("blocked dates:")
    for d, r in blocked.items():
        b = bmap.get(d, np.nan)
        print(f"  {d.date()}  breadth={b:.2%}  day_ret={r:+.2%}")
    # era split for MA120 and MA250
    for n in (20, 120, 250):
        close, flags = regimes["hs300"]
        regn = flags[n].reindex(sessions).fillna(False)
        on = pd.Series(regn.reindex(all_dates).fillna(False).values, index=all_dates)
        per_all = portfolio(pd.Series(True, index=all_dates)).set_index("signal_date")["day_ret"]
        kept = per_all[on]
        print(f"\n=== era split with hs300>MA{n} (kept {len(kept)}, "
              f"blocked mean {per_all[~on].mean():+.2%}) ===")
        per_kept = kept.reset_index(name="day_ret")
        for label, s0, s1 in [("2023-04~2024-12", "2023-04-01", "2024-12-31"),
                              ("2025-01~2026-10", "2025-01-01", "2026-10-16")]:
            sub = per_kept[(per_kept["signal_date"] >= s0) & (per_kept["signal_date"] <= s1)]
            if len(sub):
                comp = (1 + sub["day_ret"]).prod() - 1
                print(f"  {label}: {len(sub)} dates | win {(sub['day_ret']>0).mean():.0%} | "
                      f"mean {sub['day_ret'].mean():+.2%} | comp {comp:+.2%}")
            else:
                print(f"  {label}: no dates")
        if n == 250:
            print("blocked dates (MA250):")
            for d, r in per_all[~on].items():
                print(f"  {d.date()}  breadth={bmap.get(d, np.nan):.2%}  day_ret={r:+.2%}")

    print(f"\ntotal: {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
