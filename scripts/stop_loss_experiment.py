"""Stop-loss design experiment on the frozen v2 signal set.

Baseline exit: close >= +5% -> exit at that close; else exit at close of
day 10. Variants add a close-based stop at -3%/-5%/-8%, plus a trailing
stop (-6% off the max close since entry). Execution mirrors the engine
convention (exit at the triggering close); a next-open stop variant is
reported for the recommended level to bound slippage optimism.

Signals: final config (V0 + vol cap 2.5 + score>=60 + q>=55) on all
breadth>=0.9% dates from outputs/full_breadth_history.csv, data to 2026-10-16.
All events fully realized. Returns net of 0.10% round-trip cost.
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from qfree.config import Settings
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider
from qfree.strategy.batch_scan import BatchScanConfig
from qfree.strategy.fast_scan import fast_walk_forward_scan

WORKERS = 12
COST = 0.001
TARGET = 0.05
HOLDING = 10
STOPS = (None, 0.08, 0.12, 0.15)
TRAIL = 0.06  # trailing: -6% off max close since entry


def main() -> int:
    started = time.time()
    hist = pd.read_csv("outputs/full_breadth_history.csv", parse_dates=["cal_date"])
    dates = sorted(hist.loc[hist["breadth"] >= 0.009, "cal_date"]
                   .dt.strftime("%Y-%m-%d").tolist())

    signals_path = "outputs/stop_loss_signals.csv"
    if __import__("os").path.exists(signals_path):
        signals = pd.read_csv(signals_path, parse_dates=["date"])
        print(f"loaded cached signals: {len(signals)}")
    else:
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
        signals = (cand.loc[sig, ["ts_code", "signal_date", "base_total_score"]]
                   .rename(columns={"signal_date": "date"}).copy())
        signals.to_csv(signals_path, index=False)
        print(f"dates {len(dates)} | signals {len(signals)} ({time.time()-started:.0f}s)")

    store = BarStore("data")
    syms = signals["ts_code"].unique()
    bars_map = {}
    for sym in syms:
        try:
            b = store.load(Instrument(sym), "2024-06-01", "2100-01-01").copy()
        except FileNotFoundError:
            continue
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        bars_map[sym] = (b.dropna(subset=["date"]).drop_duplicates("date")
                         .sort_values("date").reset_index(drop=True))
    print(f"bars {len(bars_map)}/{len(syms)} ({time.time()-started:.0f}s)")

    def simulate(stop: float | None = None, trail: float | None = None,
                 next_open: bool = False) -> pd.DataFrame:
        rows = []
        for r in signals.itertuples():
            frame = bars_map.get(r.ts_code)
            if frame is None:
                continue
            future = frame[frame["date"] > r.date].reset_index(drop=True)
            if len(future) < 2:
                continue
            entry = float(future.iloc[0]["open"])
            up = future.iloc[0].get("up_limit")
            if pd.notna(up) and entry >= float(up):
                continue
            if not entry or entry <= 0:
                continue
            path = future.iloc[1:HOLDING + 1]
            exit_idx, status = None, None
            peak = entry
            regret = float("nan")
            for i in range(len(path)):
                close = float(path.iloc[i]["close"])
                ret = close / entry - 1
                peak = max(peak, close)
                hit_stop = (stop is not None and ret <= -stop) or \
                           (trail is not None and close / peak - 1 <= -trail)
                if hit_stop:
                    exit_idx, status = i, "stop_exit"
                    if next_open and i + 1 < len(future) - 1:
                        pass  # handled below with full frame
                    # regret: best forward close return after stop day
                    fwd = future.iloc[1 + i:1 + HOLDING]
                    if len(fwd):
                        regret = float(fwd["close"].max()) / entry - 1
                    break
                if ret >= TARGET:
                    exit_idx, status = i, "profit_exit"
                    break
            if exit_idx is None:
                if len(path) >= HOLDING:
                    exit_idx, status = HOLDING - 1, "time_exit"
                else:
                    continue
            if status == "stop_exit" and next_open:
                ni = 1 + exit_idx + 1  # next bar after stop day
                if ni >= len(future):
                    continue
                exit_price = float(future.iloc[ni]["open"])
                exit_date = future.iloc[ni]["date"]
            else:
                exit_price = float(future.iloc[1 + exit_idx]["close"])
                exit_date = future.iloc[1 + exit_idx]["date"]
            rows.append({"ts_code": r.ts_code, "date": r.date,
                         "score": r.base_total_score, "status": status,
                         "ret": exit_price / entry - 1, "regret": regret,
                         "exit_date": exit_date})
        return pd.DataFrame(rows)

    def portfolio(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["r_net"] = (1 + d["ret"]) * (1 - COST) - 1
        top = (d.sort_values("score", ascending=False)
               .groupby("date").head(10))
        per = top.groupby("date")["r_net"].mean().reset_index(name="day_ret")
        return per

    print("\n=== stop-loss variants (all signals, 10d/+5% baseline, net 0.10% cost) ===")
    print(f"{'variant':<18}{'n':>5}{'win':>7}{'mean':>8}{'median':>8}"
          f"{'worst':>8}{'stop-outs':>10}{'dates':>6}{'date-win':>9}{'compounded':>11}")
    variants = [(f"stop {int(s*100)}%" if s else "no stop (base)", dict(stop=s))
                for s in STOPS]
    variants.append((f"trailing -{int(TRAIL*100)}%", dict(trail=TRAIL)))
    for label, kw in variants:
        df = simulate(**kw)
        per = portfolio(df)
        comp = (1 + per["day_ret"]).prod() - 1
        n_stop = int((df["status"] == "stop_exit").sum())
        print(f"{label:<18}{len(df):>5}{(df['ret']>0).mean():>7.0%}"
              f"{df['ret'].mean():>+8.2%}{df['ret'].median():>+8.2%}"
              f"{df['ret'].min():>+8.2%}{n_stop:>10}{len(per):>6}"
              f"{(per['day_ret']>0).mean():>9.0%}{comp:>+11.2%}")

    # recommended level: execution realism (next-open vs same-close)
    for level in (0.05, 0.08):
        for no in (False, True):
            df = simulate(stop=level, next_open=no)
            per = portfolio(df)
            comp = (1 + per["day_ret"]).prod() - 1
            n_stop = int((df["status"] == "stop_exit").sum())
            regret = df.loc[df["status"] == "stop_exit", "regret"]
            print(f"stop {int(level*100)}% {'next-open' if no else 'same-close'}"
                  f": win {(df['ret']>0).mean():.0%} mean {df['ret'].mean():+.2%} "
                  f"worst {df['ret'].min():+.2%} stops {n_stop} "
                  f"| stop-outs that later recovered >+5%: {(regret>0.05).mean():.0%} "
                  f"(avg post-stop max {regret.mean():+.2%})"
                  f" | dates {len(per)} compounded {comp:+.2%}")

    print(f"\ntotal: {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
