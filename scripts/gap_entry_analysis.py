"""Next-open gap analysis: does the opening gap predict forward returns?

For each cached final-config signal (outputs/stop_loss_signals.csv), compute
the next-session open gap vs the signal close, bucket it, and report the
realized 10d/+5% outcome per bucket. Informs the skip-if-gapped-up rule in
the order sheet.
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from qfree.data.models import Instrument
from qfree.data.store import BarStore

TARGET = 0.05
HOLDING = 10
BUCKETS = [-1.0, -0.05, -0.02, 0.0, 0.02, 0.05, 0.07, 0.095, 10.0]


def main() -> int:
    started = time.time()
    signals = pd.read_csv("outputs/stop_loss_signals.csv", parse_dates=["date"])
    store = BarStore("data")
    bars_map = {}
    for sym in signals["ts_code"].unique():
        try:
            b = store.load(Instrument(sym), "2024-06-01", "2100-01-01").copy()
        except FileNotFoundError:
            continue
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        bars_map[sym] = (b.dropna(subset=["date"]).drop_duplicates("date")
                         .sort_values("date").reset_index(drop=True))
    print(f"bars {len(bars_map)} ({time.time()-started:.0f}s)")

    rows = []
    for r in signals.itertuples():
        frame = bars_map.get(r.ts_code)
        if frame is None:
            continue
        future = frame[frame["date"] > r.date].reset_index(drop=True)
        if len(future) < HOLDING + 1:
            continue
        sig_close = float(frame[frame["date"] <= r.date].iloc[-1]["close"])
        entry = float(future.iloc[0]["open"])
        up = future.iloc[0].get("up_limit")
        if pd.notna(up) and entry >= float(up):
            rows.append({"gap": 0.2, "ret": None, "limit_up_entry": True})
            continue
        path = future.iloc[1:HOLDING + 1]["close"].astype(float)
        ret, hit = None, None
        for c in path:
            rr = float(c) / entry - 1
            if rr >= TARGET:
                hit, ret = True, TARGET
                break
        if hit is None:
            ret = float(path.iloc[-1]) / entry - 1
        rows.append({"gap": entry / sig_close - 1, "ret": ret,
                     "limit_up_entry": False})
    df = pd.DataFrame(rows).dropna(subset=["ret"])
    df["bucket"] = pd.cut(df["gap"], BUCKETS)
    print(f"\n=== realized 10d/+5% outcome by next-open gap ({len(df)} events) ===")
    per = (df.groupby("bucket", observed=True)
           .agg(n=("ret", "size"), win=("ret", lambda s: (s > 0).mean()),
                mean=("ret", "mean"), median=("ret", "median"),
                worst=("ret", "min"))
           .reset_index())
    per["bucket"] = per["bucket"].astype(str)
    print(per.to_string(index=False,
          formatters={"win": "{:.0%}".format, "mean": "{:+.2%}".format,
                      "median": "{:+.2%}".format, "worst": "{:+.2%}".format}))
    n_lu = int(df["limit_up_entry"].sum()) if "limit_up_entry" in df else 0
    print(f"limit-up at open (unbuyable): {n_lu}")
    for thr in (0.0, 0.02, 0.05, 0.07):
        sub = df[df["gap"] <= thr]
        print(f"skip if gap > {thr:+.0%}: keep {len(sub)} | win {(sub['ret']>0).mean():.0%} "
              f"| mean {sub['ret'].mean():+.2%} | compounded(top-equal) n/a")
    print(f"\ntotal: {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
