"""Gentle-volume first-launch trigger vs current startup signal.

Variants, all restricted to breadth-gate pass dates from the walk-forward run:
  V0 current: price_position_60d<=0.35 & vol_ratio>=1.8 & MA5>MA20 rising & ret20<=25%
  V1 proposal: B1 bottom structure (dd120 -15..-45%, dist_low60<=15%, no new
     10d lows) + volume 1.3-2.5x + 3d volume trend >1.1 + day return 0.5-5%
     + close position >0.65 + 5d return <=8%
  Ablations: V1 without close strength, V1 without bottom structure,
  V1 with uncapped volume (>=1.8).

Workers scan the full universe once (one bars load per symbol, fork pool) and
emit factor rows for gate-pass dates only; the parent evaluates variants with
the canonical 5% profit-target event study at 5d/10d holdings.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from qfree.backtest.event_study import evaluate_events, summarize_events
from qfree.config import Settings
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider

CANDIDATES_CSV = "outputs/walk_exp_candidates.csv"
OUT_CSV = "outputs/volume_trigger_summary.csv"
SIGNALS_CSV = "outputs/volume_trigger_signals.csv"
WORKERS = 12
TARGET = 0.05
DATA_DIR = "data"

_GATE_DAYS: np.ndarray | None = None


def _symbol_factor_rows(ts_code: str) -> pd.DataFrame:
    store = BarStore(DATA_DIR)
    try:
        bars = store.load(Instrument(ts_code), "2024-01-01", "2100-01-01")
    except FileNotFoundError:
        return pd.DataFrame()
    if len(bars) < 130:
        return pd.DataFrame()
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    bars = bars.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)
    low = bars["low"].astype(float)
    high = bars["high"].astype(float)
    out = pd.DataFrame({"date": bars["date"].values})

    baseline = volume.shift(1).rolling(20).mean()
    out["vol_ratio"] = volume / baseline
    out["vol3_trend"] = volume.rolling(3).mean() / baseline
    out["day_ret"] = close.pct_change()
    rng = high - low
    out["close_pos"] = (close - low).div(rng.where(rng > 0))
    out["plat_dist"] = close / close.shift(1).rolling(20).max() - 1.0
    out["ret5"] = close.pct_change(5)
    out["ret20"] = close.pct_change(20)
    out["price_position_60d"] = (
        (close - close.rolling(60).min())
        / (close.rolling(60).max() - close.rolling(60).min())
    )
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    out["trend_up"] = (ma5 > ma20) & (ma5.diff() > 0)
    # Bottom-structure factors (same definitions as xushi experiment).
    out["dd120"] = close / close.rolling(120).max() - 1.0
    out["dist_low60"] = close / low.rolling(60).min() - 1.0
    prior_low = low.shift(10).rolling(50).min()
    out["low_stable"] = low.rolling(10).min().ge(prior_low * 0.97)

    assert _GATE_DAYS is not None
    days = out["date"].values.astype("datetime64[D]").astype(np.int64)
    positions = np.searchsorted(days, _GATE_DAYS, side="right") - 1
    valid = positions >= 0
    if not valid.any():
        return pd.DataFrame()
    picked = out.iloc[positions[valid]].copy()
    picked["as_of"] = pd.to_datetime(np.asarray(_GATE_DAYS)[valid], unit="D")
    picked["ts_code"] = ts_code
    return picked.reset_index(drop=True)


def main() -> int:
    global _GATE_DAYS
    started = time.time()
    cand = pd.read_csv(CANDIDATES_CSV, parse_dates=["signal_date", "as_of"])
    gate_dates = sorted(cand.loc[cand["startup_signal_breadth"] >= 0.01, "as_of"].unique())
    _GATE_DAYS = pd.DatetimeIndex(gate_dates).values.astype("datetime64[D]").astype(np.int64)
    print(f"gate-pass dates: {[str(pd.Timestamp(d).date()) for d in gate_dates]}")

    universe = pd.read_csv("data/universe.csv")
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        parts = list(pool.map(_symbol_factor_rows,
                              universe["ts_code"].astype(str).tolist(), chunksize=32))
    rows = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    rows = rows.rename(columns={"as_of": "signal_date"})
    print(f"{len(rows)} symbol-date rows on gate dates ({time.time() - started:.0f}s)")

    structure = (rows["dd120"].between(-0.45, -0.15)
                 & rows["dist_low60"].le(0.15)
                 & rows["low_stable"].fillna(False).astype(bool))
    v1_common = (rows["vol_ratio"].between(1.3, 2.5)
                 & rows["vol3_trend"].gt(1.1)
                 & rows["day_ret"].between(0.005, 0.05)
                 & rows["ret5"].le(0.08))
    v1 = structure & v1_common & rows["close_pos"].gt(0.65)
    v0 = (rows["price_position_60d"].le(0.35)
          & rows["vol_ratio"].ge(1.8)
          & rows["trend_up"].fillna(False)
          & rows["ret20"].le(0.25))

    variants = (
        ("V0|current_startup", v0),
        ("V1|structure+gentle_volume", v1),
        ("V1a|no_close_strength", structure & v1_common),
        ("V1b|day_conditions_only", v1_common & rows["close_pos"].gt(0.65)),
        ("V1c|uncapped_volume", structure
         & rows["vol_ratio"].ge(1.8)
         & rows["vol3_trend"].gt(1.1)
         & rows["day_ret"].between(0.005, 0.05)
         & rows["ret5"].le(0.08)
         & rows["close_pos"].gt(0.65)),
    )
    for label, mask in variants:
        print(f"{label}: {int(mask.sum())} signals")

    rows.loc[rows["vol_ratio"].between(1.3, 2.5) & structure & v1_common, "_keep"] = 1
    keep = rows[v0 | v1 | (structure & v1_common)
                | (v1_common & rows["close_pos"].gt(0.65))
                | (structure & rows["vol_ratio"].ge(1.8)
                   & rows["vol3_trend"].gt(1.1)
                   & rows["day_ret"].between(0.005, 0.05)
                   & rows["ret5"].le(0.08)
                   & rows["close_pos"].gt(0.65))].copy()
    keep.to_csv(SIGNALS_CSV, index=False)

    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    trade_dates = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2024-01-01", "2026-10-14")))

    store = BarStore(DATA_DIR)
    symbols = keep["ts_code"].dropna().unique()
    bars_map: dict[str, pd.DataFrame] = {}
    t0 = time.time()
    for symbol in symbols:
        try:
            bars = store.load(Instrument(symbol), "2024-01-01", "2100-01-01")
        except FileNotFoundError:
            continue
        bars = bars.copy()
        bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
        bars_map[symbol] = (bars.dropna(subset=["date"]).drop_duplicates("date")
                            .sort_values("date").reset_index(drop=True))
    print(f"loaded bars for {len(bars_map)}/{len(symbols)} symbols ({time.time() - t0:.0f}s)")

    out_rows = []
    for label, mask in variants:
        signals = rows.loc[mask, ["ts_code", "signal_date"]].rename(
            columns={"signal_date": "date"}).copy()
        signals["selected"] = True
        if signals.empty:
            out_rows.append({"variant": label, "holding_days": 5, "samples": 0,
                             "win_rate": None, "mean_net_return": None})
            continue
        events = evaluate_events(
            signals, bars_map, (5, 10), exit_modes=("profit_target_close",),
            trade_dates=trade_dates, profit_target=TARGET,
        )
        for record in summarize_events(events).to_dict("records"):
            out_rows.append({"variant": label, **record})
        print(f"evaluated {label} ({time.time() - started:.0f}s)")

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_CSV, index=False)
    print("\n=== gentle-volume trigger vs current startup (5% target) ===")
    print(out.to_string(index=False,
                        formatters={"win_rate": "{:.1%}".format,
                                    "mean_net_return": "{:+.2%}".format,
                                    "median_net_return": "{:+.2%}".format}))
    print(f"\ntotal: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
