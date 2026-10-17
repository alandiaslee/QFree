"""Bottom-consolidation (xushi) factor experiment.

Two questions:
A. Do consolidation factors improve the EXISTING right-side startup signal?
   Variants add "main filters" (120d drawdown band, distance to 60d low,
   no new lows in last 10 sessions) and "convergence bonus" (volatility
   convergence, MA convergence, MA20 slope) to the breadth/AND pools.
B. Do the consolidation factors work as a STANDALONE left-side signal?
   Full-universe scan over the 85 walk-forward dates (fork pool, one bars
   load per symbol), evaluated with 5/10/20d holdings.

All factors are vectorized per symbol from cached qfq bars; the candidates
CSV supplies the scan dates and the per-date startup-breadth gate.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from qfree.backtest.event_study import evaluate_events, summarize_events
from qfree.config import Settings
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider

CANDIDATES_CSV = "outputs/walk_exp_candidates.csv"
OUT_A = "outputs/xushi_variant_summary.csv"
OUT_B = "outputs/xushi_standalone_summary.csv"
BREADTH = 0.01
MIN_FINANCIAL_FIELDS = 3
MIN_SCORE = 60.0
TARGET = 0.05
WORKERS = 12

_DATA_DIR = "data"
_ASO_DAYS: np.ndarray | None = None


def add_xushi(frame: pd.DataFrame) -> pd.DataFrame:
    """Vectorized consolidation factors; all point-in-time (rolling backward)."""
    close = frame["close"].astype(float)
    low = frame["low"].astype(float)
    ret = close.pct_change()
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    out = frame.copy()
    out["dd120"] = close / close.rolling(120).max() - 1.0
    low60 = low.rolling(60).min()
    out["dist_low60"] = close / low60 - 1.0
    prior_low = low.shift(10).rolling(50).min()
    out["low_stable"] = low.rolling(10).min().ge(prior_low * 0.97)
    out["vol_conv"] = ret.rolling(10).std() / ret.rolling(20).std()
    ma_min = pd.concat([ma5, ma10, ma20], axis=1).min(axis=1)
    ma_max = pd.concat([ma5, ma10, ma20], axis=1).max(axis=1)
    out["ma_conv"] = (ma_max - ma_min) / ma20
    out["ma20_slope"] = ma20 / ma20.shift(5) - 1.0
    out["conv_bonus"] = (
        out["vol_conv"].lt(0.8).astype(float)
        + out["ma_conv"].lt(0.05).astype(float)
        + out["ma20_slope"].gt(-0.005).astype(float)
    )
    return out


def _extract_at_dates(frame: pd.DataFrame, aso_days: np.ndarray) -> pd.DataFrame:
    """One row per as-of date, from the latest session strictly on/before it."""
    days = frame["date"].values.astype("datetime64[D]").astype(np.int64)
    positions = np.searchsorted(days, aso_days, side="right") - 1
    valid = positions >= 0
    if not valid.any():
        return pd.DataFrame()
    picked = frame.iloc[positions[valid]].copy()
    picked["as_of"] = pd.to_datetime(aso_days[valid], unit="D")
    return picked


def _scan_symbol(ts_code: str) -> pd.DataFrame:
    store = BarStore(_DATA_DIR)
    try:
        bars = store.load(Instrument(ts_code), "2024-01-01", "2100-01-01")
    except FileNotFoundError:
        return pd.DataFrame()
    if len(bars) < 30:
        return pd.DataFrame()
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    bars = bars.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    assert _ASO_DAYS is not None
    factors = add_xushi(bars)
    rows = _extract_at_dates(factors, _ASO_DAYS)
    if rows.empty:
        return rows
    rows["ts_code"] = ts_code
    keep = ["ts_code", "as_of", "dd120", "dist_low60", "low_stable", "vol_conv",
            "ma_conv", "ma20_slope", "conv_bonus"]
    return rows[keep].reset_index(drop=True)


def _load_bars(symbol: str, store: BarStore) -> pd.DataFrame | None:
    try:
        bars = store.load(Instrument(symbol), "2024-01-01", "2100-01-01")
    except FileNotFoundError:
        return None
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    bars = bars.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    return bars.reset_index(drop=True)


def main() -> int:
    global _ASO_DAYS
    started = time.time()
    cand = pd.read_csv(CANDIDATES_CSV, parse_dates=["signal_date", "as_of"])
    aso_dates = sorted(cand["as_of"].unique())
    _ASO_DAYS = pd.DatetimeIndex(aso_dates).values.astype("datetime64[D]").astype(np.int64)
    startup = cand["startup_signal"].astype(str).str.lower().isin(("true", "1"))
    fin_ok = cand["financial_field_count"] >= MIN_FINANCIAL_FIELDS
    score_ok = cand["base_total_score"] >= MIN_SCORE
    breadth_ok = cand["startup_signal_breadth"] >= BREADTH
    gate_dates = set(cand.loc[breadth_ok, "as_of"].dt.normalize())

    # ---------- Part A: consolidation factors on the startup pools ----------
    pools = {
        "AND": startup & breadth_ok & (cand["market_return_5d"] >= 0.017) & fin_ok & score_ok,
        "breadth": startup & breadth_ok & fin_ok & score_ok,
    }
    store = BarStore("data")
    pool_symbols = sorted(cand.loc[pools["breadth"] | pools["AND"], "ts_code"].dropna().unique())
    factor_rows = []
    for symbol in pool_symbols:
        bars = _load_bars(symbol, store)
        if bars is None:
            continue
        factors = add_xushi(bars)
        picked = _extract_at_dates(
            factors, pd.DatetimeIndex(cand.loc[pools["breadth"] | pools["AND"], "signal_date"].unique())
            .values.astype("datetime64[D]").astype(np.int64))
        if not picked.empty:
            picked["ts_code"] = symbol
            factor_rows.append(picked[["ts_code", "as_of", "dd120", "dist_low60",
                                       "low_stable", "vol_conv", "ma_conv",
                                       "ma20_slope", "conv_bonus"]])
    xushi = pd.concat(factor_rows, ignore_index=True) if factor_rows else pd.DataFrame()
    cand = cand.merge(xushi, left_on=["ts_code", "signal_date"],
                      right_on=["ts_code", "as_of"], how="left", suffixes=("", "_x"))
    print(f"part A: xushi factors merged for {len(pool_symbols)} pool symbols "
          f"({time.time() - started:.0f}s)")

    dd_ok = cand["dd120"].between(-0.45, -0.15)
    dist_ok = cand["dist_low60"].le(0.15)
    stable_ok = cand["low_stable"].fillna(False).astype(bool)
    main_ok = dd_ok & dist_ok & stable_ok
    conv_ok = cand["conv_bonus"].ge(1)

    bars_cache: dict[str, pd.DataFrame | None] = {}

    def ev(mask: pd.Series, label: str, holdings: tuple[int, ...],
           frame: pd.DataFrame | None = None) -> list[dict]:
        source = cand if frame is None else frame
        signals = source.loc[mask, ["ts_code", "signal_date"]].rename(
            columns={"signal_date": "date"}).copy()
        signals["selected"] = True
        if signals.empty:
            return [{"variant": label, "samples": 0, "win_rate": None,
                     "mean_net_return": None}]
        symbols = signals["ts_code"].dropna().unique()
        for symbol in symbols:
            if symbol not in bars_cache:
                bars_cache[symbol] = _load_bars(symbol, store)
        bars_map = {s: b for s, b in bars_cache.items() if b is not None and s in set(symbols)}
        events = evaluate_events(
            signals, bars_map, holdings, exit_modes=("profit_target_close",),
            trade_dates=TRADE_DATES, profit_target=TARGET,
        )
        summary = summarize_events(events)
        return [{"variant": label, **record} for record in summary.to_dict("records")]

    from qfree.config import Settings as _S
    settings = _S.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    global TRADE_DATES
    TRADE_DATES = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2024-01-01", "2026-10-14")))

    rows_a = []
    for pool_name, pool in pools.items():
        rows_a += ev(pool, f"{pool_name}|base", (5,))
        rows_a += ev(pool & main_ok, f"{pool_name}|main_filters", (5,))
        rows_a += ev(pool & stable_ok, f"{pool_name}|low_stable_only", (5,))
        rows_a += ev(pool & dd_ok, f"{pool_name}|dd120_band_only", (5,))
        rows_a += ev(pool & main_ok & conv_ok, f"{pool_name}|main+conv>=1", (5,))
    out_a = pd.DataFrame(rows_a)
    out_a.to_csv(OUT_A, index=False)
    print("\n=== Part A: startup signal + consolidation overlays (5d, 5% target) ===")
    print(out_a.to_string(index=False,
                          formatters={"win_rate": "{:.1%}".format,
                                      "mean_net_return": "{:+.2%}".format,
                                      "median_net_return": "{:+.2%}".format}))
    print(f"part A done ({time.time() - started:.0f}s)")

    # ---------- Part B: standalone left-side consolidation signal ----------
    print(f"\npart B: scanning full universe x {len(aso_dates)} dates ...")
    universe = pd.read_csv("data/universe.csv")
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        parts = list(pool.map(_scan_symbol, universe["ts_code"].astype(str).tolist(),
                              chunksize=32))
    scan = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    scan = scan[scan["as_of"].isin(gate_dates)]
    print(f"part B: {len(scan)} symbol-date rows on gate-pass dates ({time.time() - started:.0f}s)")

    s_dd = scan["dd120"].between(-0.45, -0.15)
    s_dist = scan["dist_low60"].le(0.15)
    s_stable = scan["low_stable"].fillna(False).astype(bool)
    b1 = s_dd & s_dist & s_stable
    b2 = b1 & scan["conv_bonus"].ge(1)
    print(f"B1 main-filter signals: {int(b1.sum())}, B2 +convergence: {int(b2.sum())}")

    scan = scan.rename(columns={"as_of": "signal_date"})
    rows_b = []
    for label, mask, holdings in (
        ("B1|main_filters", b1, (5, 10, 20)),
        ("B2|main+conv>=1", b2, (5, 10, 20)),
    ):
        rows_b += ev(mask, label, holdings, frame=scan)
    out_b = pd.DataFrame(rows_b)
    out_b.to_csv(OUT_B, index=False)
    print("\n=== Part B: standalone consolidation signal (breadth-gate dates) ===")
    print(out_b.to_string(index=False,
                          formatters={"win_rate": "{:.1%}".format,
                                      "mean_net_return": "{:+.2%}".format,
                                      "median_net_return": "{:+.2%}".format}))
    print(f"\ntotal: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
