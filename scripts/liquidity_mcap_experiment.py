"""Compare liquidity (20d avg turnover) and market-cap band filters on the
final-config signal pool, using the expanded walk-forward candidates.

Variants are post-filters on the AND-gate base pool (startup signal, breadth>=1%,
market_return_5d>=1.7%, base score>=60, financial fields>=3). Event study uses
the 5% profit-target / 5-day holding exit. Also runs the same variants on the
looser breadth-only pool for a larger sample view.

Efficiency: bars load once per symbol in the parent; daily_basic is fetched
only for the handful of trigger dates (cached to outputs/daily_basic_<date>.csv,
reused by daily-scan --min-market-cap).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from qfree.backtest.event_study import evaluate_events, summarize_events
from qfree.config import Settings
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider

CANDIDATES_CSV = "outputs/walk_exp_candidates.csv"
OUT_CSV = "outputs/liquidity_mcap_summary.csv"
BREADTH = 0.01
MARKET_RET5 = 0.017
MIN_FINANCIAL_FIELDS = 3
MIN_SCORE = 60.0
TARGET = 0.05
HOLDING = 5
# Bars store amount in thousand CNY.
AMT_5000W = 5e4     # 千元 == 5000 万元
AMT_1Y = 1e5        # 千元 == 1 亿元
MCAP_MIN = 30e4     # 万元 == 30 亿
MCAP_MAX = 1000e4   # 万元 == 1000 亿


def _load_bars(symbol: str, store: BarStore) -> pd.DataFrame | None:
    try:
        bars = store.load(Instrument(symbol), "1900-01-01", "2100-01-01")
    except FileNotFoundError:
        return None
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    bars = bars.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    bars["avg_amount_20d"] = bars["amount"].astype(float).rolling(20).mean()
    return bars.reset_index(drop=True)


def _at_date(frame: pd.DataFrame, day: pd.Timestamp, column: str) -> float | None:
    positions = frame.index[frame["date"] <= day]
    if len(positions) == 0:
        return None
    value = frame.at[positions[-1], column]
    return None if pd.isna(value) else float(value)


def main() -> int:
    started = time.time()
    cand = pd.read_csv(CANDIDATES_CSV, parse_dates=["signal_date", "as_of"])
    startup = cand["startup_signal"].astype(str).str.lower().isin(("true", "1"))
    fin_ok = cand["financial_field_count"] >= MIN_FINANCIAL_FIELDS
    score_ok = cand["base_total_score"] >= MIN_SCORE
    quality = pd.to_numeric(cand["quality_score"], errors="coerce")
    breadth_ok = cand["startup_signal_breadth"] >= BREADTH

    pools = {
        "AND": startup & breadth_ok & (cand["market_return_5d"] >= MARKET_RET5) & fin_ok & score_ok,
        "breadth": startup & breadth_ok & fin_ok & score_ok,
    }

    # --- liquidity from cached bars (parent, pool symbols only) ---
    union_mask = pools["AND"] | pools["breadth"]
    symbols = cand.loc[union_mask, "ts_code"].dropna().unique()
    store = BarStore("data")
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    t0 = time.time()
    for symbol in symbols:
        frame = _load_bars(symbol, store)
        if frame is not None:
            bars_by_symbol[symbol] = frame
    print(f"loaded {len(bars_by_symbol)}/{len(symbols)} symbols in {time.time() - t0:.1f}s")

    sig_dates = pd.to_datetime(cand.loc[union_mask, "signal_date"])
    avg_amount = pd.Series(float("nan"), index=cand.index, dtype="float64")
    for idx in cand.index[union_mask]:
        frame = bars_by_symbol.get(str(cand.at[idx, "ts_code"]))
        if frame is not None:
            avg_amount.at[idx] = _at_date(frame, sig_dates.at[idx], "avg_amount_20d")
    cand["avg_amount_20d"] = avg_amount

    # --- market cap from daily_basic (fetch only trigger dates, cache) ---
    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    trigger_dates = sorted(cand.loc[union_mask, "signal_date"].dt.strftime("%Y-%m-%d").unique())
    basic_frames = []
    for day in trigger_dates:
        cache_path = Path(f"outputs/daily_basic_{day}.csv")
        if cache_path.exists():
            frame = pd.read_csv(cache_path)
        else:
            frame = provider._pro.daily_basic(trade_date=day.replace("-", ""))
            frame.to_csv(cache_path, index=False)
        basic_frames.append(frame[["ts_code", "total_mv"]].assign(signal_date=pd.Timestamp(day)))
    basic = pd.concat(basic_frames, ignore_index=True).drop_duplicates(["ts_code", "signal_date"])
    cand = cand.merge(basic, on=["ts_code", "signal_date"], how="left")
    print(f"daily_basic ready for {len(trigger_dates)} dates: {trigger_dates}")

    # --- variants ---
    def make_variant(label: str, mask: pd.Series) -> tuple[str, pd.DataFrame]:
        signals = cand.loc[mask, ["ts_code", "signal_date", "name"]].rename(
            columns={"signal_date": "date"}).copy()
        signals["selected"] = True
        return label, signals

    variants: list[tuple[str, pd.DataFrame]] = []
    for pool_name, pool in pools.items():
        variants.append(make_variant(f"{pool_name}|base", pool))
        variants.append(make_variant(
            f"{pool_name}|amt>=5000w",
            pool & cand["avg_amount_20d"].ge(AMT_5000W)))
        variants.append(make_variant(
            f"{pool_name}|amt>=1y",
            pool & cand["avg_amount_20d"].ge(AMT_1Y)))
        mv = pd.to_numeric(cand["total_mv"], errors="coerce")
        variants.append(make_variant(
            f"{pool_name}|mcap30-1000",
            pool & mv.between(MCAP_MIN, MCAP_MAX)))
        variants.append(make_variant(
            f"{pool_name}|amt5000w+mcap",
            pool & cand["avg_amount_20d"].ge(AMT_5000W) & mv.between(MCAP_MIN, MCAP_MAX)))
        variants.append(make_variant(
            f"{pool_name}|amt5000w+mcap+q55",
            pool & cand["avg_amount_20d"].ge(AMT_5000W) & mv.between(MCAP_MIN, MCAP_MAX)
            & (quality >= 55)))
    for label, signals in variants:
        print(f"{label}: {len(signals)} signals")

    trade_dates = pd.DatetimeIndex(pd.to_datetime(
        provider.get_trade_dates("2025-01-01", "2026-10-14")))
    rows = []
    for label, signals in variants:
        events = evaluate_events(
            signals, bars_by_symbol, (HOLDING,),
            exit_modes=("profit_target_close",),
            trade_dates=trade_dates, profit_target=TARGET,
        )
        summary = summarize_events(events)
        for record in summary.to_dict("records"):
            rows.append({"variant": label, **record})
        print(f"evaluated {label} ({time.time() - started:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    view = out[["variant", "samples", "win_rate", "mean_net_return", "median_net_return"]]
    view = view.sort_values(["variant"], kind="stable")
    print()
    print(view.to_string(index=False,
                         formatters={"win_rate": "{:.1%}".format,
                                     "mean_net_return": "{:+.2%}".format,
                                     "median_net_return": "{:+.2%}".format}))
    print(f"\ntotal: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
