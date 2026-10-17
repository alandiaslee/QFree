"""Sync cached daily bars from the last session to a target date via trade-date
batch fetch (the fast, row-cap-safe path), then report conflicts.

Usage:
    .venv/bin/python scripts/sync_bars.py [target_date] [--data-dir data] [--workers 12]

target_date defaults to the latest trade date known to Tushare.
After this, always run scripts/fix_qfq_conflicts.py to repair ex-dividend
staled qfq caches (the batch syncer leaves conflicting caches untouched).
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

sys.path.insert(0, "src")
from qfree.config import Settings
from qfree.data.batch_fetch import batch_prefetch_bars
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.data.tushare_client import TushareProvider


def latest_cached_date(store: BarStore, universe_path: str) -> pd.Timestamp | None:
    try:
        universe = pd.read_csv(universe_path)
        for ts_code in universe["ts_code"].astype(str).head(50):
            try:
                bars = store.load(Instrument(ts_code), "2020-01-01", "2100-01-01",
                                  adjust="qfq")
            except FileNotFoundError:
                continue
            if not bars.empty:
                return pd.Timestamp(bars["date"].max()).normalize()
        return None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_date", nargs="?", default=None,
                        help="sync up to this date (default: latest trade date)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--universe", default="data/universe.csv")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    settings = Settings.from_env()
    provider = TushareProvider(settings.require_token(), api_url="https://t.xiaodefa.top/",
                               request_timeout=30)
    if args.target_date:
        target = pd.Timestamp(args.target_date).normalize()
    else:
        days = provider.get_trade_dates("2020-01-01", "2100-01-01")
        target = pd.Timestamp(days.max()).normalize()
    store = BarStore(args.data_dir)
    last = latest_cached_date(store, args.universe)

    if last is not None and last >= target:
        print(f"cache already covers {target.date()} (last cached {last.date()}); nothing to do")
        return 0

    start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if last is not None \
        else "2025-01-01"
    trade_dates = pd.DatetimeIndex(pd.to_datetime(provider.get_trade_dates(start, target)))
    sessions = [str(d.date()) for d in trade_dates]
    if not sessions:
        print(f"no exchange sessions between {start} and {target.date()}")
        return 0
    print(f"sessions to fill: {sessions}")

    universe = pd.read_csv(args.universe)
    t0 = time.time()
    stats = batch_prefetch_bars(
        universe, provider, store,
        start=start, end=target.strftime("%Y-%m-%d"),
        trade_dates=trade_dates, workers=args.workers,
    )
    print(f"done in {time.time() - t0:.1f}s")
    print(stats.as_dict())
    if stats.symbols_conflict:
        print(f"next step: .venv/bin/python scripts/fix_qfq_conflicts.py "
              f"({stats.symbols_conflict} conflicts expected)")
    # Keep the CSI300 regime-gate index cache in sync (daily-scan reads it).
    from qfree.data.index_cache import DEFAULT_INDEX, update_index
    try:
        added = update_index(provider, args.data_dir, DEFAULT_INDEX)
        print(f"index cache {DEFAULT_INDEX}: +{added} sessions")
    except Exception as exc:
        print(f"warning: index cache update failed ({exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
