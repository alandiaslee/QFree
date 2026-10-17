"""Repair qfq caches whose derived returns no longer match the stored series.

Root cause: after an ex-dividend/split event the provider recomputes adj_factor,
which re-bases every historical qfq price, so a cache derived before the event
conflicts with qfq derived from today's raw+factor data. The batch prefetcher
leaves such caches untouched and counts a conflict; this tool re-derives qfq
from the CURRENT raw bars and factors already in the cache and overwrites the
stale series. Safe to run repeatedly: symbols whose qfq already match are
skipped untouched.

Usage: .venv/bin/python scripts/fix_qfq_conflicts.py [--data-dir data] [--workers 12]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, "src")
from qfree.data.batch_fetch import DERIVATION_MARKER, _load_factors, derive_qfq, returns_compatible
from qfree.data.models import Instrument
from qfree.data.store import BarStore

_WORKERS = 12
_DATA_DIR = "data"


def _fix_one(job: tuple[str, str | None, str]) -> tuple[str, str, str | None]:
    """Returns (ts_code, status, detail): ok_fixed | ok_match | no_raw | no_factor | failed."""
    ts_code, name, data_dir = job
    store = BarStore(data_dir)
    instrument = Instrument(ts_code, name=name)
    try:
        try:
            raw = store.load(instrument, "1900-01-01", "2100-01-01", adjust="")
        except FileNotFoundError:
            return ts_code, "no_raw", "raw cache missing"
        factors = _load_factors(store, instrument)
        if factors.empty:
            return ts_code, "no_factor", "factor cache missing"
        qfq = derive_qfq(raw, factors)
        if store.path(instrument, "qfq").exists():
            try:
                existing = store.load(instrument, "1900-01-01", "2100-01-01")
            except (FileNotFoundError, RuntimeError):
                existing = None
            if existing is not None and returns_compatible(existing, qfq):
                return ts_code, "ok_match", None
        store.save(
            instrument, qfq, adjust="qfq", source="conflict_fix",
            requested_start=raw["date"].iloc[0], requested_end=raw["date"].iloc[-1],
            extra_metadata={"derivation": DERIVATION_MARKER},
            overwrite=True,
        )
        return ts_code, "ok_fixed", None
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        return ts_code, "failed", str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--universe", default="data/universe.csv")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    started = time.time()
    universe = pd.read_csv(args.universe)
    jobs = [
        (str(row.ts_code), getattr(row, "name", None), args.data_dir)
        for row in universe.itertuples(index=False)
    ]
    print(f"checking {len(jobs)} symbols on {args.workers} workers ...")

    counts: dict[str, int] = {}
    problems: list[tuple[str, str, str | None]] = []
    fixed: list[str] = []
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for index, (ts_code, status, detail) in enumerate(
            pool.map(_fix_one, jobs, chunksize=max(1, len(jobs) // (args.workers * 4))),
            start=1,
        ):
            counts[status] = counts.get(status, 0) + 1
            if status == "ok_fixed":
                fixed.append(ts_code)
            if status not in ("ok_fixed", "ok_match"):
                problems.append((ts_code, status, detail))
            if index % 1000 == 0:
                print(f"[{index}/{len(jobs)}] {time.time() - started:.0f}s", flush=True)

    print(f"done in {time.time() - started:.1f}s: {counts}")
    if fixed:
        print(f"fixed {len(fixed)} qfq caches: {' '.join(fixed[:20])}"
              + (" ..." if len(fixed) > 20 else ""))
    for ts_code, status, detail in problems:
        print(f"  {ts_code}: {status} ({detail})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
