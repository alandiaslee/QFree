"""Multi-date orchestration for point-in-time screening validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd

from ..data.financial_store import FinancialStore
from ..data.store import BarStore
from ..data.tushare_client import TushareProvider
from ..features.technical import TechnicalConfig
from ..strategy.batch_scan import BatchScanConfig, BatchScanResult, batch_scan_detailed


@dataclass(frozen=True)
class WalkForwardResult:
    """Concatenated scan outputs with one coverage row per observation date."""

    candidates: pd.DataFrame
    failures: pd.DataFrame
    coverage: pd.DataFrame


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _scan_date_worker(payload: tuple) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan one cutoff in a worker process (fresh provider/store instances).

    Dates are independent (each reads the same immutable CSV caches and only
    writes its own in-memory frames), so the loop parallelizes cleanly across
    processes, bypassing the GIL for pandas-heavy per-symbol work.
    """
    (universe, config, trade_dates, data_dir, token, api_url, request_timeout) = payload
    provider = (
        TushareProvider(token, api_url=api_url, request_timeout=request_timeout)
        if token else None
    )
    store = BarStore(Path(data_dir))
    financial_store = FinancialStore(Path(data_dir))
    report = batch_scan_detailed(
        universe, provider, store, financial_store, config,
        trade_dates=trade_dates,
    )
    return report.candidates, report.failures, report.coverage


def walk_forward_scan(
    universe: pd.DataFrame,
    provider: TushareProvider,
    store: BarStore,
    financial_store: FinancialStore,
    as_of_dates: Sequence[date | str],
    *,
    start: date | str,
    end: date | str | None = None,
    config: BatchScanConfig | None = None,
    technical_config: TechnicalConfig | None = None,
    trade_dates: pd.DatetimeIndex | None = None,
    universe_by_as_of: Mapping[date | str, pd.DataFrame] | None = None,
    workers: int | None = None,
    data_dir: str | Path | None = None,
    token: str | None = None,
    api_url: str | None = None,
    request_timeout: float | None = None,
) -> WalkForwardResult:
    """Run the same frozen scanner independently at each information cutoff.

    Each scan only requests bars through its own observation date.  The
    resulting tables retain ``as_of`` so dates cannot be accidentally pooled
    without an explicit decision by the caller.

    With ``workers`` set, each cutoff runs in its own process (the provider
    and stores are rebuilt from ``data_dir``/``token`` inside the worker);
    otherwise the caller's instances are reused serially.
    """
    if not as_of_dates:
        raise ValueError("at least one as-of date is required")
    base = config or BatchScanConfig(
        as_of=pd.Timestamp(as_of_dates[0]).date(),
        start=start,
        end=end or as_of_dates[0],
    )
    payloads: list[tuple] = []
    if workers is not None:
        if workers < 1:
            raise ValueError("workers must be at least 1 when set")
        if data_dir is None or token is None:
            raise ValueError("parallel scan requires data_dir and token")
    for as_of in as_of_dates:
        cutoff = pd.Timestamp(as_of).normalize()
        requested_end = pd.Timestamp(end).normalize() if end is not None else cutoff
        scan_end = min(cutoff, requested_end)
        current = replace(base, as_of=cutoff, start=start, end=scan_end)
        current_universe = universe
        if universe_by_as_of is not None:
            current_universe = universe_by_as_of.get(as_of, universe_by_as_of.get(cutoff, pd.DataFrame()))
        if workers is not None:
            payloads.append((current_universe, current, trade_dates,
                             str(data_dir), token, api_url, request_timeout))
    if workers is not None and len(payloads) > 1:
        if data_dir is None or token is None:
            raise ValueError("parallel scan requires data_dir and token")
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as pool:
            results = list(pool.map(_scan_date_worker, payloads))
    else:
        results = _run_serial(
            universe, as_of_dates, base, start, end, universe_by_as_of,
            provider, store, financial_store, technical_config, trade_dates,
        )
    reports = [BatchScanResult(c, f, cov) for c, f, cov in results]

    candidates = _concat([report.candidates for report in reports])
    failures = _concat([report.failures for report in reports])
    coverage = _concat([report.coverage for report in reports])
    if not coverage.empty:
        coverage["universe_snapshot_source"] = (
            "universe_by_as_of" if universe_by_as_of is not None
            else "single_current_snapshot_survivorship_limited"
        )
    return WalkForwardResult(candidates, failures, coverage)


def _run_serial(universe, as_of_dates, base, start, end, universe_by_as_of,
                provider, store, financial_store, technical_config, trade_dates):
    results = []
    for as_of in as_of_dates:
        cutoff = pd.Timestamp(as_of).normalize()
        requested_end = pd.Timestamp(end).normalize() if end is not None else cutoff
        scan_end = min(cutoff, requested_end)
        current = replace(base, as_of=cutoff, start=start, end=scan_end)
        current_universe = universe
        if universe_by_as_of is not None:
            current_universe = universe_by_as_of.get(as_of, universe_by_as_of.get(cutoff, pd.DataFrame()))
        report = batch_scan_detailed(
            current_universe, provider, store, financial_store, current,
            technical_config=technical_config,
            trade_dates=trade_dates,
        )
        results.append((report.candidates, report.failures, report.coverage))
    return results
