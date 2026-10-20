"""Amortized multi-date scanning.

The per-date scanner re-reads and re-computes every symbol for every
observation date. This module inverts the loop: each symbol's bars are loaded
once, its technical features computed once, and the bar-coverage decision is
evaluated against precomputed per-date session sets instead of rebuilding
them per symbol. Scoring, gating and selection reuse the shared helpers from
batch_scan so both paths produce identical results.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd

from ..backtest.walk_forward import WalkForwardResult
from ..data.calendar import expected_sessions
from ..data.financial_store import FinancialStore
from ..data.models import Instrument
from ..data.store import BarStore
from ..features.technical import TechnicalConfig, calculate_technical, technical_score
from ..universe import filter_universe
from .batch_scan import (
    QUALITY_FIELDS,
    BatchScanConfig,
    OfflineCacheMissError,
    _coverage_row,
    _failure,
    _finalize_date_rows,
)


def _day_ints(index_or_series) -> np.ndarray:
    """Normalized dates as int64 day numbers for cheap set comparisons."""
    values = pd.DatetimeIndex(pd.to_datetime(index_or_series, errors="coerce").dropna())
    if values.empty:
        return np.empty(0, dtype=np.int64)
    return values.values.astype("datetime64[D]").astype(np.int64)


class _DateMeta:
    """Precomputed per-as-of coverage inputs (shared across all symbols)."""

    __slots__ = ("as_of", "as_of_day", "required_end", "required_end_day",
                 "required_minus3_day", "weekend_relax", "expected", "expected_first",
                 "expected_last")

    def __init__(self, config: BatchScanConfig, trade_dates: pd.DatetimeIndex | None) -> None:
        self.as_of = pd.Timestamp(config.as_of).normalize()
        self.as_of_day = np.datetime64(self.as_of, "D").astype(np.int64)
        self.required_end = min(pd.Timestamp(config.end).normalize(), self.as_of)
        self.required_end_day = np.datetime64(self.required_end, "D").astype(np.int64)
        self.required_minus3_day = np.datetime64(
            self.required_end - pd.Timedelta(days=3), "D"
        ).astype(np.int64)
        self.weekend_relax = self.required_end.weekday() >= 5
        if trade_dates is None:
            self.expected = None
            self.expected_first = None
            self.expected_last = None
        else:
            expected = expected_sessions(
                trade_dates, pd.Timestamp(config.start).normalize(), self.required_end
            )
            self.expected = _day_ints(expected)
            self.expected_first = int(self.expected[0]) if len(self.expected) else None
            self.expected_last = int(self.expected[-1]) if len(self.expected) else None


def _coverage_ok(avail_days: np.ndarray, start_day: int, meta: _DateMeta) -> bool:
    """Vectorized replica of batch_scan._bar_coverage using day integers."""
    if avail_days.size == 0:
        return False
    eligible_count = int(np.searchsorted(avail_days, meta.as_of_day, side="right"))
    if eligible_count == 0:
        return False
    last_eligible = int(avail_days[eligible_count - 1])
    if meta.expected is None:
        if int(avail_days[0]) > start_day:
            return False
    else:
        if meta.expected.size == 0:
            return False
        if int(avail_days[0]) > meta.expected_first:
            return False
    endpoint_reached = last_eligible >= meta.required_end_day
    if not endpoint_reached and meta.expected is not None and meta.expected.size:
        endpoint_reached = last_eligible >= meta.expected_last
    if not endpoint_reached and meta.expected is None and meta.weekend_relax:
        endpoint_reached = last_eligible >= meta.required_minus3_day
    if not endpoint_reached:
        return False
    if meta.expected is None or meta.expected.size == 0:
        return True
    # Mid-window suspensions are fine (features roll over the stock's own
    # trading days); only require that the stock traded on the signal day.
    return bool(np.isin(meta.expected[-1], avail_days))


def _scan_symbol_for_dates(job: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one symbol once and emit one snapshot row per eligible as-of date."""
    (ts_code, name, industry, template, date_metas, start_day, data_dir) = job
    store = BarStore(data_dir)
    financial_store = FinancialStore(data_dir)
    instrument = Instrument(ts_code, name=name)

    try:
        bars = store.load(instrument, template.start, template.end, adjust=template.adjust)
        cache_hit = True
    except FileNotFoundError:
        bars = pd.DataFrame()
        cache_hit = False

    features = pd.DataFrame()
    if not bars.empty:
        features = calculate_technical(
            bars, TechnicalConfig(max_volume_ratio=template.max_volume_ratio)
        )
        features["technical_score"] = technical_score(features)

    try:
        financials = financial_store.load(ts_code)
    except FileNotFoundError:
        financials = pd.DataFrame()

    bar_days = _day_ints(bars["date"]) if "date" in bars.columns else np.empty(0, dtype=np.int64)
    feat_days = (
        _day_ints(features["date"]) if "date" in features.columns
        else np.empty(0, dtype=np.int64)
    )
    asof_days = np.array([meta.as_of_day for meta in date_metas], dtype=np.int64)
    feat_index = np.searchsorted(feat_days, asof_days, side="right")

    stock = SimpleNamespace(ts_code=ts_code, name=name)
    failures: list[dict] = []
    kept: list[int] = []
    feat_rows: list[int] = []
    for position, meta in enumerate(date_metas):
        config = replace(template, as_of=meta.as_of)
        if not _coverage_ok(bar_days, start_day, meta):
            failures.append(_failure(
                stock, config, "bars", "cache_miss_provider_unavailable",
                OfflineCacheMissError(
                    f"cache miss or incomplete bar coverage for {instrument.ts_code}"
                ),
                cache_hit=True,
            ))
            continue
        if feat_index[position] == 0:
            failures.append(_failure(stock, config, "features", "missing_target_date",
                                     cache_hit=True))
            continue
        kept.append(position)
        feat_rows.append(feat_index[position] - 1)
    if not kept:
        return pd.DataFrame(), pd.DataFrame(failures)

    sub = features.iloc[feat_rows].reset_index(drop=True).copy()
    kept_days = asof_days[kept]
    sub["as_of"] = [date_metas[i].as_of for i in kept]
    sub["signal_date"] = sub["date"]
    sub["bars_start_date"] = bars["date"].iloc[0]
    sub["bars_last_date"] = bars["date"].iloc[-1]
    sub["ts_code"] = ts_code
    sub["name"] = name
    sub["industry"] = industry
    sub["bar_adjust"] = template.adjust
    sub["cache_hit"] = cache_hit

    if financials.empty:
        sub["financial_available_date"] = pd.NaT
        for column in QUALITY_FIELDS:
            sub[column] = pd.NA
    else:
        fin_days = _day_ints(financials["available_date"])
        order = np.argsort(fin_days, kind="stable")
        fin_sorted = financials.iloc[order].reset_index(drop=True)
        fin_index = np.searchsorted(fin_days[order], kept_days, side="right") - 1
        valid = fin_index >= 0
        chosen = fin_sorted.iloc[np.where(valid, fin_index, 0)]
        sub["financial_available_date"] = [
            pd.Timestamp(value) if ok else pd.NaT
            for value, ok in zip(chosen["available_date"].tolist(), valid)
        ]
        for column in QUALITY_FIELDS:
            sub[column] = [
                value if ok else pd.NA
                for value, ok in zip(chosen[column].tolist(), valid)
            ]
    return sub, pd.DataFrame(failures)


def fast_walk_forward_scan(
    universe: pd.DataFrame,
    template: BatchScanConfig,
    as_of_dates: Iterable[date | str],
    *,
    trade_dates: pd.DatetimeIndex | None,
    workers: int = 1,
    data_dir: str = "data",
) -> WalkForwardResult:
    """Scan many observation dates with one load/compute pass per symbol."""
    as_of_list = [pd.Timestamp(d).normalize() for d in as_of_dates]
    if not as_of_list:
        raise ValueError("at least one as-of date is required")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    date_metas = [_DateMeta(replace(template, as_of=d), trade_dates) for d in as_of_list]
    start_day = np.datetime64(pd.Timestamp(template.start).normalize(), "D").astype(np.int64)

    eligibility: dict[str, dict] = {}
    stocks_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for as_of, meta in zip(as_of_list, date_metas):
        stocks = filter_universe(
            universe, as_of=as_of, min_list_days=template.min_list_days
        )
        stocks_by_date[as_of] = stocks
        for stock in stocks.itertuples(index=False):
            entry = eligibility.setdefault(
                str(stock.ts_code),
                {"name": getattr(stock, "name", None),
                 "industry": getattr(stock, "industry", None), "dates": []},
            )
            entry["dates"].append(meta)

    jobs = [
        (ts_code, entry["name"], entry["industry"], template,
         entry["dates"], start_day, data_dir)
        for ts_code, entry in eligibility.items()
    ]

    if workers == 1:
        results = [_scan_symbol_for_dates(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
            results = list(pool.map(
                _scan_symbol_for_dates, jobs,
                chunksize=max(1, len(jobs) // (workers * 4)),
            ))

    all_rows = pd.concat([rows for rows, _ in results], ignore_index=True) \
        if any(not rows.empty for rows, _ in results) else pd.DataFrame()
    all_failures = pd.concat([failures for _, failures in results], ignore_index=True) \
        if any(not failures.empty for _, failures in results) else pd.DataFrame()
    rows_grouped = dict(tuple(all_rows.groupby("as_of"))) if not all_rows.empty else {}
    failures_grouped = dict(tuple(all_failures.groupby("as_of"))) if not all_failures.empty else {}

    candidate_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    for as_of, meta in zip(as_of_list, date_metas):
        config = replace(template, as_of=as_of)
        frame = rows_grouped.get(as_of)
        result = _finalize_date_rows([frame] if frame is not None else [], config)
        failure_frame = failures_grouped.get(as_of, pd.DataFrame())
        coverage = pd.DataFrame([_coverage_row(
            universe, stocks_by_date[as_of], config, result, failure_frame
        )])
        candidate_frames.append(result)
        failure_frames.append(failure_frame)
        coverage_frames.append(coverage)

    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    failures = pd.concat(failure_frames, ignore_index=True) \
        if any(not frame.empty for frame in failure_frames) else pd.DataFrame()
    coverage = pd.concat(coverage_frames, ignore_index=True)
    coverage["universe_snapshot_source"] = "single_current_snapshot_survivorship_limited"
    return WalkForwardResult(candidates, failures, coverage)
