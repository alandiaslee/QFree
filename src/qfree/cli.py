"""Command-line entry points for offline screening and Tushare ingestion."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .backtest.event_study import evaluate_events, summarize_events
from .backtest.report import evaluate_walk_forward_events
from .backtest.walk_forward import walk_forward_scan
from .config import Settings
from .data.financial_store import FinancialStore
from .data.models import Instrument
from .data.store import BarStore
from .data.tushare_client import TushareProvider
from .strategy.batch_scan import BatchScanConfig, batch_scan_detailed
from .strategy.fast_scan import fast_walk_forward_scan
from .strategy.screener import screen
from .universe import filter_universe


def _selected_values(frame: pd.DataFrame) -> pd.Series:
    values = frame.get("selected", pd.Series(False, index=frame.index))
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.lower().isin(("true", "1", "yes", "y"))


def _prefetch_one(
    stock: object,
    provider: TushareProvider,
    store: BarStore,
    financial_store: FinancialStore,
    config: BatchScanConfig,
) -> tuple[int, int]:
    """Populate one symbol's caches; per-symbol files make this thread-safe."""
    instrument = Instrument(str(stock.ts_code), name=getattr(stock, "name", None))
    bars_result = 0
    financial_result = 0
    try:
        cached = store.load(instrument, config.start, config.end, adjust=config.adjust)
        complete = not cached.empty
    except (FileNotFoundError, RuntimeError):
        complete = False
    if not complete:
        try:
            bars = provider.get_daily_bars(instrument, config.start, config.end, config.adjust)
            if bars.empty:
                bars_result = -1
            else:
                store.save(
                    instrument, bars, adjust=config.adjust, source=provider.name,
                    requested_start=config.start, requested_end=config.end,
                    extra_metadata={"provider": provider.name},
                )
                bars_result = 1
        except (RuntimeError, ValueError, TypeError):
            bars_result = -1
    try:
        financial_store.as_of(instrument.ts_code, config.as_of)
    except FileNotFoundError:
        try:
            raw_financials = provider.financial_snapshot(
                instrument.ts_code, start_date=None, end_date=config.as_of
            )
            if raw_financials.empty:
                financial_result = -1
            else:
                financial_store.save(raw_financials, source=provider.name)
                financial_result = 1
        except (RuntimeError, ValueError, TypeError):
            financial_result = -1
    return bars_result, financial_result


def _fetch_financial_chunk(
    provider: TushareProvider,
    data_dir: Path,
    codes: list[str],
    as_of: str,
) -> tuple[int, int]:
    """Fetch and persist one multi-symbol financial request."""
    try:
        frame = provider.financial_snapshot_batch(
            codes, start_date=None, end_date=as_of,
        )
        if frame.empty or "ts_code" not in frame:
            return 0, len(codes)
        store = FinancialStore(data_dir)
        fetched = failed = 0
        for code in codes:
            subset = frame[frame["ts_code"].eq(code)].copy()
            if subset.empty:
                failed += 1
            else:
                store.save(subset, source=provider.name)
                fetched += 1
        return fetched, failed
    except (RuntimeError, ValueError, TypeError, KeyError):
        return 0, len(codes)


def _prefetch_financial_batch(
    stocks: pd.DataFrame,
    provider: TushareProvider,
    data_dir: Path,
    as_of: str,
    *,
    max_symbols: int | None = None,
    chunk_size: int = 50,
    workers: int | None = None,
) -> dict[str, int]:
    """Fill missing financial caches with concurrent multi-symbol requests."""
    financial_store = FinancialStore(data_dir)
    rows = list(stocks.itertuples(index=False))
    if max_symbols is not None:
        rows = rows[:max_symbols]
    missing = []
    for stock in rows:
        code = str(stock.ts_code)
        try:
            financial_store.as_of(code, as_of)
        except FileNotFoundError:
            missing.append(code)
    size = max(1, chunk_size)
    chunks = [missing[offset:offset + size] for offset in range(0, len(missing), size)]
    if not chunks:
        return {
            "financial_missing": 0, "financial_fetched": 0,
            "financial_failed": 0, "financial_api_batches": 0,
        }
    worker_count = max(1, workers or int(os.getenv("QFREE_PREFETCH_WORKERS", "12")))
    fetched = failed = 0
    with ThreadPoolExecutor(max_workers=min(worker_count, len(chunks))) as pool:
        jobs = [pool.submit(_fetch_financial_chunk, provider, data_dir, chunk, as_of)
                for chunk in chunks]
        for index, job in enumerate(jobs, start=1):
            got, bad = job.result()
            fetched += got
            failed += bad
            print(f"financial batch prefetch: {index}/{len(chunks)} chunks", flush=True)
    return {
        "financial_missing": len(missing),
        "financial_fetched": fetched,
        "financial_failed": failed,
        "financial_api_batches": len(chunks),
    }


def _prefetch_missing_data(
    stocks: pd.DataFrame,
    provider: TushareProvider,
    data_dir: Path,
    config: BatchScanConfig,
    *,
    max_symbols: int | None = None,
    workers: int | None = None,
) -> dict[str, int]:
    """Populate local CSV caches once so later scans can run mostly offline.

    Each symbol is independent (separate cache files), so the loop runs
    in a thread pool; symbols are re-counted in input order afterwards.
    """
    store = BarStore(data_dir)
    financial_store = FinancialStore(data_dir)
    rows = list(stocks.itertuples(index=False))
    if max_symbols is not None:
        rows = rows[:max_symbols]
    workers = workers or int(os.getenv("QFREE_PREFETCH_WORKERS", "6"))
    bars_fetched = 0
    bars_failed = 0
    financial_fetched = 0
    financial_failed = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for bars_result, financial_result in pool.map(
            _prefetch_one, rows, [provider] * len(rows),
            [store] * len(rows), [financial_store] * len(rows),
            [config] * len(rows),
        ):
            done += 1
            if bars_result == 1:
                bars_fetched += 1
            elif bars_result == -1:
                bars_failed += 1
            if financial_result == 1:
                financial_fetched += 1
            elif financial_result == -1:
                financial_failed += 1
            if done % 25 == 0 or done == len(rows):
                print(f"prefetch: {done}/{len(rows)} symbols", flush=True)
    return {
        "bars_fetched": bars_fetched,
        "bars_failed": bars_failed,
        "financial_fetched": financial_fetched,
        "financial_failed": financial_failed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qfree")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="fetch one symbol's daily bars")
    fetch.add_argument("symbol", help="six-digit A-share code, e.g. 002487")
    fetch.add_argument("--start", default="2020-01-01")
    fetch.add_argument("--end", default=None)
    fetch.add_argument("--adjust", choices=("qfq", "hfq", ""), default="qfq")
    fetch.add_argument("--data-dir", type=Path, default=None)
    fetch.add_argument("--api-url", default=None, help="optional explicitly authorized gateway URL")
    scan = sub.add_parser("scan", help="screen a cached symbol")
    scan.add_argument("symbol")
    scan.add_argument("--data-dir", type=Path, default=None)
    scan.add_argument("--min-score", type=float, default=60.0)

    universe = sub.add_parser("universe", help="download and filter the current Shanghai/Shenzhen universe")
    universe.add_argument("--output", type=Path, default=Path("data/universe.csv"))
    universe.add_argument("--as-of", default=None)
    universe.add_argument("--api-url", default=None)

    financials = sub.add_parser("financials", help="download one symbol's financial indicators")
    financials.add_argument("symbol")
    financials.add_argument("--start", default="2020-01-01")
    financials.add_argument("--end", default=None)
    financials.add_argument("--data-dir", type=Path, default=None)
    financials.add_argument("--api-url", default=None)

    walk = sub.add_parser("walk-forward", help="scan multiple point-in-time observation dates")
    walk.add_argument("--as-of", action="append", required=True,
                      help="observation date; repeat for multiple dates")
    walk.add_argument("--start", default="2025-01-01")
    walk.add_argument("--end", default=None)
    walk.add_argument("--data-dir", type=Path, default=None)
    walk.add_argument("--universe", type=Path, default=Path("data/universe.csv"))
    walk.add_argument("--universe-by-as-of-dir", type=Path, default=None,
                      help="directory containing YYYY-MM-DD.csv frozen universe snapshots")
    walk.add_argument("--api-url", default=None)
    walk.add_argument("--limit", type=int, default=None)
    walk.add_argument("--prefetch", action="store_true",
                      help="populate missing bars/financial caches before scanning")
    walk.add_argument("--prefetch-batch", action="store_true",
                      help="prefetch bars whole-market per session via daily+adj_factor "
                           "(much faster for many symbols; requires trade calendar)")
    walk.add_argument("--prefetch-limit", type=int, default=None,
                      help="limit prefetch to first N filtered symbols")
    walk.add_argument("--prefetch-workers", type=int, default=None,
                      help="parallel prefetch threads (default: QFREE_PREFETCH_WORKERS or 6)")
    walk.add_argument("--scan-workers", type=int, default=None,
                      help="scan observation dates in parallel processes")
    walk.add_argument("--offline-scan", action="store_true",
                      help="scan only from local caches; missing bars are reported as failures")
    walk.add_argument("--fast-scan", action="store_true",
                      help="amortized multi-date scan: load/compute each symbol once "
                           "(results identical to the per-date scanner; requires --offline-scan)")
    walk.add_argument("--request-timeout", type=float, default=30.0,
                      help="per-request network timeout in seconds")
    walk.add_argument("--strategy-a", action="store_true",
                      help="enable market gate, pullback-quality boost, and profit-target close exits")
    walk.add_argument("--min-startup-breadth", type=float, default=0.01,
                      help="strategy-a market gate startup-signal breadth threshold")
    walk.add_argument("--min-market-ret5", type=float, default=0.017,
                      help="strategy-a market gate 5-day equal-weight return threshold")
    walk.add_argument("--max-boost-return-20d", type=float, default=-0.006,
                      help="strategy-a maximum 20-day return for quality boost eligibility")
    walk.add_argument("--min-boost-ocf-profit", type=float, default=-13.0,
                      help="strategy-a minimum OCF/profit for quality boost eligibility")
    walk.add_argument("--max-boost-ocf-profit", type=float, default=98.0,
                      help="strategy-a maximum OCF/profit for quality boost eligibility")
    walk.add_argument("--boost-points", type=float, default=5.0,
                      help="score points added by strategy-a quality boost")
    walk.add_argument("--profit-target", type=float, default=0.06,
                      help="strategy-a close-based profit target")
    walk.add_argument("--output", type=Path, default=Path("outputs/walk_candidates.csv"))
    walk.add_argument("--failures-output", type=Path, default=Path("outputs/walk_failures.csv"))
    walk.add_argument("--coverage-output", type=Path, default=Path("outputs/walk_coverage.csv"))
    walk.add_argument("--events-output", type=Path, default=Path("outputs/walk_events.csv"))
    walk.add_argument("--event-summary-output", type=Path, default=Path("outputs/walk_event_summary.csv"))
    walk.add_argument("--holdings", default=None,
                      help="comma-separated max holding days (default: 3,5 with --strategy-a, else 5,10,20)")
    walk.add_argument("--exit-modes", default="fixed_open")
    walk.add_argument("--min-score", type=float, default=60.0)

    batch = sub.add_parser("batch-scan", help="scan a universe snapshot with failures reported")
    batch.add_argument("--as-of", required=True)
    batch.add_argument("--start", default="2025-01-01")
    batch.add_argument("--end", default=None)
    batch.add_argument("--data-dir", type=Path, default=None)
    batch.add_argument("--universe", type=Path, default=Path("data/universe.csv"))
    batch.add_argument("--api-url", default=None)
    batch.add_argument("--limit", type=int, default=None,
                       help="optional explicit sample size after universe filtering")
    batch.add_argument("--prefetch", action="store_true",
                       help="populate missing bars/financial caches before scanning")
    batch.add_argument("--prefetch-batch", action="store_true",
                       help="prefetch bars whole-market per session via daily+adj_factor")
    batch.add_argument("--strategy-a", action="store_true",
                       help="enable market gate and pullback-quality boost in scan output")
    batch.add_argument("--min-startup-breadth", type=float, default=0.01)
    batch.add_argument("--min-market-ret5", type=float, default=0.017)
    batch.add_argument("--max-boost-return-20d", type=float, default=-0.006)
    batch.add_argument("--min-boost-ocf-profit", type=float, default=-13.0)
    batch.add_argument("--max-boost-ocf-profit", type=float, default=98.0)
    batch.add_argument("--boost-points", type=float, default=5.0)
    batch.add_argument("--output", type=Path, default=Path("outputs/batch_scan.csv"))
    batch.add_argument("--failures-output", type=Path, default=Path("outputs/batch_failures.csv"))
    batch.add_argument("--coverage-output", type=Path, default=Path("outputs/batch_coverage.csv"))
    batch.add_argument("--min-score", type=float, default=60.0)

    daily = sub.add_parser(
        "daily-scan",
        help="fast end-of-day scan for one as-of date (intended for live tracking)",
    )
    daily.add_argument("--as-of", default=None,
                       help="scan date (default: last session in cache, or today if prefetching)")
    daily.add_argument("--start", default="2025-01-01")
    daily.add_argument("--end", default=None)
    daily.add_argument("--data-dir", type=Path, default=None)
    daily.add_argument("--universe", type=Path, default=Path("data/universe.csv"))
    daily.add_argument("--api-url", default=None)
    daily.add_argument("--prefetch", action="store_true",
                       help="fetch today's bars/financials from the provider before scanning")
    daily.add_argument("--offline-scan", action="store_true",
                       help="skip provider fetch; fail if data is missing")
    daily.add_argument("--request-timeout", type=float, default=30.0)
    daily.add_argument("--strategy-a", action="store_true",
                       help="default A-version: OR gate, boost, 6%% profit target / 3+5 days")
    daily.add_argument("--recommended", action="store_true",
                       help="final config: breadth gate + regime gate (CSI300>MA20) + "
                            "V0 startup (volume <=2.5x) + quality>=55, "
                            "5%% profit target / 10d holding")
    daily.add_argument("--regime-ma", type=int, default=None,
                       help="regime gate: CSI300 close > MA-N on signal date "
                            "(default 20 with --recommended; 0 or --no-regime disables)")
    daily.add_argument("--no-regime", action="store_true", help="disable the regime gate")
    daily.add_argument("--min-startup-breadth", type=float, default=0.01)
    daily.add_argument("--min-market-ret5", type=float, default=0.017)
    daily.add_argument("--gate-mode", choices=("OR", "AND"), default="OR")
    daily.add_argument("--use-boost", action="store_true")
    daily.add_argument("--max-boost-return-20d", type=float, default=-0.006)
    daily.add_argument("--min-boost-ocf-profit", type=float, default=-13.0)
    daily.add_argument("--max-boost-ocf-profit", type=float, default=98.0)
    daily.add_argument("--boost-points", type=float, default=5.0)
    daily.add_argument("--min-score", type=float, default=60.0)
    daily.add_argument("--min-avg-amount", type=float, default=None,
                       help="liquidity floor: 20-day mean turnover in 万元 (e.g. 5000)")
    daily.add_argument("--min-quality", type=float, default=None,
                       help="quality-score floor (0-100, e.g. 55)")
    daily.add_argument("--max-volume-ratio", type=float, default=None,
                       help="gentle-volume cap on startup signal (e.g. 2.5)")
    daily.add_argument("--min-market-cap", type=float, default=None,
                       help="min total market cap in 亿元 (uses daily_basic cache or provider)")
    daily.add_argument("--max-market-cap", type=float, default=None,
                       help="max total market cap in 亿元 (uses daily_basic cache or provider)")
    daily.add_argument("--output", type=Path, default=None,
                       help="selected candidates CSV (default: outputs/daily_<date>_selected.csv)")
    daily.add_argument("--track-log", type=Path, default=Path("outputs/daily_tracking.csv"),
                       help="append one 'gate summary' row per run for long-term tracking")
    daily.add_argument("--workers", type=int, default=12,
                       help="parallel scan workers (default: 12)")
    daily.add_argument("--top", type=int, default=None,
                       help="only print/export top N candidates by total_score")

    event = sub.add_parser("evaluate-event", help="evaluate selected signals using future cached bars")
    event.add_argument("--signals", type=Path, required=True)
    event.add_argument("--data-dir", type=Path, default=None)
    event.add_argument("--output", type=Path, default=Path("outputs/event_returns.csv"))
    event.add_argument("--holdings", default="5,10,20")
    event.add_argument("--exit-modes", default="fixed_open",
                       help="comma-separated fixed_open,window_end_close")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    data_dir = args.data_dir or settings.data_dir

    if args.command == "universe":
        provider = TushareProvider(settings.require_token(), api_url=args.api_url)
        stocks = filter_universe(provider.stock_basic(), as_of=args.as_of)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stocks.to_csv(args.output, index=False)
        print(f"saved {len(stocks)} symbols to {args.output}")
        return
    if args.command == "financials":
        instrument = Instrument(args.symbol)
        provider = TushareProvider(settings.require_token(), api_url=args.api_url)
        frame = provider.financial_snapshot(instrument.ts_code, start_date=args.start, end_date=args.end)
        path = FinancialStore(data_dir).save(frame, source=provider.name)
        print(f"saved {len(frame)} financial rows to {path}")
        return
    if args.command == "walk-forward":
        provider = TushareProvider(
            settings.require_token(), api_url=args.api_url,
            request_timeout=args.request_timeout,
        )
        universe = pd.read_csv(args.universe)
        filtered = filter_universe(universe, as_of=max(args.as_of))
        universe_by_as_of = None
        if args.universe_by_as_of_dir is not None:
            universe_by_as_of = {
                as_of: pd.read_csv(args.universe_by_as_of_dir / f"{as_of}.csv")
                for as_of in args.as_of
                if (args.universe_by_as_of_dir / f"{as_of}.csv").exists()
            }
        if len(args.as_of) > 1 and universe_by_as_of is None:
            print("warning: using one current universe snapshot; historical results are survivorship-limited")
        calendar_end = max(args.as_of) if args.end is None else max(args.end, max(args.as_of))
        try:
            trade_dates = provider.get_trade_dates(args.start, calendar_end)
        except RuntimeError as exc:
            if "trade_cal" not in str(exc):
                raise
            print("warning: trade calendar unavailable; using returned bar dates only")
            trade_dates = None
        if args.prefetch or args.prefetch_batch:
            prefetch_as_of = max(args.as_of)
            prefetch_start = trade_dates[0] if trade_dates is not None else args.start
            prefetch_end = args.end or prefetch_as_of
            prefetch_config = BatchScanConfig(
                as_of=prefetch_as_of, start=prefetch_start,
                end=prefetch_end, min_score=args.min_score,
            )
            if args.prefetch_batch:
                if trade_dates is None:
                    print("warning: --prefetch-batch needs the trade calendar; falling back to serial prefetch")
                else:
                    from .data.batch_fetch import batch_prefetch_bars_by_symbols
                    batch_stats = batch_prefetch_bars_by_symbols(
                        filtered, provider, BarStore(data_dir),
                        start=prefetch_start, end=prefetch_end,
                        trade_dates=trade_dates, max_symbols=args.prefetch_limit,
                        workers=args.prefetch_workers or int(os.getenv("QFREE_PREFETCH_WORKERS", "12")),
                        chunk_size=14,
                    )
                    print(f"batch prefetch complete: {batch_stats.as_dict()}")
                    for failure in batch_stats.failures:
                        print(f"batch prefetch failure: {failure}")
            if args.prefetch_batch:
                financial_stats = _prefetch_financial_batch(
                    filtered, provider, data_dir, prefetch_as_of,
                    max_symbols=args.prefetch_limit,
                    chunk_size=50,
                    workers=args.prefetch_workers or 12,
                )
                print(f"financial batch prefetch complete: {financial_stats}")
            if args.prefetch_batch and args.offline_scan:
                prefetch_config = BatchScanConfig(
                    as_of=prefetch_as_of, start=prefetch_start,
                    end=prefetch_end, min_score=args.min_score,
                    allow_provider_fetch=False,
                )
            # Serial/threaded fallback fills only caches missed by the batch
            # paths; successful batch financial rows are cache hits here.
            stats = _prefetch_missing_data(
                filtered, provider, data_dir, prefetch_config,
                max_symbols=args.prefetch_limit,
                workers=args.prefetch_workers,
            )
            print(f"prefetch complete: {stats}")
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            filtered = filtered.head(args.limit)
            if universe_by_as_of is not None:
                universe_by_as_of = {
                    key: value.head(args.limit) for key, value in universe_by_as_of.items()
                }
        base_config = BatchScanConfig(
            as_of=args.as_of[0], start=args.start,
            end=args.end or args.as_of[0], min_score=args.min_score,
            allow_provider_fetch=not args.offline_scan,
            use_market_gate=args.strategy_a,
            min_startup_breadth=args.min_startup_breadth,
            min_market_return_5d=args.min_market_ret5,
            use_pullback_quality_boost=args.strategy_a,
            max_boost_return_20d=args.max_boost_return_20d,
            min_boost_ocf_to_profit=args.min_boost_ocf_profit,
            max_boost_ocf_to_profit=args.max_boost_ocf_profit,
            boost_points=args.boost_points,
        )
        report = (
            fast_walk_forward_scan(
                filtered, base_config, args.as_of,
                trade_dates=trade_dates, workers=args.scan_workers, data_dir=data_dir,
            )
            if args.fast_scan else walk_forward_scan(
                filtered, provider, BarStore(data_dir), FinancialStore(data_dir), args.as_of,
                start=args.start, end=args.end, config=base_config,
                trade_dates=trade_dates, universe_by_as_of=universe_by_as_of,
                workers=args.scan_workers,
                data_dir=data_dir, token=settings.tushare_token, api_url=args.api_url,
                request_timeout=args.request_timeout,
            )
        )
        bars_by_symbol = {}
        event_store = BarStore(data_dir)
        selected_symbols = report.candidates.loc[
            _selected_values(report.candidates), "ts_code"
        ].dropna().unique() if not report.candidates.empty else []
        for symbol in selected_symbols:
            try:
                bars_by_symbol[symbol] = event_store.load(
                    Instrument(symbol), "1900-01-01", "2100-01-01"
                )
            except FileNotFoundError:
                continue
        holding_spec = args.holdings or ("3,5" if args.strategy_a else "5,10,20")
        exit_modes = ("profit_target_close",) if args.strategy_a else tuple(args.exit_modes.split(","))
        profit_target = args.profit_target if args.strategy_a else None
        event_report = evaluate_walk_forward_events(
            report, bars_by_symbol, tuple(int(value) for value in holding_spec.split(",")),
            exit_modes=exit_modes, trade_dates=trade_dates, profit_target=profit_target,
        )
        for output in (args.output, args.failures_output, args.coverage_output,
                       args.events_output, args.event_summary_output):
            output.parent.mkdir(parents=True, exist_ok=True)
        report.candidates.to_csv(args.output, index=False)
        report.failures.to_csv(args.failures_output, index=False)
        report.coverage.to_csv(args.coverage_output, index=False)
        event_report.events.to_csv(args.events_output, index=False)
        event_report.summary.to_csv(args.event_summary_output, index=False)
        print(f"saved {len(report.candidates)} candidates to {args.output}")
        print(f"saved {len(report.failures)} failures to {args.failures_output}")
        print(f"saved {len(event_report.events)} event rows to {args.events_output}")
        print(report.coverage.to_string(index=False))
        return
    if args.command == "daily-scan":
        import time as _time
        t_start = _time.time()
        provider = (TushareProvider(settings.require_token(), api_url=args.api_url,
                                    request_timeout=args.request_timeout)
                    if args.prefetch else None)
        universe = pd.read_csv(args.universe)
        # Resolve as-of: default to the most recent session in cache, or today if prefetching.
        if args.as_of is None:
            if args.prefetch:
                args.as_of = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
            else:
                try:
                    probe_store = BarStore(data_dir)
                    probe = probe_store.load(
                        Instrument(universe.iloc[0]["ts_code"]), args.start, "2100-01-01", adjust="qfq"
                    )
                    args.as_of = probe["date"].max().strftime("%Y-%m-%d")
                except Exception:
                    args.as_of = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        if args.end is None:
            args.end = args.as_of

        # Build config from flags, with shorthand presets.
        if args.recommended:
            # Final config (validated 2026-10, cost-model update): V0 startup
            # signal with gentle-volume cap, breadth gate at 0.9%, quality floor,
            # 5% target / 10d holding, top-10 equal weight per trigger date.
            # Basis: 723 events, 10d = 69.6% win / +2.68% mean at 10bp cost;
            # realistic annualized 6-12% with 21-27% capital deployment.
            gate_mode = "AND"
            args.min_startup_breadth = 0.009
            args.min_market_ret5 = -1.0  # breadth-only gate: ret5 condition always true
            use_boost = False
            profit_target = 0.05
            holdings = (10,)
            if args.max_volume_ratio is None:
                args.max_volume_ratio = 2.5
            if args.min_quality is None:
                args.min_quality = 55
            if args.top is None:
                args.top = 10  # validated sweet spot: 84.4% trigger-date win rate
            if args.regime_ma is None and not args.no_regime:
                args.regime_ma = 20  # long-period validated regime gate (2023-2026)
        elif args.strategy_a:
            gate_mode = "OR"
            use_boost = True
            profit_target = 0.06
            holdings = (3, 5)
        else:
            gate_mode = args.gate_mode
            use_boost = args.use_boost
            profit_target = None
            holdings = (5,)

        # Regime gate: CSI300 close > MA-N on the signal date. Reads the cached
        # index bars; updates the cache from the provider when one is available.
        regime_flags = None
        regime_close = regime_ma_val = None
        if args.regime_ma and not args.no_regime:
            from .data.index_cache import DEFAULT_INDEX, load_index, regime_flags as _mk_flags, update_index
            if provider is not None:
                try:
                    update_index(provider, data_dir, DEFAULT_INDEX)
                except Exception as exc:
                    print(f"warning: index cache update failed ({exc})")
            idx = load_index(data_dir, DEFAULT_INDEX)
            as_of_ts = pd.Timestamp(args.as_of).normalize()
            if idx.empty:
                print("warning: index cache missing (run scripts/sync_bars.py); "
                      "regime gate fails closed")
                regime_flags = pd.Series(dtype=bool)
            else:
                if idx["date"].max() < as_of_ts:
                    print(f"warning: index cache ends {idx['date'].max().date()} "
                          f"< as-of {args.as_of}; regime flag for as-of is unknown, fails closed")
                regime_flags = _mk_flags(idx, args.regime_ma)
                row = idx[idx["date"] <= as_of_ts]
                if len(row):
                    regime_close = float(row["close"].iloc[-1])
                    regime_ma_val = float(pd.Series(
                        pd.to_numeric(idx["close"], errors="coerce"))
                        .rolling(args.regime_ma).mean().iloc[len(row) - 1])

        base_config = BatchScanConfig(
            as_of=args.as_of, start=args.start, end=args.end, min_score=args.min_score,
            allow_provider_fetch=not args.offline_scan,
            use_market_gate=True,
            min_startup_breadth=args.min_startup_breadth,
            min_market_return_5d=args.min_market_ret5,
            use_pullback_quality_boost=use_boost,
            max_boost_return_20d=args.max_boost_return_20d,
            min_boost_ocf_to_profit=args.min_boost_ocf_profit,
            max_boost_ocf_to_profit=args.max_boost_ocf_profit,
            boost_points=args.boost_points,
            # CLI takes 万元; bars store amount in 千元 (1 万元 = 10 千元).
            min_avg_amount=(args.min_avg_amount * 10.0
                            if args.min_avg_amount is not None else None),
            min_quality_score=args.min_quality,
            max_volume_ratio=args.max_volume_ratio,
            regime_flags=regime_flags,
        )
        # AND gate: raise both thresholds so both conditions are required.
        if gate_mode == "AND":
            # Implement AND by making each threshold individually high enough that
            # either one alone is never sufficient? No — cleaner: handle post-scan.
            pass

        if args.prefetch and provider is not None:
            from .data.batch_fetch import batch_prefetch_bars_by_symbols
            t_prefetch = _time.time()
            filtered = filter_universe(universe, as_of=args.as_of,
                                      min_list_days=base_config.min_list_days)
            trade_dates_pf = pd.DatetimeIndex(pd.to_datetime(
                provider.get_trade_dates(args.start, args.end)))
            stats = batch_prefetch_bars_by_symbols(
                filtered, provider, BarStore(data_dir), start=args.start, end=args.end,
                trade_dates=trade_dates_pf,
            )
            print(f"prefetch: {stats.symbols_total - stats.symbols_complete} symbols fetched "
                  f"({stats.symbols_complete} already complete) in "
                  f"{_time.time() - t_prefetch:.1f}s")

        # Trade calendar: prefer provider, otherwise infer from the cached bars
        # of the first symbol (sufficient for coverage checks even if a few dates
        # are missing for that particular symbol).
        trade_dates = None
        if provider is not None:
            try:
                trade_dates = pd.DatetimeIndex(pd.to_datetime(
                    provider.get_trade_dates(args.start, args.end)))
            except RuntimeError as exc:
                if "trade_cal" not in str(exc):
                    raise
                trade_dates = None
        if trade_dates is None and not args.offline_scan:
            trade_dates = None
        elif trade_dates is None:
            # Infer from first universe symbol's cached bars.
            try:
                probe_store = BarStore(data_dir)
                probe = probe_store.load(
                    Instrument(universe.iloc[0]["ts_code"]), args.start, args.end,
                    adjust="qfq",
                )
                trade_dates = pd.DatetimeIndex(
                    pd.to_datetime(pd.Series(probe["date"])).dt.normalize()
                )
            except Exception:
                trade_dates = None
        report = fast_walk_forward_scan(
            universe, base_config, [args.as_of],
            trade_dates=trade_dates, workers=args.workers, data_dir=str(data_dir),
        )

        # Apply AND gate if requested: _finalize_date_rows always uses OR
        # semantics. For AND we rebuild the candidates and coverage.
        if gate_mode == "AND" and not report.candidates.empty:
            c = report.candidates.copy()
            breadth = float(c["startup_signal_breadth"].iloc[0]) if not c.empty else 0.0
            ret5 = float(pd.to_numeric(c["market_return_5d"], errors="coerce").iloc[0]) if not c.empty else 0.0
            gate_pass = breadth >= args.min_startup_breadth and ret5 >= args.min_market_ret5
            c["market_gate_pass"] = gate_pass
            selected_mask = (
                c["startup_signal"].astype(bool)
                & c["market_gate_pass"]
                & c.get("regime_gate_pass", pd.Series(True, index=c.index)).astype(bool)
                & c["total_score"].ge(args.min_score)
                & c["financial_field_count"].ge(base_config.min_financial_fields)
            )
            if args.min_quality is not None:
                selected_mask &= pd.to_numeric(
                    c["quality_score"], errors="coerce").ge(args.min_quality)
            if args.min_avg_amount is not None:
                selected_mask &= pd.to_numeric(
                    c.get("avg_amount_20d"), errors="coerce").ge(args.min_avg_amount * 10.0)
            c["selected"] = selected_mask
            from dataclasses import replace
            coverage = report.coverage.copy()
            coverage.loc[0, "market_gate_pass"] = gate_pass
            coverage.loc[0, "selected"] = int(c["selected"].sum())
            report = replace(report, candidates=c, coverage=coverage)

        # Market-cap band filter (experiment parameter, not a hard value call).
        # Needs daily_basic; provider fetch otherwise outputs/daily_basic_<date>.csv.
        if (args.min_market_cap is not None or args.max_market_cap is not None) \
                and not report.candidates.empty:
            from dataclasses import replace as _replace
            basic = None
            cache_path = Path(f"outputs/daily_basic_{args.as_of}.csv")
            if provider is not None:
                try:
                    basic = provider._pro.daily_basic(
                        trade_date=str(args.as_of).replace("-", ""))
                    basic.to_csv(cache_path, index=False)
                except Exception as exc:
                    print(f"warning: daily_basic fetch failed ({exc}); trying cache")
            if basic is None and cache_path.exists():
                basic = pd.read_csv(cache_path)
            if basic is not None and not basic.empty:
                c = report.candidates.copy()
                basic = basic[["ts_code", "total_mv"]].drop_duplicates("ts_code")
                c = c.merge(basic, on="ts_code", how="left")
                mv = pd.to_numeric(c["total_mv"], errors="coerce")  # 万元
                cap_ok = pd.Series(True, index=c.index)
                if args.min_market_cap is not None:
                    cap_ok &= mv.ge(args.min_market_cap * 1e4)
                if args.max_market_cap is not None:
                    cap_ok &= mv.le(args.max_market_cap * 1e4)
                cap_ok = cap_ok.where(mv.notna(), False)
                dropped = int((c["selected"].astype(bool) & ~cap_ok).sum())
                c.loc[~cap_ok, "selected"] = False
                coverage = report.coverage.copy()
                coverage.loc[:, "selected"] = int(c["selected"].sum())
                report = _replace(report, candidates=c, coverage=coverage)
                print(f"market-cap filter [{args.min_market_cap},{args.max_market_cap}]亿: "
                      f"dropped {dropped} selected candidates")
            else:
                print("warning: market-cap filter requested but no daily_basic data "
                      "available; skipped")

        selected = report.candidates[report.candidates["selected"].astype(bool)].copy() \
            if not report.candidates.empty else pd.DataFrame()
        if args.top is not None and not selected.empty:
            selected = selected.head(args.top)

        # Outputs
        if args.output is None:
            args.output = Path(f"outputs/daily_{args.as_of}_selected.csv")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(args.output, index=False)
        print(f"\n=== daily scan {args.as_of} ===")
        print(f"config: gate={gate_mode}(breadth>={args.min_startup_breadth:.1%}, "
              f"ret5>={args.min_market_ret5:.1%}), boost={'on' if use_boost else 'off'}, "
              f"min_score={args.min_score}")
        cov = report.coverage.iloc[0]
        breadth = cov.get("startup_signal_breadth")
        ret5 = cov.get("market_return_5d")
        breadth_s = f"{breadth:.2%}" if pd.notna(breadth) else "N/A"
        ret5_s = f"{ret5:.2%}" if pd.notna(ret5) else "N/A"
        regime_s = ""
        if args.regime_ma and not args.no_regime:
            rp = bool(cov.get("regime_gate_pass", False))
            if regime_close is not None and regime_ma_val is not None:
                regime_s = (f", regime: CSI300 {regime_close:.2f} vs MA{args.regime_ma} "
                            f"{regime_ma_val:.2f} ({'pass' if rp else 'BLOCK'})")
            else:
                regime_s = f", regime gate: {'pass' if rp else 'BLOCK (no index data)'}"
        print(f"valid snapshots: {int(cov['valid_snapshots'])}, "
              f"startup signals: {int(cov['technical_signals'])} (breadth={breadth_s}), "
              f"market ret5: {ret5_s}, "
              f"gate pass: {bool(cov['market_gate_pass'])}{regime_s}")
        print(f"selected: {len(selected)} stocks")
        if not selected.empty:
            cols = ["rank", "ts_code", "name", "industry", "total_score",
                    "technical_score", "quality_score", "return_20d", "price_position_60d"]
            print_cols = [c for c in cols if c in selected.columns]
            print(selected[print_cols].to_string(index=False))

        # Tracking log
        args.track_log.parent.mkdir(parents=True, exist_ok=True)
        track_row = {
            "as_of": args.as_of,
            "valid_snapshots": int(cov["valid_snapshots"]),
            "startup_breadth": cov["startup_signal_breadth"],
            "market_return_5d": cov["market_return_5d"],
            "gate_mode": gate_mode,
            "gate_pass": bool(cov["market_gate_pass"]),
            "regime_pass": bool(cov.get("regime_gate_pass", True)),
            "regime_ma": args.regime_ma if (args.regime_ma and not args.no_regime) else None,
            "selected_count": len(selected),
            "config": f"boost={'on' if use_boost else 'off'},min_score={args.min_score}",
            "elapsed_seconds": round(_time.time() - t_start, 1),
        }
        track_df = pd.DataFrame([track_row])
        if args.track_log.exists():
            existing = pd.read_csv(args.track_log)
            if str(args.as_of) in existing["as_of"].astype(str).values:
                existing = existing[existing["as_of"].astype(str) != str(args.as_of)]
            track_df = pd.concat([existing, track_df], ignore_index=True)
        track_df.to_csv(args.track_log, index=False)
        print(f"\ntracking log updated: {args.track_log} ({len(track_df)} rows)")
        print(f"total time: {_time.time() - t_start:.1f}s")
        return
    if args.command == "batch-scan":
        provider = TushareProvider(settings.require_token(), api_url=args.api_url)
        universe = pd.read_csv(args.universe)
        filtered = filter_universe(universe, as_of=args.as_of)
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            filtered = filtered.head(args.limit)
        if args.prefetch or args.prefetch_batch:
            calendar_end = args.end or args.as_of
            try:
                trade_dates = provider.get_trade_dates(args.start, calendar_end)
            except RuntimeError as exc:
                if "trade_cal" not in str(exc):
                    raise
                print("warning: trade calendar unavailable; using returned bar dates only")
                trade_dates = None
            prefetch_start = trade_dates[0] if trade_dates is not None else args.start
            prefetch_config = BatchScanConfig(
                as_of=args.as_of, start=prefetch_start,
                end=args.end or args.as_of, min_score=args.min_score,
            )
            if args.prefetch_batch and trade_dates is not None:
                from .data.batch_fetch import batch_prefetch_bars
                batch_stats = batch_prefetch_bars(
                    filtered, provider, BarStore(data_dir),
                    start=prefetch_start, end=args.end or args.as_of,
                    trade_dates=trade_dates,
                )
                print(f"batch prefetch complete: {batch_stats.as_dict()}")
                for failure in batch_stats.failures:
                    print(f"batch prefetch failure: {failure}")
            stats = _prefetch_missing_data(filtered, provider, data_dir, prefetch_config)
            print(f"prefetch complete: {stats}")
        config = BatchScanConfig(
            as_of=args.as_of,
            start=args.start,
            end=args.end or args.as_of,
            min_score=args.min_score,
            use_market_gate=args.strategy_a,
            min_startup_breadth=args.min_startup_breadth,
            min_market_return_5d=args.min_market_ret5,
            use_pullback_quality_boost=args.strategy_a,
            max_boost_return_20d=args.max_boost_return_20d,
            min_boost_ocf_to_profit=args.min_boost_ocf_profit,
            max_boost_ocf_to_profit=args.max_boost_ocf_profit,
            boost_points=args.boost_points,
        )
        report = batch_scan_detailed(
            filtered, provider, BarStore(data_dir), FinancialStore(data_dir), config
        )
        for output in (args.output, args.failures_output, args.coverage_output):
            output.parent.mkdir(parents=True, exist_ok=True)
        report.candidates.to_csv(args.output, index=False)
        report.failures.to_csv(args.failures_output, index=False)
        report.coverage.to_csv(args.coverage_output, index=False)
        coverage = report.coverage.iloc[0]
        print(f"saved {len(report.candidates)} candidates to {args.output}")
        print(f"saved {len(report.failures)} failures to {args.failures_output}")
        print(coverage.to_string())
        return
    if args.command == "evaluate-event":
        signals = pd.read_csv(args.signals)
        bars_by_symbol = {}
        store = BarStore(data_dir)
        for symbol in signals.loc[_selected_values(signals), "ts_code"].unique():
            instrument = Instrument(symbol)
            try:
                bars_by_symbol[symbol] = store.load(instrument, "1900-01-01", "2100-01-01")
            except FileNotFoundError:
                continue
        holdings = tuple(int(value) for value in args.holdings.split(","))
        exit_modes = tuple(args.exit_modes.split(","))
        events = evaluate_events(signals, bars_by_symbol, holdings, exit_modes=exit_modes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(args.output, index=False)
        print(f"saved {len(events)} event rows to {args.output}")
        summary = summarize_events(events)
        if not summary.empty:
            print(summary.to_string(index=False))
        return
    if args.command == "fetch":
        instrument = Instrument(args.symbol)
        provider = TushareProvider(settings.require_token(), api_url=args.api_url)
        end = args.end or datetime.now(UTC).date().isoformat()
        bars = provider.get_daily_bars(instrument, args.start, end, args.adjust)
        path = BarStore(data_dir).save(instrument, bars, adjust=args.adjust, source=provider.name)
        print(f"saved {len(bars)} rows to {path}")
        return

    instrument = Instrument(args.symbol)
    bars = BarStore(data_dir).load(instrument, "1900-01-01", "2100-01-01")
    result = screen(bars, min_score=args.min_score)
    volume_columns = [column for column in result if column.startswith("volume_ratio_")]
    print(result.loc[result["selected"], ["date", "close", "total_score", *volume_columns]].to_string(index=False))


if __name__ == "__main__":
    main()
