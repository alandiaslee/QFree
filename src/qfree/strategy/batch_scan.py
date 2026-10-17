"""Batch screening orchestration for a small trial or the full universe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from ..data.calendar import expected_sessions
from ..data.financial_store import FinancialStore
from ..data.models import Instrument
from ..data.store import BarStore
from ..data.tushare_client import TushareProvider
from ..features.quality import quality_score
from ..features.technical import TechnicalConfig, calculate_technical, technical_score
from ..universe import filter_universe

QUALITY_FIELDS = ("roe", "netprofit_yoy", "ocf_to_profit", "debt_to_assets", "grossprofit_margin")


class IncompleteBarCoverageError(FileNotFoundError):
    """Raised when neither cache nor provider covers the requested sessions."""


class OfflineCacheMissError(IncompleteBarCoverageError):
    """Raised when an offline scan cannot find complete local coverage."""


@dataclass(frozen=True)
class BatchScanConfig:
    as_of: date | str
    start: date | str
    end: date | str
    min_list_days: int = 250
    min_score: float = 60.0
    min_financial_fields: int = 3
    adjust: str = "qfq"
    allow_provider_fetch: bool = True
    use_market_gate: bool = False
    min_startup_breadth: float = 0.01
    min_market_return_5d: float = 0.017
    use_pullback_quality_boost: bool = False
    max_boost_return_20d: float = -0.006
    min_boost_ocf_to_profit: float = -13.0
    max_boost_ocf_to_profit: float = 98.0
    boost_points: float = 5.0
    # Liquidity floor on 20-session mean turnover, in the bars' amount unit
    # (thousand CNY). None disables the filter.
    min_avg_amount: float | None = None
    # Quality-score floor applied after quality_score is computed per date.
    min_quality_score: float | None = None
    # Gentle-volume cap on the startup signal (volume_ratio <= cap; None disables).
    max_volume_ratio: float | None = None
    # Regime gate: boolean Series (DatetimeIndex -> bool) requiring a True flag
    # on the snapshot date (e.g. CSI300 close > MA20); None disables the gate.
    regime_flags: pd.Series | None = None

    def __post_init__(self) -> None:
        if self.min_startup_breadth < 0:
            raise ValueError("min_startup_breadth must be non-negative")
        if self.min_market_return_5d < -1:
            raise ValueError("min_market_return_5d must be greater than -1")
        if self.max_boost_return_20d < -1:
            raise ValueError("max_boost_return_20d must be greater than -1")
        if self.min_boost_ocf_to_profit > self.max_boost_ocf_to_profit:
            raise ValueError("boost OCF/profit range is invalid")
        if self.boost_points < 0:
            raise ValueError("boost_points must be non-negative")
        if self.min_avg_amount is not None and self.min_avg_amount < 0:
            raise ValueError("min_avg_amount must be non-negative")
        if self.max_volume_ratio is not None and self.max_volume_ratio <= 0:
            raise ValueError("max_volume_ratio must be positive")


@dataclass(frozen=True)
class BatchScanResult:
    """Detailed result of one batch, retaining operational failures."""

    candidates: pd.DataFrame
    failures: pd.DataFrame
    coverage: pd.DataFrame


def _latest_technical(
    bars: pd.DataFrame,
    as_of: date | str,
    config: TechnicalConfig | None,
) -> pd.DataFrame:
    features = calculate_technical(bars, config)
    if features.empty:
        return pd.DataFrame()
    features["technical_score"] = technical_score(features)
    eligible = features[features["date"] <= pd.Timestamp(as_of).normalize()]
    if eligible.empty:
        return pd.DataFrame()
    return eligible.tail(1)


def _failure(stock: object, config: BatchScanConfig, stage: str, status: str,
             error: Exception | None = None, *, cache_hit: bool = False) -> dict[str, object]:
    return {
        "as_of": pd.Timestamp(config.as_of).normalize(),
        "ts_code": str(getattr(stock, "ts_code", "")),
        "name": getattr(stock, "name", None),
        "stage": stage,
        "status": status,
        "error_type": type(error).__name__ if error else None,
        "message": str(error) if error else None,
        "cache_hit": cache_hit,
    }


def _bar_coverage(
    dates: pd.Series,
    config: BatchScanConfig,
    trade_dates: pd.DatetimeIndex | None,
) -> bool:
    """Check whether cached bar dates cover the requested window (see _load_bars)."""
    start = pd.Timestamp(config.start).normalize()
    requested_end = pd.Timestamp(config.end).normalize()
    as_of = pd.Timestamp(config.as_of).normalize()
    required_end = min(requested_end, as_of)

    normalized = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if normalized.empty:
        return False
    eligible = normalized[normalized <= as_of]
    if eligible.empty:
        return False
    # Coverage is measured in exchange sessions, not calendar endpoints.
    if trade_dates is None:
        if normalized.iloc[0] > start:
            return False
        expected = pd.DatetimeIndex([])
    else:
        expected = expected_sessions(trade_dates, start, required_end)
        if expected.empty:
            return False
        if normalized.iloc[0] > expected[0]:
            return False
    # A weekend/holiday as-of is valid when the cache reaches its latest
    # preceding session; trade_cal handles longer holidays exactly.
    endpoint_reached = eligible.iloc[-1] >= required_end
    if not endpoint_reached and trade_dates is not None and not expected.empty:
        endpoint_reached = eligible.iloc[-1] >= expected[-1]
    if not endpoint_reached and trade_dates is None and required_end.weekday() >= 5:
        endpoint_reached = bool(
            eligible.iloc[-1] >= required_end - pd.Timedelta(days=3)
        )
    if not endpoint_reached:
        return False
    if trade_dates is None:
        return True
    available = pd.DatetimeIndex(normalized)
    return bool(set(expected).issubset(set(available)))


def _load_bars(
    instrument: Instrument,
    provider: TushareProvider | None,
    store: BarStore,
    config: BatchScanConfig,
    *,
    trade_dates: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Load bars and reject stale or incomplete coverage around the signal date."""
    cache_hit = True
    try:
        bars = store.load(instrument, config.start, config.end, adjust=config.adjust)
    except FileNotFoundError:
        bars = pd.DataFrame()
        cache_hit = False

    bar_dates = bars["date"] if "date" in bars.columns else pd.Series(dtype="datetime64[ns]")
    if not _bar_coverage(bar_dates, config, trade_dates):
        cache_hit = False
        if not config.allow_provider_fetch or provider is None:
            raise OfflineCacheMissError(
                f"cache miss or incomplete bar coverage for {instrument.ts_code}"
            )
        fresh = provider.get_daily_bars(instrument, config.start, config.end, config.adjust)
        if fresh.empty:
            raise IncompleteBarCoverageError(f"provider returned no bars for {instrument.ts_code}")
        store.save(
            instrument, fresh, adjust=config.adjust, source=provider.name,
            requested_start=config.start, requested_end=config.end,
            extra_metadata={"provider": provider.name},
        )
        bars = store.load(instrument, config.start, config.end, adjust=config.adjust)
        fresh_dates = bars["date"] if "date" in bars.columns else pd.Series(dtype="datetime64[ns]")
        if not _bar_coverage(fresh_dates, config, trade_dates):
            required_end = min(pd.Timestamp(config.end), pd.Timestamp(config.as_of)).normalize()
            raise IncompleteBarCoverageError(
                f"bar coverage incomplete for {instrument.ts_code}: "
                f"need {pd.Timestamp(config.start).date()}..{required_end.date()}"
            )
    return bars, cache_hit


def _finalize_date_rows(rows: list[pd.DataFrame], config: BatchScanConfig) -> pd.DataFrame:
    """Score, gate and rank one date's snapshot rows (shared by scan paths)."""
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    quality = quality_score(result, group_column="industry")
    result["quality_score"] = quality["quality_score"]
    result["quality_missing_count"] = quality["quality_missing_count"]
    result["quality_valid_weight"] = quality["quality_valid_weight"]
    result["financial_field_count"] = len(QUALITY_FIELDS) - result["quality_missing_count"]
    result["financial_missing_fields"] = result.apply(
        lambda row: ",".join(field for field in QUALITY_FIELDS if pd.isna(row[field])), axis=1
    )
    result["data_quality_status"] = result["financial_field_count"].map(
        lambda count: "complete" if count == len(QUALITY_FIELDS)
        else ("insufficient_financials" if count < config.min_financial_fields else "partial_financials")
    )
    result["base_total_score"] = 0.55 * result["quality_score"] + 0.45 * result["technical_score"]
    result["pullback_quality_boost"] = False
    boost_points = 0.0
    if config.use_pullback_quality_boost:
        boost_mask = (
            result["financial_field_count"].eq(len(QUALITY_FIELDS))
            & pd.to_numeric(result["return_20d"], errors="coerce").lt(config.max_boost_return_20d)
            & pd.to_numeric(result["ocf_to_profit"], errors="coerce")
                .between(config.min_boost_ocf_to_profit, config.max_boost_ocf_to_profit)
        )
        result["pullback_quality_boost"] = boost_mask
        boost_points = float(config.boost_points)
    result["boost_points"] = boost_points
    result["total_score"] = result["base_total_score"] + (
        result["pullback_quality_boost"].astype(float) * boost_points
    )
    startup_breadth = float(result["startup_signal"].astype(bool).mean())
    market_return_5d = float(pd.to_numeric(result["return_5d"], errors="coerce").mean())
    market_gate_pass = bool(
        startup_breadth >= config.min_startup_breadth
        or market_return_5d >= config.min_market_return_5d
    )
    result["startup_signal_breadth"] = startup_breadth
    result["market_return_5d"] = market_return_5d
    result["market_gate_enabled"] = config.use_market_gate
    result["market_gate_pass"] = market_gate_pass if config.use_market_gate else True
    regime_enabled = config.regime_flags is not None
    regime_pass = True
    if regime_enabled:
        flag = config.regime_flags.get(pd.Timestamp(config.as_of).normalize())
        regime_pass = bool(flag) if flag is not None else False
    result["regime_gate_enabled"] = regime_enabled
    result["regime_gate_pass"] = regime_pass
    selected = (
        result["startup_signal"]
        & result["market_gate_pass"]
        & result["regime_gate_pass"]
        & result["total_score"].ge(config.min_score)
        & result["financial_field_count"].ge(config.min_financial_fields)
    )
    if config.min_avg_amount is not None:
        avg_amount = pd.to_numeric(result.get("avg_amount_20d"), errors="coerce")
        selected &= avg_amount.ge(config.min_avg_amount)
    if config.min_quality_score is not None:
        selected &= quality["quality_score"].ge(config.min_quality_score)
    result["selected"] = selected
    result = result.sort_values("total_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


def _coverage_row(
    universe: pd.DataFrame,
    stocks: pd.DataFrame,
    config: BatchScanConfig,
    result: pd.DataFrame,
    failure_frame: pd.DataFrame,
) -> dict[str, object]:
    return {
        "as_of": pd.Timestamp(config.as_of).normalize(),
        "input_universe": len(universe),
        "filtered_universe": len(stocks),
        "valid_snapshots": len(result),
        "failures": len(failure_frame),
        "technical_signals": int(result["startup_signal"].sum()) if not result.empty else 0,
        "selected": int(result["selected"].sum()) if not result.empty else 0,
        "financial_complete": int((result["data_quality_status"] == "complete").sum())
        if not result.empty else 0,
        "financial_field_coverage": float(
            result["financial_field_count"].sum() / (len(result) * len(QUALITY_FIELDS))
        ) if not result.empty else 0.0,
        "quality_scoring_denominator": len(result),
        "quality_valid_weight_mean": float(result["quality_valid_weight"].mean())
        if not result.empty else 0.0,
        "market_gate_enabled": config.use_market_gate,
        "startup_signal_breadth": float(result["startup_signal_breadth"].iloc[0])
        if not result.empty else 0.0,
        "market_return_5d": float(result["market_return_5d"].iloc[0])
        if not result.empty else None,
        "market_gate_pass": bool(result["market_gate_pass"].iloc[0])
        if not result.empty else False,
        "regime_gate_enabled": bool(result["regime_gate_enabled"].iloc[0])
        if not result.empty else False,
        "regime_gate_pass": bool(result["regime_gate_pass"].iloc[0])
        if not result.empty else False,
        "score_pool": "all_valid_snapshots",
    }


def batch_scan_detailed(
    universe: pd.DataFrame,
    provider: TushareProvider | None,
    store: BarStore,
    financial_store: FinancialStore,
    config: BatchScanConfig,
    *,
    technical_config: TechnicalConfig | None = None,
    trade_dates: pd.DatetimeIndex | None = None,
) -> BatchScanResult:
    """Scan a universe while retaining failures and coverage diagnostics."""
    stocks = filter_universe(universe, as_of=config.as_of, min_list_days=config.min_list_days)
    rows: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    for stock in stocks.itertuples(index=False):
        instrument = Instrument(str(stock.ts_code), name=getattr(stock, "name", None))
        cache_hit = True
        try:
            bars, cache_hit = _load_bars(
                instrument, provider, store, config, trade_dates=trade_dates
            )
            latest = _latest_technical(bars, config.as_of, technical_config)
            if latest.empty:
                failures.append(_failure(stock, config, "features", "missing_target_date",
                                         cache_hit=cache_hit))
                continue
            latest = latest.copy()
            latest["as_of"] = pd.Timestamp(config.as_of).normalize()
            latest["signal_date"] = latest["date"].iloc[0]
            latest["bars_start_date"] = bars["date"].iloc[0]
            latest["bars_last_date"] = bars["date"].iloc[-1]
            latest["ts_code"] = instrument.ts_code
            latest["name"] = getattr(stock, "name", None)
            latest["industry"] = getattr(stock, "industry", None)
            latest["bar_adjust"] = config.adjust
            latest["cache_hit"] = cache_hit

            try:
                financial = financial_store.as_of(instrument.ts_code, config.as_of)
            except FileNotFoundError:
                financial = pd.DataFrame()
            if financial.empty and config.allow_provider_fetch and provider is not None:
                # Fetch the full available history unless the provider supports a
                # narrower range without dropping the latest report usable at as-of.
                raw_financials = provider.financial_snapshot(
                    instrument.ts_code, start_date=None, end_date=config.as_of
                )
                if raw_financials.empty:
                    financial = pd.DataFrame()
                else:
                    financial_store.save(raw_financials, source=provider.name)
                    financial = financial_store.as_of(instrument.ts_code, config.as_of)
            latest["financial_available_date"] = (
                financial["available_date"].iloc[0] if not financial.empty else pd.NaT
            )
            for column in QUALITY_FIELDS:
                latest[column] = financial[column].iloc[0] if column in financial and not financial.empty else pd.NA
            rows.append(latest)
        except OfflineCacheMissError as exc:
            failures.append(_failure(stock, config, "bars", "cache_miss_provider_unavailable", exc,
                                     cache_hit=cache_hit))
        except IncompleteBarCoverageError as exc:
            failures.append(_failure(stock, config, "bars", "incomplete_bar_coverage", exc,
                                     cache_hit=cache_hit))
        except FileNotFoundError as exc:
            failures.append(_failure(stock, config, "bars", "missing_bars", exc, cache_hit=cache_hit))
        except Exception as exc:  # noqa: BLE001 - keep one bad symbol from aborting the date
            message = str(exc).lower()
            if isinstance(exc, (RuntimeError, ValueError, TypeError)):
                stage = "financials" if "financial" in message else "scan"
                status = (
                    "provider_auth_error"
                    if "token" in message or "permission" in message
                    or "request failed after" in message
                    else (
                        "cache_miss_provider_unavailable"
                        if "cache miss" in message
                        else "provider_or_data_error"
                    )
                )
            else:
                stage = "scan"
                status = "unexpected_error"
            failures.append(_failure(stock, config, stage, status, exc, cache_hit=cache_hit))

    result = _finalize_date_rows(rows, config)

    failure_frame = pd.DataFrame(failures)
    coverage = pd.DataFrame([_coverage_row(universe, stocks, config, result, failure_frame)])
    return BatchScanResult(result, failure_frame, coverage)


def batch_scan(
    universe: pd.DataFrame,
    provider: TushareProvider | None,
    store: BarStore,
    financial_store: FinancialStore,
    config: BatchScanConfig,
    *,
    technical_config: TechnicalConfig | None = None,
) -> pd.DataFrame:
    """Backward-compatible candidate-only wrapper."""
    return batch_scan_detailed(
        universe, provider, store, financial_store, config,
        technical_config=technical_config,
    ).candidates
