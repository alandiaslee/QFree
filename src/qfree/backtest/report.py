"""Reusable reports for multi-date screening and event validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .engine import BacktestConfig
from .event_study import ExitMode, evaluate_events, summarize_events
from .walk_forward import WalkForwardResult


@dataclass(frozen=True)
class WalkForwardEventResult:
    """Event rows and per-date/pooled summaries for a scan result."""

    events: pd.DataFrame
    summary: pd.DataFrame


def evaluate_walk_forward_events(
    scan: WalkForwardResult,
    bars_by_symbol: dict[str, pd.DataFrame],
    holding_periods: tuple[int, ...] = (5, 10, 20),
    config: BacktestConfig | None = None,
    *,
    exit_modes: tuple[ExitMode, ...] = ("fixed_open",),
    trade_dates: pd.DatetimeIndex | None = None,
    profit_target: float | None = None,
) -> WalkForwardEventResult:
    """Evaluate selected candidates without pooling observation dates prematurely."""
    if scan.candidates.empty or "selected" not in scan.candidates:
        return WalkForwardEventResult(pd.DataFrame(), pd.DataFrame())

    event_frames: list[pd.DataFrame] = []
    selected = scan.candidates["selected"]
    if not pd.api.types.is_bool_dtype(selected):
        selected = selected.astype("string").str.lower().isin(("true", "1", "yes", "y"))
    candidates = scan.candidates.loc[selected].copy()
    for as_of, group in candidates.groupby("as_of", dropna=False):
        signals = group.rename(columns={"signal_date": "date"})
        signals = signals[[column for column in ("ts_code", "name", "date", "selected")
                           if column in signals]]
        events = evaluate_events(
            signals, bars_by_symbol, holding_periods, config,
            exit_modes=exit_modes, trade_dates=trade_dates, profit_target=profit_target,
        )
        if not events.empty:
            events.insert(0, "as_of", as_of)
            event_frames.append(events)

    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    if events.empty:
        return WalkForwardEventResult(events, pd.DataFrame())

    summaries: list[pd.DataFrame] = []
    for as_of, group in events.groupby("as_of", dropna=False):
        current = summarize_events(group).copy()
        current.insert(0, "as_of", as_of)
        current.insert(0, "scope", "observation_date")
        summaries.append(current)
    pooled = summarize_events(events.drop(columns=["as_of"])).copy()
    pooled.insert(0, "as_of", pd.NaT)
    pooled.insert(0, "scope", "pooled")
    summaries.append(pooled)
    summary = pd.concat(summaries, ignore_index=True)
    return WalkForwardEventResult(events, summary)
