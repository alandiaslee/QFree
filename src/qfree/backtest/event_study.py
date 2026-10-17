"""Point-in-time event returns after a screening signal."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from .engine import BacktestConfig, execute_trade

ExitMode = Literal["fixed_open", "window_end_close", "profit_target_close"]


def _selected_mask(signals: pd.DataFrame) -> pd.Series:
    values = signals.get("selected", pd.Series(False, index=signals.index))
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.lower().isin(("true", "1", "yes", "y"))


def evaluate_events(
    signals: pd.DataFrame,
    bars_by_symbol: dict[str, pd.DataFrame],
    holding_periods: tuple[int, ...] = (5, 10, 20),
    config: BacktestConfig | None = None,
    *,
    exit_modes: tuple[ExitMode, ...] = ("fixed_open",),
    trade_dates: pd.DatetimeIndex | None = None,
    profit_target: float | None = None,
) -> pd.DataFrame:
    """Evaluate selected signals at the next open without extrapolating exits.

    ``fixed_open`` exits after ``holding`` subsequent trading sessions at the
    next session's open. ``window_end_close`` is a separate diagnostic exit at
    the final available bar close and must not be mixed with fixed periods.
    ``profit_target_close`` treats each holding value as a maximum holding
    period and exits at the first close after entry that reaches the target,
    otherwise at the maximum-period close.
    """
    config = config or BacktestConfig()
    invalid_modes = set(exit_modes) - {"fixed_open", "window_end_close", "profit_target_close"}
    if invalid_modes:
        raise ValueError(f"unsupported exit mode(s): {sorted(invalid_modes)}")
    if any(holding <= 0 for holding in holding_periods):
        raise ValueError("holding periods must be positive")
    if "profit_target_close" in exit_modes:
        if profit_target is None or profit_target <= 0:
            raise ValueError("profit_target must be positive for profit_target_close exits")

    selected = signals.loc[_selected_mask(signals)].copy()
    rows: list[dict[str, object]] = []
    for signal in selected.itertuples(index=False):
        symbol = str(signal.ts_code)
        signal_date = pd.Timestamp(signal.date).normalize()
        base = {
            "ts_code": symbol,
            "name": getattr(signal, "name", None),
            "signal_date": signal_date,
        }
        bars = bars_by_symbol.get(symbol)
        if bars is None:
            for mode in exit_modes:
                periods = holding_periods if mode in ("fixed_open", "profit_target_close") else (None,)
                for holding in periods:
                    rows.append({
                        **base,
                        "exit_mode": mode,
                        "holding_days": holding,
                        "status": "missing_bars",
                    })
            continue

        frame = bars.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame = frame.dropna(subset=["date"]).sort_values("date")
        frame = frame.drop_duplicates("date").reset_index(drop=True)
        if "open" not in frame or "close" not in frame:
            for mode in exit_modes:
                periods = holding_periods if mode in ("fixed_open", "profit_target_close") else (None,)
                for holding in periods:
                    rows.append({**base, "exit_mode": mode, "holding_days": holding,
                                 "status": "invalid_bar_schema"})
            continue
        if trade_dates is not None:
            session_dates = pd.DatetimeIndex(trade_dates)
            frame = frame[frame["date"].isin(session_dates)].reset_index(drop=True)
        future_positions = frame.index[frame["date"].gt(signal_date)].tolist()
        if not future_positions:
            for mode in exit_modes:
                periods = holding_periods if mode in ("fixed_open", "profit_target_close") else (None,)
                for holding in periods:
                    rows.append({
                        **base,
                        "exit_mode": mode,
                        "holding_days": holding,
                        "status": "missing_signal_date",
                    })
            continue

        entry_index = future_positions[0]
        for mode in exit_modes:
            periods = holding_periods if mode in ("fixed_open", "profit_target_close") else (None,)
            for holding in periods:
                result: dict[str, object] = {
                    **base,
                    "exit_mode": mode,
                    "holding_days": holding,
                    "window_end_date": frame["date"].iloc[-1],
                }
                entry_row = frame.iloc[entry_index]
                if bool(entry_row.get("is_suspended", False)) or str(
                    entry_row.get("trade_status", "open")
                ).lower() not in ("open", "1", "true", "nan"):
                    result["status"] = "entry_not_tradable"
                    rows.append(result)
                    continue
                entry_price = pd.to_numeric(
                    pd.Series([entry_row["open"]]), errors="coerce"
                ).iloc[0]
                up_limit = pd.to_numeric(
                    pd.Series([entry_row.get("up_limit")]), errors="coerce"
                ).iloc[0]
                if pd.notna(up_limit) and float(entry_price) >= float(up_limit):
                    result["status"] = "entry_limit_up_not_tradable"
                    rows.append(result)
                    continue
                if pd.isna(entry_price) or float(entry_price) <= 0:
                    result["status"] = "invalid_entry_bar"
                    rows.append(result)
                    continue

                if mode == "fixed_open":
                    assert holding is not None
                    exit_index = entry_index + holding
                    if exit_index >= len(frame):
                        result["status"] = "insufficient_future_data"
                        rows.append(result)
                        continue
                    exit_row = frame.iloc[exit_index]
                    if bool(exit_row.get("is_suspended", False)) or str(
                        exit_row.get("trade_status", "open")
                    ).lower() not in ("open", "1", "true", "nan"):
                        result["status"] = "exit_not_tradable"
                        rows.append(result)
                        continue
                    exit_price = float(exit_row["open"])
                    down_limit = pd.to_numeric(
                        pd.Series([exit_row.get("down_limit")]), errors="coerce"
                    ).iloc[0]
                    if pd.notna(down_limit) and exit_price <= float(down_limit):
                        result["status"] = "exit_limit_down_not_tradable"
                        rows.append(result)
                        continue
                    exit_date = frame["date"].iloc[exit_index]
                    status = "ok"
                    price_source = "open"
                elif mode == "profit_target_close":
                    assert holding is not None
                    assert profit_target is not None
                    target_index: int | None = None
                    last_candidate = min(entry_index + holding, len(frame) - 1)
                    for candidate_index in range(entry_index + 1, last_candidate + 1):
                        candidate_close = pd.to_numeric(
                            pd.Series([frame.iloc[candidate_index]["close"]]), errors="coerce"
                        ).iloc[0]
                        if pd.notna(candidate_close) and float(candidate_close) / float(entry_price) - 1 >= profit_target:
                            target_index = candidate_index
                            break
                    if target_index is not None:
                        exit_index = target_index
                        status = "profit_target_exit"
                    else:
                        exit_index = entry_index + holding
                        if exit_index >= len(frame):
                            result["status"] = "insufficient_future_data"
                            rows.append(result)
                            continue
                        status = "max_holding_exit"
                    exit_row = frame.iloc[exit_index]
                    if bool(exit_row.get("is_suspended", False)) or str(
                        exit_row.get("trade_status", "open")
                    ).lower() not in ("open", "1", "true", "nan"):
                        result["status"] = "exit_not_tradable"
                        rows.append(result)
                        continue
                    exit_price = float(exit_row["close"])
                    exit_date = frame["date"].iloc[exit_index]
                    price_source = "close"
                else:
                    exit_index = len(frame) - 1
                    if exit_index <= entry_index:
                        result["status"] = "insufficient_future_data"
                        rows.append(result)
                        continue
                    exit_row = frame.iloc[exit_index]
                    if bool(exit_row.get("is_suspended", False)) or str(
                        exit_row.get("trade_status", "open")
                    ).lower() not in ("open", "1", "true", "nan"):
                        result["status"] = "exit_not_tradable"
                        rows.append(result)
                        continue
                    exit_price = float(exit_row["close"])
                    exit_date = frame["date"].iloc[exit_index]
                    status = "window_end_exit"
                    price_source = "close"

                if not pd.notna(exit_price) or exit_price <= 0:
                    result["status"] = "invalid_exit_bar"
                    rows.append(result)
                    continue
                try:
                    execution = execute_trade(entry_price, exit_price, config)
                except ValueError:
                    result["status"] = "invalid_exit_bar"
                    rows.append(result)
                    continue
                result.update({
                    "status": status,
                    "entry_date": frame["date"].iloc[entry_index],
                    "exit_date": exit_date,
                    "realized_holding_days": int(exit_index - entry_index),
                    "entry_price_source": "open",
                    "exit_price_source": price_source,
                    **execution,
                })
                rows.append(result)
    return pd.DataFrame(rows)


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize only completed events, preserving exit-mode separation."""
    if events.empty:
        return pd.DataFrame()
    group_columns = ["exit_mode", "holding_days"]
    for column in group_columns:
        if column not in events:
            events[column] = None
    summaries: list[dict[str, object]] = []
    for (mode, holding), group in events.groupby(group_columns, dropna=False):
        valid = group[group["status"].isin(("ok", "window_end_exit", "profit_target_exit", "max_holding_exit"))]
        returns = pd.to_numeric(valid.get("net_return", pd.Series(dtype=float)), errors="coerce").dropna()
        summaries.append({
            "exit_mode": mode,
            "holding_days": holding,
            "samples": len(returns),
            "total_events": len(group),
            "insufficient_samples": len(group) - len(returns),
            "mean_net_return": float(returns.mean()) if not returns.empty else None,
            "median_net_return": float(returns.median()) if not returns.empty else None,
            "win_rate": float((returns > 0).mean()) if not returns.empty else None,
            "p25_net_return": float(returns.quantile(0.25)) if not returns.empty else None,
            "p75_net_return": float(returns.quantile(0.75)) if not returns.empty else None,
        })
    return pd.DataFrame(summaries).sort_values(group_columns).reset_index(drop=True)
