"""Simple next-open signal backtest with explicit costs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    holding_days: int = 5
    commission: float = 0.0003
    stamp_tax: float = 0.0005
    slippage: float = 0.001

    def __post_init__(self) -> None:
        if self.holding_days <= 0:
            raise ValueError("holding_days must be positive")
        if min(self.commission, self.stamp_tax, self.slippage) < 0:
            raise ValueError("cost rates must be non-negative")


def execute_trade(entry_price: float, exit_price: float, config: BacktestConfig) -> dict[str, float]:
    """Apply the canonical next-open execution and round-trip costs."""
    if entry_price <= 0 or exit_price <= 0:
        raise ValueError("execution prices must be positive")
    entry = float(entry_price) * (1 + config.slippage)
    exit = float(exit_price) * (1 - config.slippage)
    gross = exit / entry - 1
    commission_cost = config.commission * 2
    stamp_tax_cost = config.stamp_tax
    net = gross - commission_cost - stamp_tax_cost
    return {
        "entry": entry,
        "exit": exit,
        "gross_return": gross,
        "commission_cost": commission_cost,
        "stamp_tax_cost": stamp_tax_cost,
        "slippage_cost": (float(exit_price) / float(entry_price) - 1) - gross,
        "net_return": net,
    }


def backtest(bars: pd.DataFrame, signals: pd.DataFrame, config: BacktestConfig | None = None) -> pd.DataFrame:
    config = config or BacktestConfig()
    frame = bars.sort_values("date").reset_index(drop=True)
    signal_frame = signals.set_index("date")
    trades: list[dict[str, object]] = []
    for index, row in frame.iterrows():
        if not bool(signal_frame.get("selected", pd.Series(dtype=bool)).get(row["date"], False)):
            continue
        entry_index = index + 1
        exit_index = entry_index + config.holding_days
        if exit_index >= len(frame):
            continue
        execution = execute_trade(
            frame.iloc[entry_index]["open"], frame.iloc[exit_index]["open"], config
        )
        trades.append({"signal_date": row["date"], "entry_date": frame.iloc[entry_index]["date"],
                       "exit_date": frame.iloc[exit_index]["date"], **execution})
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{"trades": 0, "win_rate": None, "mean_return": None, "max_drawdown": None}])
    equity = (1 + trades["net_return"]).cumprod()
    drawdown = equity / equity.cummax() - 1
    return pd.DataFrame([{"trades": len(trades), "win_rate": float((trades["net_return"] > 0).mean()),
                          "mean_return": float(trades["net_return"].mean()),
                          "max_drawdown": float(drawdown.min())}])
