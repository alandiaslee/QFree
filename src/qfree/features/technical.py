"""Explainable bottom-and-volume-start technical features."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TechnicalConfig:
    bottom_window: int = 60
    volume_window: int = 20
    volume_ratio: float = 1.8
    bottom_percentile: float = 0.35
    max_volume_ratio: float | None = None  # gentle-volume cap; None disables
    fast_window: int = 5
    slow_window: int = 20
    max_recent_return: float = 0.25

    def __post_init__(self) -> None:
        if min(self.bottom_window, self.volume_window, self.fast_window, self.slow_window) < 1:
            raise ValueError("technical windows must be positive")
        if not 0 < self.bottom_percentile < 1:
            raise ValueError("bottom_percentile must be between 0 and 1")
        if self.volume_ratio <= 0 or self.max_recent_return < -1:
            raise ValueError("technical thresholds are invalid")
        if self.max_volume_ratio is not None and self.max_volume_ratio < self.volume_ratio:
            raise ValueError("max_volume_ratio must be >= volume_ratio")


def calculate_technical(bars: pd.DataFrame, config: TechnicalConfig | None = None) -> pd.DataFrame:
    """Calculate features using only data through each row's date."""
    config = config or TechnicalConfig()
    result = bars.sort_values("date").reset_index(drop=True).copy()
    close = result["close"].astype(float)
    volume = result["volume"].astype(float)
    result["return_5d"] = close.pct_change(config.fast_window)
    result["return_20d"] = close.pct_change(config.slow_window)
    rolling_min = close.rolling(config.bottom_window).min()
    rolling_max = close.rolling(config.bottom_window).max()
    result[f"price_position_{config.bottom_window}d"] = (
        (close - rolling_min) / (rolling_max - rolling_min)
    )
    baseline_volume = volume.shift(1).rolling(config.volume_window).mean()
    result[f"volume_ratio_{config.volume_window}d"] = (
        volume.div(baseline_volume).where(baseline_volume.gt(0))
    )
    result["ma_fast"] = close.rolling(config.fast_window).mean()
    result["ma_slow"] = close.rolling(config.slow_window).mean()
    # 20-session mean turnover in the bars' amount unit (thousand CNY from Tushare).
    result["avg_amount_20d"] = result["amount"].astype(float).rolling(20).mean()
    result["trend_up"] = (result["ma_fast"] > result["ma_slow"]) & (result["ma_fast"].diff() > 0)
    result["startup_signal"] = (
        result[f"price_position_{config.bottom_window}d"].le(config.bottom_percentile)
        & result[f"volume_ratio_{config.volume_window}d"].ge(config.volume_ratio)
        & result["trend_up"]
        & result["return_20d"].le(config.max_recent_return)
    )
    return result


def technical_score(features: pd.DataFrame) -> pd.Series:
    """A 0-100 score retaining NaN for insufficient history."""
    position_column = next(column for column in features if column.startswith("price_position_"))
    volume_column = next(column for column in features if column.startswith("volume_ratio_"))
    position = (1 - features[position_column]).clip(0, 1)
    volume = ((features[volume_column] - 1) / 2).clip(0, 1)
    trend = features["trend_up"].astype(float)
    score = 100 * (0.4 * position + 0.35 * volume + 0.25 * trend)
    return score.where(features[position_column].notna())
