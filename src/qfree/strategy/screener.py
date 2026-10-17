"""Combine quality and bottom-volume-start signals into ranked candidates."""

from __future__ import annotations

import pandas as pd

from ..features.quality import quality_score
from ..features.technical import TechnicalConfig, calculate_technical, technical_score


def screen(bars: pd.DataFrame, financials: pd.DataFrame | None = None, *,
           technical_config: TechnicalConfig | None = None, min_score: float = 60.0) -> pd.DataFrame:
    """Screen one symbol's bars, optionally joining a point-in-time financial row."""
    technical = calculate_technical(bars, technical_config)
    technical["technical_score"] = technical_score(technical)
    if financials is None or financials.empty:
        technical["quality_score"] = 50.0
        technical["quality_missing_count"] = 5
        technical["quality_valid_weight"] = 0.0
        technical["financial_field_count"] = 0
        technical["data_quality_status"] = "missing_financials"
    else:
        quality = quality_score(financials)
        if len(quality) != 1:
            raise ValueError("financials must contain one as-of row for a single-symbol screen")
        technical["quality_score"] = float(quality["quality_score"].iloc[0])
        technical["quality_missing_count"] = int(quality["quality_missing_count"].iloc[0])
        technical["quality_valid_weight"] = float(quality["quality_valid_weight"].iloc[0])
        technical["financial_field_count"] = 5 - technical["quality_missing_count"]
        technical["data_quality_status"] = technical["financial_field_count"].map(
            lambda count: "complete" if count == 5 else "partial_financials"
        )
    technical["total_score"] = 0.55 * technical["quality_score"] + 0.45 * technical["technical_score"]
    technical["selected"] = technical["startup_signal"] & technical["total_score"].ge(min_score)
    technical["reason"] = technical.apply(
        lambda row: "底部位置+放量+趋势改善" if row["selected"] else "", axis=1
    )
    return technical
