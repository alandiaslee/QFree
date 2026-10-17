"""Cross-sectional quality scoring with robust missing-value handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

QUALITY_WEIGHTS = {
    "roe": 0.25,
    "netprofit_yoy": 0.20,
    "ocf_to_profit": 0.20,
    "debt_to_assets": -0.15,
    "grossprofit_margin": 0.20,
}


def _winsorize(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    if series.notna().sum() < 3:
        return series
    return series.clip(series.quantile(lower), series.quantile(upper))


def _robust_zscore(values: pd.Series) -> pd.Series:
    median = values.median()
    scale = (values - median).abs().median()
    if pd.isna(scale) or scale == 0:
        scale = values.std(ddof=0)
    if pd.isna(scale) or scale == 0:
        return values * 0.0
    return (values - median) / scale


def quality_score(financials: pd.DataFrame, *, group_column: str | None = "industry") -> pd.DataFrame:
    """Return quality sub-scores and a 0-100 score for one as-of date."""
    result = financials.copy()
    components: list[pd.Series] = []
    for column, weight in QUALITY_WEIGHTS.items():
        if column not in result:
            result[column] = np.nan
        values = _winsorize(pd.to_numeric(result[column], errors="coerce"))
        if group_column and group_column in result:
            groups = result[group_column].fillna("__missing__")
            counts = groups.value_counts()
            valid_groups = groups.map(counts).ge(3)
            zscore = _robust_zscore(values)
            if valid_groups.any():
                centered = values - values.groupby(groups).transform("median")
                scale = centered.groupby(groups).transform(
                    lambda group: (group - group.mean()).abs().median()
                )
                group_zscore = centered / scale.replace(0, np.nan)
                zscore = group_zscore.where(valid_groups, zscore)
        else:
            zscore = _robust_zscore(values)
        zscore = zscore.replace([np.inf, -np.inf], np.nan).clip(-3, 3)
        result[f"{column}_z"] = zscore
        components.append(zscore.fillna(0) * weight)
    available = result[list(QUALITY_WEIGHTS)].notna().sum(axis=1)
    valid_weight = sum(
        result[column].notna().astype(float) * abs(weight)
        for column, weight in QUALITY_WEIGHTS.items()
    )
    result["quality_score"] = (50 + sum(components) * 12.5).clip(0, 100)
    result.loc[available == 0, "quality_score"] = np.nan
    result["quality_valid_count"] = available
    result["quality_valid_weight"] = valid_weight
    result["quality_missing_count"] = len(QUALITY_WEIGHTS) - available
    return result
