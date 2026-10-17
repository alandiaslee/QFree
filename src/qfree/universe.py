"""Point-in-time-ish stock universe filters for initial research."""

from __future__ import annotations

import pandas as pd


def filter_universe(stocks: pd.DataFrame, *, as_of: str | pd.Timestamp | None = None,
                    min_list_days: int = 250, exchanges: tuple[str, ...] = ("SSE", "SZSE")) -> pd.DataFrame:
    """Filter ST/delisted/new/illiquid names without silently filling missing values."""
    result = stocks.copy()
    required = {"ts_code", "name", "list_date", "exchange"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"missing stock columns: {sorted(missing)}")
    result["list_date"] = pd.to_datetime(result["list_date"], errors="coerce")
    cutoff = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    result = result[result["exchange"].isin(exchanges)]
    result = result[~result["name"].fillna("").str.upper().str.contains("ST|退", regex=True)]
    result = result[result["list_date"].notna() & ((cutoff - result["list_date"]).dt.days >= min_list_days)]
    if "list_status" in result:
        result = result[result["list_status"].fillna("L").eq("L")]
    return result.reset_index(drop=True)
