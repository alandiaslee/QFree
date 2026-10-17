import pandas as pd

from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.strategy.screener import screen

bars = BarStore("data").load(
    Instrument("002487"),
    "2025-01-01",
    "2026-09-30",
)

result = screen(bars)
row = result[result["date"] == pd.Timestamp("2026-09-15")]

columns = [
    "date",
    "close",
    "return_5d",
    "return_20d",
    "trend_up",
    "startup_signal",
    "technical_score",
    "quality_score",
    "total_score",
    "selected",
    "reason",
]

dynamic_columns = [
    c for c in result.columns
    if c.startswith("volume_ratio_") or c.startswith("price_position_")
]

print(row[dynamic_columns + columns].to_string(index=False))
