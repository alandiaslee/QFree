import pandas as pd

from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.strategy.batch_scan import BatchScanConfig, _load_bars


class CalendarProvider:
    name = "fixture"

    def __init__(self, bars):
        self.bars = bars

    def get_daily_bars(self, instrument, start, end, adjust):
        return self.bars.copy()


def _bars(dates):
    close = pd.Series(range(10, 10 + len(dates)), dtype=float)
    return pd.DataFrame({
        "date": dates, "open": close, "high": close + 1,
        "low": close - 0.5, "close": close, "volume": 1000.0,
        "amount": 100000.0,
    })


def test_trade_calendar_rejects_missing_middle_session(tmp_path):
    dates = pd.to_datetime(["2026-09-01", "2026-09-03", "2026-09-04"])
    provider = CalendarProvider(_bars(dates))
    config = BatchScanConfig(
        as_of="2026-09-04", start="2026-09-01", end="2026-09-04"
    )
    try:
        _load_bars(
            Instrument("002487"), provider, BarStore(tmp_path), config,
            trade_dates=pd.DatetimeIndex(pd.to_datetime([
                "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
            ])),
        )
    except FileNotFoundError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("incomplete trade-session coverage was accepted")
