import pandas as pd

from qfree.features.technical import TechnicalConfig, calculate_technical
from qfree.strategy.screener import screen


def test_future_rows_do_not_change_previous_features():
    dates = pd.date_range("2026-01-01", periods=80, freq="D")
    close = pd.Series(range(10, 90), dtype=float)
    bars = pd.DataFrame({"date": dates, "open": close, "high": close + 1,
                         "low": close - 0.5, "close": close, "volume": 1000.0,
                         "amount": 100000.0})
    first = calculate_technical(bars.iloc[:70], TechnicalConfig())
    second = calculate_technical(bars, TechnicalConfig())
    pd.testing.assert_series_equal(first["price_position_60d"], second.iloc[:70]["price_position_60d"], check_names=False)


def test_zero_volume_baseline_does_not_create_infinite_ratio():
    dates = pd.date_range("2026-01-01", periods=80, freq="D")
    close = pd.Series(10.0, index=range(80))
    bars = pd.DataFrame({"date": dates, "open": close, "high": close + 1,
                         "low": close - 0.5, "close": close, "volume": 0.0,
                         "amount": 100000.0})
    result = calculate_technical(bars)
    ratio = result["volume_ratio_20d"]
    assert not ratio.replace([float("inf"), float("-inf")], pd.NA).notna().any()


def test_screen_has_no_start_signal_before_warmup():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    close = pd.Series(10.0, index=range(30))
    bars = pd.DataFrame({"date": dates, "open": close, "high": close + 1,
                         "low": close - 0.5, "close": close, "volume": 1000.0,
                         "amount": 100000.0})
    result = screen(bars)
    assert not result["selected"].any()
