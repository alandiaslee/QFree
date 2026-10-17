import pandas as pd

from qfree.backtest.engine import BacktestConfig, backtest, summarize


def test_backtest_enters_next_open_and_charges_costs():
    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    close = pd.Series([10, 10, 10, 11, 11, 11, 11, 11], dtype=float)
    bars = pd.DataFrame({"date": dates, "open": close, "high": close + 0.2,
                         "low": close - 0.2, "close": close, "volume": 1000,
                         "amount": 100000})
    signals = pd.DataFrame({"date": [dates[0]], "selected": [True]})
    trades = backtest(bars, signals, BacktestConfig(holding_days=2))
    assert len(trades) == 1
    assert trades.iloc[0]["entry_date"] == dates[1]
    assert trades.iloc[0]["net_return"] < trades.iloc[0]["gross_return"]
    assert summarize(trades).iloc[0]["trades"] == 1
