import pandas as pd

from qfree.backtest.engine import BacktestConfig, backtest
from qfree.backtest.event_study import evaluate_events, summarize_events


def test_window_end_exit_is_separate_from_fixed_holding():
    dates = pd.date_range("2026-09-15", periods=6, freq="D")
    close = pd.Series([10, 10, 10, 11, 12, 13], dtype=float)
    bars = pd.DataFrame({"date": dates, "open": close, "high": close + .2,
                         "low": close - .2, "close": close, "volume": 1000,
                         "amount": 100000})
    signals = pd.DataFrame({"ts_code": ["002487.SZ"], "date": [dates[0]], "selected": [True]})
    events = evaluate_events(signals, {"002487.SZ": bars}, (5,), exit_modes=("fixed_open", "window_end_close"))
    assert set(events["exit_mode"]) == {"fixed_open", "window_end_close"}
    assert events.loc[events["exit_mode"].eq("window_end_close"), "status"].iloc[0] == "window_end_exit"
    assert events.loc[events["exit_mode"].eq("fixed_open"), "status"].iloc[0] == "insufficient_future_data"


def test_event_study_enters_next_open_and_marks_short_history():
    dates = pd.date_range("2026-09-15", periods=11, freq="D")
    close = pd.Series([10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11], dtype=float)
    bars = pd.DataFrame({"date": dates, "open": close, "high": close + .2,
                         "low": close - .2, "close": close, "volume": 1000,
                         "amount": 100000})
    signals = pd.DataFrame({"ts_code": ["002487.SZ"], "name": ["大金重工"],
                            "date": [dates[0]], "selected": [True]})
    events = evaluate_events(signals, {"002487.SZ": bars}, (5, 20), BacktestConfig())
    assert events.loc[events["holding_days"].eq(5), "status"].iloc[0] == "ok"
    assert events.loc[events["holding_days"].eq(5), "entry_date"].iloc[0] == dates[1]
    assert events.loc[events["holding_days"].eq(20), "status"].iloc[0] == "insufficient_future_data"
    assert summarize_events(events).iloc[0]["samples"] == 1


def test_event_marks_non_tradable_entry_and_exit():
    dates = pd.date_range("2026-09-15", periods=5, freq="D")
    close = pd.Series([10, 10, 10, 11, 12], dtype=float)
    bars = pd.DataFrame({"date": dates, "open": close, "high": close + .2,
                         "low": close - .2, "close": close, "volume": 1000,
                         "amount": 100000, "trade_status": ["open", "closed", "open", "open", "open"]})
    signals = pd.DataFrame({"ts_code": ["002487.SZ"], "date": [dates[0]], "selected": [True]})
    events = evaluate_events(signals, {"002487.SZ": bars}, (2,))
    assert events.loc[0, "status"] == "entry_not_tradable"


def test_event_costs_match_next_open_backtest():
    dates = pd.date_range("2026-09-15", periods=6, freq="D")
    close = pd.Series([10, 10, 10, 11, 12, 13], dtype=float)
    bars = pd.DataFrame({"date": dates, "open": close, "high": close + .2,
                         "low": close - .2, "close": close, "volume": 1000,
                         "amount": 100000})
    config = BacktestConfig(holding_days=2)
    signals = pd.DataFrame({"ts_code": ["002487.SZ"], "date": [dates[0]], "selected": [True]})
    events = evaluate_events(signals, {"002487.SZ": bars}, (2,), config)
    trades = backtest(bars, signals[["date", "selected"]], config)
    assert events.loc[0, "net_return"] == trades.loc[0, "net_return"]
    assert events.loc[0, "commission_cost"] == config.commission * 2
    assert events.loc[0, "stamp_tax_cost"] == config.stamp_tax


def test_profit_target_close_exits_at_target_before_max_holding():
    dates = pd.date_range("2026-09-15", periods=8, freq="D")
    close = pd.Series([10, 10, 10.7, 10.5, 10.4, 10.3, 10.2, 10.1], dtype=float)
    bars = pd.DataFrame({"date": dates, "open": close, "high": close + .2,
                         "low": close - .2, "close": close, "volume": 1000,
                         "amount": 100000})
    signals = pd.DataFrame({"ts_code": ["002487.SZ"], "date": [dates[0]], "selected": [True]})
    events = evaluate_events(
        signals, {"002487.SZ": bars}, (5,),
        exit_modes=("profit_target_close",), profit_target=0.06,
    )
    row = events.iloc[0]
    assert row["status"] == "profit_target_exit"
    assert row["exit_date"] == dates[2]
    assert row["exit_price_source"] == "close"
    assert row["realized_holding_days"] == 1
    assert summarize_events(events).iloc[0]["samples"] == 1


def test_profit_target_close_uses_max_holding_when_target_is_missed():
    dates = pd.date_range("2026-09-15", periods=8, freq="D")
    close = pd.Series([10, 10, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1], dtype=float)
    bars = pd.DataFrame({"date": dates, "open": close, "high": close + .2,
                         "low": close - .2, "close": close, "volume": 1000,
                         "amount": 100000})
    signals = pd.DataFrame({"ts_code": ["002487.SZ"], "date": [dates[0]], "selected": [True]})
    events = evaluate_events(
        signals, {"002487.SZ": bars}, (5,),
        exit_modes=("profit_target_close",), profit_target=0.06,
    )
    row = events.iloc[0]
    assert row["status"] == "max_holding_exit"
    assert row["exit_date"] == dates[6]
    assert row["realized_holding_days"] == 5
