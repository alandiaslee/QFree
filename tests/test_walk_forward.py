import pandas as pd

from qfree.backtest.report import evaluate_walk_forward_events
from qfree.backtest.walk_forward import WalkForwardResult, walk_forward_scan
from qfree.data.financial_store import FinancialStore
from qfree.data.store import BarStore
from qfree.features.technical import TechnicalConfig
from qfree.strategy.batch_scan import BatchScanConfig


class RecordingProvider:
    name = "fixture"

    def __init__(self, bars):
        self.bars = bars
        self.calls = []

    def get_daily_bars(self, instrument, start, end, adjust):
        self.calls.append((instrument.ts_code, pd.Timestamp(start), pd.Timestamp(end)))
        return self.bars[self.bars["date"].between(start, end)].copy()

    def financial_snapshot(self, ts_code, *, start_date=None, end_date=None):
        return pd.DataFrame({
            "ts_code": [ts_code], "ann_date": ["2025-12-31"],
            "end_date": ["2025-09-30"], "roe": [12.0],
            "netprofit_yoy": [10.0], "ocf_to_profit": [80.0],
            "debt_to_assets": [40.0], "grossprofit_margin": [25.0],
        })


def test_walk_forward_preserves_technical_config_and_caps_end(tmp_path):
    dates = pd.date_range("2026-07-01", periods=100, freq="D")
    close = pd.Series(range(10, 110), dtype=float)
    bars = pd.DataFrame({
        "date": dates, "open": close, "high": close + 1,
        "low": close - 0.5, "close": close,
        "volume": 1000.0, "amount": 100000.0,
    })
    universe = pd.DataFrame({
        "ts_code": ["002487.SZ"], "name": ["大金重工"],
        "industry": ["制造业"], "exchange": ["SZSE"],
        "list_date": ["2010-01-01"], "list_status": ["L"],
    })
    provider = RecordingProvider(bars)
    custom = TechnicalConfig(bottom_window=10, volume_window=4, volume_ratio=1.1)
    base = BatchScanConfig(
        as_of="2026-08-01", start="2026-07-01", end="2026-12-31", min_score=0,
    )

    report = walk_forward_scan(
        universe, provider, BarStore(tmp_path), FinancialStore(tmp_path),
        ["2026-08-01", "2026-08-15"], start="2026-07-01", end="2026-12-31",
        config=base, technical_config=custom,
    )

    assert set(report.coverage["as_of"].dt.strftime("%Y-%m-%d")) == {
        "2026-08-01", "2026-08-15",
    }
    assert all(call[2] <= pd.Timestamp("2026-08-15") for call in provider.calls)
    assert all(call[1] == pd.Timestamp("2026-07-01") for call in provider.calls)


def test_walk_forward_event_report_separates_dates_and_pooled_summary():
    dates = pd.date_range("2026-09-01", periods=8, freq="D")
    close = pd.Series([10, 10, 10, 11, 12, 13, 14, 15], dtype=float)
    bars = pd.DataFrame({"date": dates, "open": close, "high": close + .2,
                         "low": close - .2, "close": close, "volume": 1000,
                         "amount": 100000})
    candidates = pd.DataFrame({
        "as_of": [dates[0], dates[1]], "ts_code": ["002487.SZ", "002487.SZ"],
        "name": ["大金重工", "大金重工"], "signal_date": [dates[0], dates[1]],
        "selected": [True, True],
    })
    result = evaluate_walk_forward_events(
        WalkForwardResult(candidates, pd.DataFrame(), pd.DataFrame()),
        {"002487.SZ": bars}, holding_periods=(2,),
    )
    assert set(result.summary["scope"]) == {"observation_date", "pooled"}
    assert len(result.events) == 2
