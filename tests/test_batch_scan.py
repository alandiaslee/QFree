import numpy as np
import pandas as pd

from qfree.data.financial_store import FinancialStore
from qfree.data.models import Instrument
from qfree.data.store import BarStore
from qfree.strategy.batch_scan import BatchScanConfig, batch_scan_detailed


class FakeProvider:
    name = "fixture"

    def __init__(self, bars: pd.DataFrame):
        self.bars = bars
        self.bar_calls = 0

    def get_daily_bars(self, instrument, start, end, adjust):
        self.bar_calls += 1
        return self.bars.copy()

    def financial_snapshot(self, ts_code, *, start_date=None, end_date=None):
        return pd.DataFrame({
            "ts_code": [ts_code],
            "ann_date": ["2025-12-31"],
            "end_date": ["2025-09-30"],
            "roe": [12.0],
            "netprofit_yoy": [10.0],
            "ocf_to_profit": [80.0],
            "debt_to_assets": [40.0],
            "grossprofit_margin": [25.0],
        })


def _bars(start="2026-08-01", periods=80):
    dates = pd.date_range(start, periods=periods, freq="D")
    close = pd.Series(range(10, 10 + periods), dtype=float)
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close + 1,
        "low": close - 0.5,
        "close": close,
        "volume": 1000.0,
        "amount": 100000.0,
    })


def _universe():
    return pd.DataFrame({
        "ts_code": ["002487.SZ"],
        "name": ["大金重工"],
        "industry": ["制造业"],
        "exchange": ["SZSE"],
        "list_date": ["2010-01-01"],
        "list_status": ["L"],
    })


def _startup_bars():
    dates = pd.date_range("2026-01-01", periods=61, freq="B")
    close = np.r_[
        np.full(10, 10.0),
        np.linspace(10, 8, 10),
        np.linspace(8, 8.3, 40),
        [8.3],
    ]
    volume = np.full(len(close), 1000.0)
    volume[-1] = 5000.0
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close + 1,
        "low": close - 0.5,
        "close": close,
        "volume": volume,
        "amount": volume * 100.0,
    })


def _pullback_startup_bars():
    dates = pd.date_range("2026-01-01", periods=76, freq="B")
    close = np.r_[
        np.full(10, 10.0),
        np.linspace(10, 7.5, 15),
        np.linspace(7.5, 9.5, 25),
        np.linspace(9.5, 7.5, 15),
        np.linspace(7.5, 8.0, 10),
        [8.0],
    ]
    volume = np.full(len(close), 1000.0)
    volume[-1] = 5000.0
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close + 1,
        "low": close - 0.5,
        "close": close,
        "volume": volume,
        "amount": volume * 100.0,
    })


def test_stale_cache_is_refreshed_and_incomplete_provider_is_reported(tmp_path):
    instrument = Instrument("002487")
    store = BarStore(tmp_path)
    store.save(instrument, _bars("2026-08-01", 20), adjust="qfq", source="fixture")
    provider = FakeProvider(_bars("2026-08-01", 80))
    config = BatchScanConfig(as_of="2026-10-01", start="2026-08-01", end="2026-10-01")

    report = batch_scan_detailed(_universe(), provider, store, FinancialStore(tmp_path), config)

    assert provider.bar_calls == 1
    assert report.coverage.iloc[0]["failures"] == 0
    assert report.candidates.iloc[0]["bars_last_date"] >= pd.Timestamp("2026-10-01")

    short_provider = FakeProvider(_bars("2026-08-01", 20))
    report = batch_scan_detailed(
        _universe(), short_provider, BarStore(tmp_path / "short"),
        FinancialStore(tmp_path / "short"), config,
    )
    assert report.failures.iloc[0]["status"] == "incomplete_bar_coverage"


def test_financial_field_threshold_blocks_selection_but_retains_snapshot(tmp_path):
    provider = FakeProvider(_bars("2026-08-01", 80))
    config = BatchScanConfig(
        as_of="2026-10-01", start="2026-08-01", end="2026-10-01",
        min_score=0, min_financial_fields=6,
    )
    report = batch_scan_detailed(
        _universe(), provider, BarStore(tmp_path), FinancialStore(tmp_path), config,
    )

    assert len(report.candidates) == 1
    assert not bool(report.candidates.iloc[0]["selected"])
    assert report.candidates.iloc[0]["financial_field_count"] == 5
    assert report.coverage.iloc[0]["financial_field_coverage"] == 1.0


def test_offline_scan_reports_cache_miss_without_provider_calls(tmp_path):
    provider = FakeProvider(_bars("2026-08-01", 80))
    config = BatchScanConfig(
        as_of="2026-10-01", start="2026-08-01", end="2026-10-01",
        allow_provider_fetch=False,
    )
    report = batch_scan_detailed(
        _universe(), provider, BarStore(tmp_path), FinancialStore(tmp_path), config,
    )

    assert provider.bar_calls == 0
    assert report.failures.iloc[0]["status"] == "cache_miss_provider_unavailable"
    assert report.coverage.iloc[0]["failures"] == 1


def test_bad_symbol_failure_does_not_abort_other_symbols(tmp_path):
    store = BarStore(tmp_path)
    good = _bars("2026-08-01", 80)
    store.save(Instrument("000001"), good, adjust="qfq", source="fixture")
    universe = pd.DataFrame({
        "ts_code": ["000001.SZ", "002487.SZ"],
        "name": ["平安银行", "大金重工"],
        "industry": ["金融业", "制造业"],
        "exchange": ["SZSE", "SZSE"],
        "list_date": ["2010-01-01", "2010-01-01"],
        "list_status": ["L", "L"],
    })
    provider = FakeProvider(_bars("2026-08-01", 80))

    class BadProvider(FakeProvider):
        def get_daily_bars(self, instrument, start, end, adjust):
            raise KeyError("bad cache columns")

    config = BatchScanConfig(as_of="2026-10-01", start="2026-08-01", end="2026-10-01", min_score=0)
    report = batch_scan_detailed(
        universe, BadProvider(provider.bars), store, FinancialStore(tmp_path), config,
    )

    assert len(report.candidates) == 1
    assert set(report.failures["status"]) == {"unexpected_error"}
    assert report.failures.iloc[0]["ts_code"] == "002487.SZ"


def test_market_gate_blocks_low_breadth_dates(tmp_path):
    store = BarStore(tmp_path)
    store.save(Instrument("000001"), _startup_bars(), adjust="qfq", source="fixture")
    store.save(Instrument("002487"), _bars("2026-01-01", 85), adjust="qfq", source="fixture")
    universe = pd.DataFrame({
        "ts_code": ["000001.SZ", "002487.SZ"],
        "name": ["平安银行", "大金重工"],
        "industry": ["金融业", "制造业"],
        "exchange": ["SZSE", "SZSE"],
        "list_date": ["2010-01-01", "2010-01-01"],
        "list_status": ["L", "L"],
    })
    provider = FakeProvider(_bars("2026-01-01", 85))
    config = BatchScanConfig(
        as_of="2026-03-26", start="2026-01-01", end="2026-03-26",
        min_score=0, use_market_gate=True,
        min_startup_breadth=0.75, min_market_return_5d=10.0,
    )

    report = batch_scan_detailed(universe, provider, store, FinancialStore(tmp_path), config)

    assert int(report.candidates["startup_signal"].sum()) == 1
    assert int(report.candidates["selected"].sum()) == 0
    assert not bool(report.candidates["market_gate_pass"].iloc[0])
    assert not bool(report.coverage.iloc[0]["market_gate_pass"])
    assert report.coverage.iloc[0]["startup_signal_breadth"] == 0.5


def test_pullback_quality_boost_adds_score_points(tmp_path):
    store = BarStore(tmp_path)
    store.save(Instrument("002487"), _pullback_startup_bars(), adjust="qfq", source="fixture")
    provider = FakeProvider(_pullback_startup_bars())
    config = BatchScanConfig(
        as_of="2026-04-16", start="2026-01-01", end="2026-04-16",
        min_score=0, use_pullback_quality_boost=True, boost_points=5.0,
    )

    report = batch_scan_detailed(_universe(), provider, store, FinancialStore(tmp_path), config)
    row = report.candidates.iloc[0]

    assert bool(row["startup_signal"])
    assert row["return_20d"] < -0.006
    assert row["financial_field_count"] == 5
    assert bool(row["pullback_quality_boost"])
    assert row["boost_points"] == 5.0
    assert row["total_score"] == row["base_total_score"] + 5.0
