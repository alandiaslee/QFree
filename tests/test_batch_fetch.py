import json

import pandas as pd
import pytest

from qfree.data.batch_fetch import (
    DERIVATION_MARKER,
    batch_prefetch_bars,
    derive_qfq,
    returns_compatible,
)
from qfree.data.models import Instrument, normalize_bars
from qfree.data.store import BarStore


def _raw_bars(dates, closes):
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": close, "high": close + 1, "low": close - 0.5,
        "close": close, "volume": 1000.0, "amount": 200000.0,
    })


def _factors(dates, values):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "adj_factor": pd.Series(values, dtype=float),
    })


def test_derive_qfq_scales_ohlc_to_latest_factor():
    dates = ["2026-09-10", "2026-09-11", "2026-09-14"]
    raw = _raw_bars(dates, [10.0, 20.0, 40.0])
    factors = _factors(dates, [1.0, 2.0, 4.0])
    qfq = derive_qfq(raw, factors)
    assert qfq["close"].tolist() == pytest.approx([2.5, 10.0, 40.0])
    assert qfq["open"].tolist() == pytest.approx([2.5, 10.0, 40.0])
    # volume/amount stay unadjusted
    assert qfq["volume"].eq(1000.0).all()
    # returns are preserved
    assert qfq["close"].pct_change().dropna().tolist() == pytest.approx([3.0, 3.0])


def test_derive_qfq_requires_factor_overlap():
    raw = _raw_bars(["2026-09-10"], [10.0])
    with pytest.raises(ValueError):
        derive_qfq(raw, _factors(["2026-09-11"], [1.0]))


def test_returns_compatible_ignores_level_but_not_shape():
    old = pd.DataFrame({"date": pd.to_datetime(["2026-09-10", "2026-09-11"]),
                        "close": [100.0, 110.0]})
    same_returns = pd.DataFrame({"date": pd.to_datetime(["2026-09-10", "2026-09-11"]),
                                 "close": [5.0, 5.5]})
    different = pd.DataFrame({"date": pd.to_datetime(["2026-09-10", "2026-09-11"]),
                              "close": [5.0, 6.0]})
    assert returns_compatible(old, same_returns)
    assert not returns_compatible(old, different)


class FakeProvider:
    name = "fixture"
    interval = 0.0

    def __init__(self, daily_by_day, factor_by_day):
        self.daily_by_day = daily_by_day
        self.factor_by_day = factor_by_day

    def daily_by_trade_date(self, trade_date):
        return self.daily_by_day[pd.Timestamp(trade_date).strftime("%Y%m%d")]

    def adj_factor_by_trade_date(self, trade_date):
        return self.factor_by_day[pd.Timestamp(trade_date).strftime("%Y%m%d")]


def _market_frame(day, rows):
    frame = pd.DataFrame(
        rows, columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]
    )
    return frame.assign(trade_date=day)


def _provider(days):
    daily_by_day = {}
    factor_by_day = {}
    for day, rows in days.items():
        daily_rows = [row for row in rows if row[0] != "FACTOR"]
        factor_rows = [row[1:] for row in rows if row[0] == "FACTOR"]
        daily_by_day[day] = _market_frame(day, daily_rows)
        factor_by_day[day] = pd.DataFrame(
            [(code, day, value) for code, value in factor_rows],
            columns=["ts_code", "trade_date", "adj_factor"],
        )
    return FakeProvider(daily_by_day, factor_by_day)


def _stocks(*ts_codes):
    return pd.DataFrame({"ts_code": list(ts_codes), "name": list(ts_codes)})


DAYS = {
    "20260914": [
        ("000001.SZ", "20260914", 10.0, 11.0, 9.5, 10.5, 1000.0, 10000.0),
        ("000002.SZ", "20260914", 20.0, 21.0, 19.5, 20.5, 2000.0, 40000.0),
        ("600000.SH", "20260914", 30.0, 31.0, 29.5, 30.5, 3000.0, 90000.0),
        ("FACTOR", "000001.SZ", 2.0),
        ("FACTOR", "000002.SZ", 1.0),
        ("FACTOR", "600000.SH", 1.0),
    ],
    "20260915": [
        ("000001.SZ", "20260915", 11.0, 12.0, 10.5, 11.5, 1100.0, 12000.0),
        ("000002.SZ", "20260915", 21.0, 22.0, 20.5, 21.5, 2100.0, 43000.0),
        ("600000.SH", "20260915", 31.0, 32.0, 30.5, 31.5, 3100.0, 96000.0),
        ("FACTOR", "000001.SZ", 2.0),
        ("FACTOR", "000002.SZ", 1.0),
        ("FACTOR", "600000.SH", 1.0),
    ],
}

SESSIONS = pd.DatetimeIndex(pd.to_datetime(["2026-09-14", "2026-09-15"]))


def test_batch_prefetch_writes_raw_factor_and_qfq_caches(tmp_path):
    store = BarStore(tmp_path)
    stats = batch_prefetch_bars(
        _stocks("000001.SZ", "000002.SZ"), _provider(DAYS), store,
        start="2026-09-14", end="2026-09-15", trade_dates=SESSIONS,
        progress=False,
    )
    assert stats.symbols_written == 2
    assert stats.api_calls == 4  # two sessions x two endpoints
    assert not stats.failures

    instrument = Instrument("000001.SZ")
    # qfq latest day matches raw close (qfq base = latest session)
    qfq = store.load(instrument, "2026-09-14", "2026-09-15")
    assert qfq["close"].tolist() == pytest.approx([10.5, 11.5])
    # factor 2.0 on both days rescales the earlier day by 2.0/2.0 == 1.0
    raw = store.load(instrument, "2026-09-14", "2026-09-15", adjust="")
    assert raw["close"].tolist() == pytest.approx([10.5, 11.5])
    metadata = json.loads(store.path(instrument, "qfq").with_suffix(".json").read_text())
    assert metadata["derivation"] == DERIVATION_MARKER


def test_batch_prefetch_skips_complete_symbols_on_second_run(tmp_path):
    store = BarStore(tmp_path)
    common = dict(
        start="2026-09-14", end="2026-09-15", trade_dates=SESSIONS, progress=False
    )
    batch_prefetch_bars(_stocks("000001.SZ"), _provider(DAYS), store, **common)
    stats = batch_prefetch_bars(_stocks("000001.SZ"), _provider(DAYS), store, **common)
    assert stats.symbols_complete == 1
    assert stats.symbols_written == 0
    assert stats.api_calls == 0


def test_batch_prefetch_extends_existing_caches(tmp_path):
    store = BarStore(tmp_path)
    batch_prefetch_bars(
        _stocks("000001.SZ"), _provider(DAYS), store,
        start="2026-09-14", end="2026-09-15", trade_dates=SESSIONS, progress=False,
    )
    extra_days = dict(DAYS)
    extra_days["20260916"] = [
        ("000001.SZ", "20260916", 12.0, 13.0, 11.5, 12.5, 1200.0, 14000.0),
        ("FACTOR", "000001.SZ", 4.0),
    ]
    sessions = SESSIONS.append(pd.DatetimeIndex(pd.to_datetime(["2026-09-16"])))
    stats = batch_prefetch_bars(
        _stocks("000001.SZ"), _provider(extra_days), store,
        start="2026-09-14", end="2026-09-16", trade_dates=sessions, progress=False,
    )
    assert stats.symbols_written == 1
    # Whole history is recomputed with the new base factor (4.0 on 09-16)
    qfq = store.load(Instrument("000001.SZ"), "2026-09-14", "2026-09-16")
    assert qfq["close"].tolist() == pytest.approx([5.25, 5.75, 12.5])


def test_batch_prefetch_leaves_conflicting_cache_untouched(tmp_path):
    store = BarStore(tmp_path)
    instrument = Instrument("000001.SZ")
    # Pre-existing per-symbol cache: both early sessions covered (so returns
    # are comparable) but the last session missing (so it enters the batch
    # path), and its returns contradict the batch data.
    store.save(instrument, pd.DataFrame({
        "date": pd.to_datetime(["2026-09-14", "2026-09-15"]),
        "open": [10.0, 10.0], "high": [10.5, 10.5], "low": [9.5, 9.5],
        "close": [10.0, 10.1], "volume": [1000.0, 1000.0], "amount": [1e5, 1e5],
    }), adjust="qfq", source="fixture")
    extra_days = dict(DAYS)
    extra_days["20260916"] = [
        ("000001.SZ", "20260916", 12.0, 13.0, 11.5, 12.5, 1200.0, 14000.0),
        ("FACTOR", "000001.SZ", 4.0),
    ]
    sessions = SESSIONS.append(pd.DatetimeIndex(pd.to_datetime(["2026-09-16"])))
    stats = batch_prefetch_bars(
        _stocks("000001.SZ"), _provider(extra_days), store,
        start="2026-09-14", end="2026-09-16", trade_dates=sessions, progress=False,
    )
    assert stats.symbols_conflict == 1
    assert stats.symbols_written == 0
    kept = store.load(instrument, "1900-01-01", "2100-01-01")
    assert kept["close"].tolist() == pytest.approx([10.0, 10.1])
