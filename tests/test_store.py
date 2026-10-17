import pandas as pd
import pytest

from qfree.data.models import Instrument
from qfree.data.store import BarStore


def _bars(start, periods):
    close = pd.Series(range(10, 10 + periods), dtype=float)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=periods),
        "open": close, "high": close + 1, "low": close - .5,
        "close": close, "volume": 1000.0, "amount": 100000.0,
    })


def test_csv_cache_round_trip(tmp_path):
    store = BarStore(tmp_path)
    path = store.save(Instrument("002487"), _bars("2026-09-15", 2), adjust="qfq", source="fixture")
    assert path.suffix == ".csv"
    loaded = store.load(Instrument("002487"), "2026-09-15", "2026-09-16")
    assert len(loaded) == 2


def test_load_can_require_complete_coverage(tmp_path):
    store = BarStore(tmp_path)
    store.save(Instrument("002487"), _bars("2026-09-15", 2), adjust="qfq", source="fixture")
    with pytest.raises(FileNotFoundError):
        store.load(Instrument("002487"), "2026-09-14", "2026-09-16", require_complete=True)


def test_save_merges_identical_overlap_and_preserves_snapshot(tmp_path):
    store = BarStore(tmp_path)
    instrument = Instrument("002487")
    store.save(instrument, _bars("2026-09-15", 5), adjust="qfq", source="fixture")
    update = _bars("2026-09-15", 7).iloc[4:].reset_index(drop=True)
    store.save(instrument, update, adjust="qfq", source="fixture")
    loaded = store.load(instrument, "2026-09-15", "2026-09-23")
    assert len(loaded) == 7


def test_save_rejects_revised_overlapping_bars(tmp_path):
    store = BarStore(tmp_path)
    instrument = Instrument("002487")
    store.save(instrument, _bars("2026-09-15", 5), adjust="qfq", source="fixture")
    revised = _bars("2026-09-15", 6).iloc[4:].reset_index(drop=True)
    revised.loc[0, ["close", "high"]] = 999.0
    with pytest.raises(RuntimeError, match="revision detected"):
        store.save(instrument, revised, adjust="qfq", source="fixture")
    loaded = store.load(instrument, "2026-09-15", "2026-09-19")
    assert loaded.loc[loaded["date"].eq(pd.Timestamp("2026-09-19")), "close"].iloc[0] == 14.0
