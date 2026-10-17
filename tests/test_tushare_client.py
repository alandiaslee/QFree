from types import SimpleNamespace

import pandas as pd

from qfree.data.models import Instrument
from qfree.data.tushare_client import TushareProvider, to_ts_code


def bars():
    return pd.DataFrame({
        "trade_date": ["20260915", "20260916"],
        "open": [10.0, 10.5], "high": [10.5, 11.0], "low": [9.8, 10.2],
        "close": [10.3, 10.8], "vol": [100, 200], "amount": [1000, 2200],
    })


def test_ts_code_mapping():
    assert to_ts_code("002487") == "002487.SZ"
    assert to_ts_code("600000") == "600000.SH"


def test_adjusted_call_uses_module_level_pro_bar():
    calls = []

    def pro_bar(**kwargs):
        calls.append(kwargs)
        return bars()

    fake_ts = SimpleNamespace(pro_bar=pro_bar)
    fake_pro = SimpleNamespace()
    result = TushareProvider("token", _pro=fake_pro, _ts=fake_ts).get_daily_bars(
        Instrument("002487"), "2026-09-15", "2026-09-16", "qfq"
    )
    assert len(result) == 2
    assert calls[0]["api"] is fake_pro
    assert calls[0]["ts_code"] == "002487.SZ"
    assert calls[0]["adj"] == "qfq"


def test_raw_call_uses_pro_daily():
    fake_pro = SimpleNamespace(daily=lambda **kwargs: bars())
    result = TushareProvider("token", _pro=fake_pro, _ts=SimpleNamespace()).get_daily_bars(
        Instrument("002487"), "2026-09-15", "2026-09-16", ""
    )
    assert list(result["date"].dt.strftime("%Y-%m-%d")) == ["2026-09-15", "2026-09-16"]


def test_request_timeout_is_forwarded_to_pro_api(monkeypatch):
    calls = []

    def fake_pro_api(token, **kwargs):
        calls.append((token, kwargs))
        return SimpleNamespace()

    fake_ts = SimpleNamespace(pro_api=fake_pro_api)
    monkeypatch.setattr("tushare.pro_api", fake_pro_api)
    TushareProvider("token", _ts=None, _pro=None, request_timeout=7.5, api_url="https://example.com")
    assert calls[0][0] == "token"
    assert calls[0][1]["timeout"] == 7.5
