from types import SimpleNamespace

import pandas as pd

from qfree.data.tushare_client import TushareProvider


def test_trade_cal_returns_sorted_open_sessions():
    fake = SimpleNamespace(trade_cal=lambda **kwargs: pd.DataFrame({
        "cal_date": ["20260904", "20260902"], "is_open": [1, 1],
    }))
    result = TushareProvider("token", _pro=fake, _ts=SimpleNamespace()).get_trade_dates(
        "2026-09-01", "2026-09-04"
    )
    assert list(result.strftime("%Y-%m-%d")) == ["2026-09-02", "2026-09-04"]


def test_financial_snapshot_maps_common_fields():
    fake = SimpleNamespace(fina_indicator=lambda **kwargs: pd.DataFrame({
        "ts_code": ["002487.SZ"],
        "ann_date": ["2026-04-30"],
        "f_ann_date": [None],
        "end_date": ["2025-12-31"],
        "roe_avg": [12.0],
        "netprofit_yoy": [15.0],
        "debt_to_assets": [40.0],
        "grossprofit_margin": [25.0],
        "ocf_to_profit": [90.0],
    }))
    result = TushareProvider("token", _pro=fake, _ts=SimpleNamespace()).financial_snapshot("002487.SZ")
    assert result.loc[0, "roe"] == 12.0
    assert result.loc[0, "ocf_to_profit"] == 90.0


def test_derived_ocf_uses_later_statement_available_date():
    fake = SimpleNamespace(
        fina_indicator=lambda **kwargs: pd.DataFrame({
            "ts_code": ["002487.SZ"], "ann_date": ["2026-04-01"],
            "end_date": ["2025-12-31"], "roe": [12.0],
        }),
        cashflow=lambda **kwargs: pd.DataFrame({
            "end_date": ["2025-12-31"], "n_cashflow_act": [80.0],
            "ann_date": ["2026-04-01"],
        }),
        income=lambda **kwargs: pd.DataFrame({
            "end_date": ["2025-12-31"], "n_income": [100.0],
            "ann_date": ["2026-05-01"],
        }),
    )
    result = TushareProvider("token", _pro=fake, _ts=SimpleNamespace()).financial_snapshot(
        "002487.SZ"
    )
    assert result.loc[0, "ocf_to_profit"] == 80.0
    assert result.loc[0, "available_date"] == pd.Timestamp("2026-05-01")
