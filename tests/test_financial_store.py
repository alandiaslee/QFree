import pandas as pd

from qfree.data.financial_store import FinancialStore, normalize_financials


def test_as_of_ignores_future_announcement(tmp_path):
    frame = pd.DataFrame({
        "ts_code": ["002487.SZ", "002487.SZ"],
        "ann_date": ["2026-04-30", "2026-10-01"],
        "f_ann_date": [None, None],
        "end_date": ["2025-12-31", "2026-06-30"],
        "roe": [10.0, 99.0],
    })
    store = FinancialStore(tmp_path)
    store.save(frame, source="fixture")
    result = store.as_of("002487.SZ", "2026-09-15")
    assert len(result) == 1
    assert result.iloc[0]["roe"] == 10.0


def test_normalize_requires_available_date():
    frame = pd.DataFrame({"ts_code": ["002487.SZ"], "end_date": ["2025-12-31"], "roe": [10]})
    assert normalize_financials(frame).empty


def test_normalize_rejects_implausible_margin_and_deduplicates_reports():
    frame = pd.DataFrame({
        "ts_code": ["002487.SZ", "002487.SZ"],
        "ann_date": ["2026-04-28", "2026-04-28"],
        "end_date": ["2026-03-31", "2026-03-31"],
        "grossprofit_margin": [39.19, 39.19],
        "gross_margin": [747342300.0, 747342300.0],
    })
    result = normalize_financials(frame)
    assert len(result) == 1
    assert result.iloc[0]["grossprofit_margin"] == 39.19


def test_save_merges_incremental_financial_history(tmp_path):
    store = FinancialStore(tmp_path)
    first = pd.DataFrame({
        "ts_code": ["002487.SZ"], "ann_date": ["2026-04-30"],
        "end_date": ["2025-12-31"], "roe": [10.0],
    })
    second = pd.DataFrame({
        "ts_code": ["002487.SZ"], "ann_date": ["2026-08-30"],
        "end_date": ["2026-06-30"], "roe": [12.0],
    })
    store.save(first, source="fixture")
    store.save(second, source="fixture")
    assert len(store.load("002487.SZ")) == 2


def test_normalize_handles_duplicate_api_columns():
    frame = pd.DataFrame(
        [["002487.SZ", "2026-04-30", 10.0, 11.0]],
        columns=["ts_code", "ann_date", "roe_avg", "roe_avg"],
    )
    result = normalize_financials(frame)
    assert result.iloc[0]["roe"] == 10.0
