import pandas as pd

from qfree.universe import filter_universe


def test_filter_universe_removes_st_new_and_delisted():
    stocks = pd.DataFrame({
        "ts_code": ["002487.SZ", "000001.SZ", "600000.SH", "300001.SZ"],
        "name": ["大金重工", "*ST风险", "老股票", "新股票"],
        "list_date": ["2010-01-01", "2010-01-01", "2010-01-01", "2026-06-01"],
        "exchange": ["SZSE", "SZSE", "SSE", "SZSE"],
        "list_status": ["L", "L", "D", "L"],
    })
    result = filter_universe(stocks, as_of="2026-10-03")
    assert list(result["ts_code"]) == ["002487.SZ"]
