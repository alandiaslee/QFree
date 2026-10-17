# QFree

A local-first A-share research screener for quality companies with bottom-volume-start signals.

## Status

The first version produces research candidate lists and point-in-time event reports only. It does **not** place orders and is not investment advice. A single selected stock or a single observation date is a case study, not evidence of stable performance; use several frozen observation dates before drawing conclusions.

The initial universe covers Shanghai and Shenzhen A shares, including main board, ChiNext, and STAR Market; Beijing shares are excluded for now. Parameters are research baselines, not guarantees.

## Security

The Tushare token must be supplied through `TUSHARE_TOKEN`; it is never stored in source code or logs. The token pasted in the original conversation is considered exposed: revoke/rotate it in Tushare before use, then export the replacement:

```bash
export TUSHARE_TOKEN='your-rotated-token'
```

The optional `--api-url` exists for explicitly configured gateways. It is not enabled by default. Use only a gateway you are authorized to use and understand its data/privacy implications.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[tushare,dev]'
export TUSHARE_TOKEN='...'
qfree fetch 002487 --start 2025-01-01 --adjust qfq
qfree scan 002487
```

`002487` is used as a reference symbol for research around the 2026-09-15 example. The date is treated as a hypothesis to validate against returned trading data; the current research date is 2026-10-05, but data availability still depends on the provider. Use `qfree walk-forward --as-of DATE --as-of DATE` to run multiple point-in-time scans and retain candidates, failures, coverage, event rows, and separate per-date/pooled event summaries.

Tushare `vol` is typically in hands and `amount` in thousand yuan. QFree preserves those provider units in the canonical bars and records the adjustment/source in the sidecar metadata. Adjusted prices use `ts.pro_bar(api=pro, adj='qfq'/'hfq')`; raw prices use `pro.daily(...)`. Daily bars are cached as CSV to avoid pandas/PyArrow version-specific Parquet read errors. Cache ranges must be checked before historical research: a file existing does not prove it covers the requested interval. When available, Tushare `trade_cal` sessions are used to detect missing middle sessions rather than treating calendar-day endpoints as sufficient. QFree's qfq series is a research convenience and may be revised after later corporate actions; overlapping cached bars are now revision-checked and sidecars include a content hash, but strict historical replay still requires frozen provider snapshots or a separately versioned raw execution-price dataset. When provider status and limit fields are available, event evaluation marks suspended or limit-bound entry/exit bars as non-tradable; absent fields are not inferred.

`qfree batch-scan` writes three artifacts: candidates, failures, and coverage. Failures are not equivalent to no signal. `qfree walk-forward` additionally writes event rows plus observation-date and pooled summaries; fixed 5/10/20-session returns are counted only when mature. `qfree evaluate-event` supports `fixed_open` and the separate `window_end_close` diagnostic; never combine those exit modes in one performance statistic. The current stock universe is based on a current stock-basic snapshot and therefore has survivorship/history limitations. Multi-date runs using this single snapshot print an explicit survivorship warning; use `--universe-by-as-of-dir` with frozen `YYYY-MM-DD.csv` snapshots before treating historical comparisons as point-in-time.

## Tests

```bash
pytest
ruff check src tests
```

Tests must use fixtures/fake clients and must not call Tushare by default. Historical research should use point-in-time financial data keyed by announcement/availability date to avoid look-ahead bias.
