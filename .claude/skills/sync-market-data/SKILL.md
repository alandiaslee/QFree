---
name: sync-market-data
description: 同步 Tushare 全市场日线数据到指定交易日（默认最新），并修复除权导致的 qfq 复权缓存冲突。当用户说“同步数据/拉数据/更新数据到某日期”时使用。
---

同步 A 股全市场日线缓存（前复权），使用交易日批量接口（防 Tushare 6000 行/请求截断），随后自动修复除权除息后的 qfq 冲突缓存。约 5~8 分钟（12 进程并行）。

## 执行步骤

1. 运行同步（`target_date` 可省略，默认拉取 Tushare 最新交易日；通常在收盘后约 18:00 后数据才完整）：

   ```bash
   .venv/bin/python scripts/sync_bars.py [YYYY-MM-DD]
   ```

2. 修复 qfq 冲突（除非第 1 步报 0 conflicts，否则必跑；幂等，约 1 分钟）：

   ```bash
   .venv/bin/python scripts/fix_qfq_conflicts.py
   ```

3. 确认指数缓存已更新（第 1 步 `sync_bars.py` 会自动更新 `data/index/000300_SH.csv`——v3 策略的 regime 门依赖它；输出末尾应有 `index cache 000300.SH: +N sessions`）。若该步告警失败，单独补跑：

   ```bash
   .venv/bin/python -c "import sys; sys.path.insert(0,'src'); from qfree.config import Settings; from qfree.data.tushare_client import TushareProvider; from qfree.data.index_cache import update_index; p=TushareProvider(Settings.from_env().require_token(), api_url='https://t.xiaodefa.top/'); print(update_index(p,'data'))"
   ```

4. 报告：写入/冲突/失败数量。失败通常为 4~5 只已退市代码（601313、000022、000043、301569 等），属于已知不可修复项，不必重试。

## 注意

- **token 安全**：Tushare token 在环境变量里，`Settings.from_env()` 读取，绝不打印或显示。
- 网关：`https://t.xiaodefa.top/`（付费版授权网关）。
- 不要逐股拉取或用符号批次接口跨长日期范围（会触发 6000 行截断，数据静默缺失）；必须走 `batch_prefetch_bars` 交易日批量路径（脚本内部已保证）。
- 若同步卡住（CPU 持续 50%+ 是正常并行写盘，不是死锁）；单日全市场预期 5~8 分钟，明显超时先 `ps` 检查进程数再排查，不要傻等。
- 同步完成后用户通常会接着要选股，直接接 `/daily-pick`。
