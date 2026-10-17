"""Generate the next-day buy order sheet from a daily-pick selection.

Usage:
    .venv/bin/python scripts/make_order_sheet.py 2026-10-16 [--capital 500000]

Reads outputs/daily_<date>_selected.csv (gate-pass picks, top 10 by
base_total_score) and writes outputs/daily_<date>_orders.csv plus a console
plan. Execution rules (validated in scripts/gap_entry_analysis.py on 713
final-config signals):

  - place a pre-open limit order at signal_close * 1.02; it fills at the open
    price iff the open gap <= +2% (matching the validated skip-chased-gaps
    rule), else cancel at 9:30;
  - also skip if the open gaps down <= -5% (overnight crash, usually bad news);
  - cash from skipped slots is redistributed equally across the remaining
    valid picks of the same date;
  - after each fill, record actual entry price; profit target = entry*1.05
    (sell next open after a close >= target), max holding 10 sessions,
    optional disaster line close <= entry*0.88.

Share amounts are indicative, rounded to 100-share lots using the signal
close; final amounts are set off the actual open price.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

GAP_SKIP_UP = 0.02
GAP_SKIP_DOWN = -0.05
TARGET = 0.05
DISASTER = 0.12
HOLDING = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="signal date YYYY-MM-DD")
    ap.add_argument("--capital", type=float, default=500_000, help="总资金 (元)")
    ap.add_argument("--selected", default=None)
    args = ap.parse_args()

    path = Path(args.selected or f"outputs/daily_{args.date}_selected.csv")
    if not path.exists():
        raise SystemExit(f"missing {path} — run daily-scan first")
    sel = pd.read_csv(path)
    if sel.empty:
        print(f"{args.date}: 无选中股票（市场门未通过或无数值），无需订单。")
        return 0
    sel = sel.sort_values("base_total_score", ascending=False).head(10)
    n = len(sel)
    per = args.capital / n

    rows = []
    for i, r in enumerate(sel.itertuples(), 1):
        close = float(r.close)
        limit_price = round(close * (1 + GAP_SKIP_UP), 2)  # 竞价限价
        ref_shares = int(per / limit_price // 100) * 100
        rows.append({
            "rank": i, "ts_code": r.ts_code, "name": r.name,
            "industry": getattr(r, "industry", ""),
            "score": round(float(r.base_total_score), 1),
            "signal_close": close,
            "limit_price": limit_price,
            "order": f"9:15 挂限价单 {limit_price} 元 × {ref_shares} 股",
            "weight_pct": round(100 / n, 1),
            "amount_ref": round(per, 0),
            "profit_target_price": round(limit_price * (1 + TARGET), 2),
            "disaster_line": round(limit_price * (1 - DISASTER), 2),
        })
    sheet = pd.DataFrame(rows)
    out = Path(f"outputs/daily_{args.date}_orders.csv")
    sheet.to_csv(out, index=False)

    print(f"\n===== {args.date} 次日买入操作清单（本金 {args.capital:,.0f} 元，top{n} 等权）=====")
    print(sheet.drop(columns=["order"]).to_string(index=False))
    print("""
执行规则（前一晚挂单，全部条件单一次挂好）：
1. 每只挂「集合竞价限价单」：限价 = 信号收盘价 × 1.02（见表 limit_price），
   数量 = 表内股数（按限价计算的 100 股整手）。
2. 9:25 竞价撮合：开盘价 ≤ 限价即以开盘价成交；高开 > +2% 不成交，9:30 前撤单
   （高开 +2~5% 桶回测胜率仅 57%、均值 -0.43%，追入是负期望）。
3. 跳过的仓位资金收盘后均分给当日其余成交标的（或留货基），次日不再追。
4. 低开 > -5% 才成交；若竞价显示低开 ≤ -5%（隔夜闪崩），手动撤单放弃该股。
5. 成交后回填实际买入价：止盈价 = 买入价 × 1.05；持有期内收盘 >= 止盈价 →
   次日开盘卖出；满 10 个交易日 → 到期卖出；
   可选保险线：收盘 <= 买入价 × 0.88 → 次日开盘卖出。
清单已写入 """ + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
