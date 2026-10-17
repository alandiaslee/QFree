import pandas as pd

path = "outputs/batch_20260915_50.csv"
df = pd.read_csv(path)

print("有效快照数：", len(df))
print("技术信号数：", int(df["startup_signal"].sum()))
print("最终入选数：", int(df["selected"].sum()))

print("\n入选结果：")
print(
    df[df["selected"]][
        [
            "ts_code",
            "name",
            "industry",
            "quality_score",
            "technical_score",
            "total_score",
            "quality_missing_count",
            "financial_available_date",
        ]
    ].sort_values("total_score", ascending=False).to_string(index=False)
)

print("\n质量分统计：")
print(df["quality_score"].describe().to_string())

print("\n缺失因子分布：")
print(df["quality_missing_count"].value_counts().sort_index().to_string())

print("\n基本面可用日期最大值：")
print(df["financial_available_date"].max())
