import pandas as pd

df = pd.read_csv("outputs/batch_20260915_small.csv")

columns = [
      "ts_code",
      "name",
      "industry",
      "quality_score",
      "technical_score",
      "total_score",
      "startup_signal",
      "selected",
      "quality_missing_count",
      "financial_available_date",
]

print(df[[c for c in columns if c in df.columns]]
        .sort_values("total_score", ascending=False)
        .to_string(index=False))
