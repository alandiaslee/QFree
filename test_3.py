import pandas as pd

df = pd.read_csv("data/universe.csv")

target = df[df["ts_code"].eq("002487.SZ")]
others = df[~df["ts_code"].eq("002487.SZ")].sample(
    n=49,
    random_state=20260915,
)

sample = pd.concat([target, others]).drop_duplicates("ts_code")
sample.to_csv("data/universe_50.csv", index=False)

print("试验股票数：", len(sample))
print("行业数：", sample["industry"].nunique())
print(sample[["ts_code", "name", "industry"]].to_string(index=False))
