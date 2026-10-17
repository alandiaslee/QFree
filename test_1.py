import pandas as pd

df = pd.read_csv("data/financials/002487_SZ.csv")

print(df[
      [
          "available_date",
          "end_date",
          "roe",
          "netprofit_yoy",
          "ocf_to_profit",
          "debt_to_assets",
          "grossprofit_margin",
      ]
].tail(10).to_string(index=False))
