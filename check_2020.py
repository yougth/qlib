import qlib
from qlib.constant import REG_CN
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
from qlib.data import D
import pandas as pd, numpy as np

fcf = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv")
profit = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv")
fcf["code"] = fcf["code"].astype(str).str.zfill(6)
profit["code"] = profit["code"].astype(str).str.zfill(6)

# W0b pool: backtest 2020
fcf_f = fcf[fcf["year"].between(2009, 2018)]
fcf_p = fcf_f.groupby("code").filter(lambda g: len(g) >= 8 and (g["fcf"] > 0).all())
pf_f = profit[profit["year"].between(2016, 2018)]
pf_p = pf_f.groupby("code").filter(lambda g: len(g) >= 2 and (g["net_profit"] > 0).all())
codes = set(fcf_p["code"]) & set(pf_p["code"])
qlib_codes = [f"SH{c}" if c.startswith("6") else f"SZ{c}" for c in codes]

prices = D.features(qlib_codes, ["$close"], start_time="2020-01-01", end_time="2020-12-31")
prices = prices.reset_index()
prices.columns = ["inst", "dt", "close"]
rets = prices.groupby("inst").apply(lambda g: (g.iloc[-1]["close"]/g.iloc[0]["close"] - 1) * 100).sort_values(ascending=False)
print(f"2020 pool: {len(rets)} stocks")
print(f"Top 10: {dict(rets.head(10).round(1))}")
print(f"Bottom 5: {dict(rets.tail(5).round(1))}")
print(f"Median: {rets.median():.1f}%, Mean: {rets.mean():.1f}%")
print(f">100%: {(rets > 100).sum()}, >200%: {(rets > 200).sum()}, >500%: {(rets > 500).sum()}")
