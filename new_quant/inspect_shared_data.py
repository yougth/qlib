#!/usr/bin/env python3
"""inspect_shared_data —— 只读盘点共享数据覆盖 (不写任何文件, 只打印摘要)

用途: 为 new_quant 十类新策略回测评估可复用数据:
- 行情/退市覆盖: qlib/data_cache/qlib_cn_tencent (bin)
- 财务PIT缓存: 根目录 fcf/profit/pershare_cache_pit.csv
- 估值/行业/财务健康/退市财务: data_cache 下 parquet/csv
"""
import os
import glob
import pandas as pd

ROOT = "/Users/11164591/Documents/Qoder目录"
DC = os.path.join(ROOT, "qlib", "data_cache")
BIN = os.path.join(DC, "qlib_cn_tencent")


def sec(t):
    print("\n" + "=" * 12, t, "=" * 12, flush=True)


# 1) bin instruments: 股票 vs 指数, 退市覆盖, 字段列表
ins = pd.read_csv(os.path.join(BIN, "instruments", "all.txt"), sep="\t",
                  header=None, names=["code", "start", "end"])
stocks = ins[~ins.code.str.match(r"^(SH000|SZ39|hk)")]
sec("bin instruments")
print("总行数", len(ins), "| 股票数", len(stocks),
      "| 指数", len(ins) - len(stocks))
ended = stocks[stocks.end < "2026-08-01"]
print("end<2026-08 的股票(疑似退市/停更):", len(ended))
if len(ended):
    print(ended.tail(5).to_string(index=False))
one_feat = os.path.join(BIN, "features", stocks.code.iloc[0])
print("features 字段样本:", sorted(os.listdir(one_feat))[:12])

# 2) 财务PIT缓存覆盖 (股票数/年份)
fcf = pd.read_csv(os.path.join(ROOT, "fcf_cache_pit.csv"), sep="\t")
prof = pd.read_csv(os.path.join(ROOT, "profit_cache_pit.csv"), sep="\t")
ps = pd.read_csv(os.path.join(ROOT, "pershare_cache_pit.csv"), sep="\t")
sec("财务缓存")
print("fcf   : 行", len(fcf), "唯一代码", fcf.code.nunique(),
      "年份", fcf.year.min(), "~", fcf.year.max())
print("profit: 行", len(prof), "唯一代码", prof.code.nunique(),
      "年份", prof.year.min(), "~", prof.year.max())
print("pershare(季度): 行", len(ps), "唯一代码", ps.code.nunique(),
      "报告期", ps.report_date.min(), "~", ps.report_date.max())

# 3) valuation_cache 结构 (>20MB, 用 pandas 读)
val = pd.read_csv(os.path.join(ROOT, "valuation_cache.csv"), sep="\t")
sec("valuation_cache")
print("shape", val.shape, "| 前12列", list(val.columns[:12]))
dcol = [c for c in val.columns if c.lower() in ("date", "datetime", "day")]
if dcol:
    print("日期范围", val[dcol[0]].min(), "~", val[dcol[0]].max())

# 4) parquet 资产: 退市财务 / 财务健康 / 行业
for f in ["delist_financials.parquet", "fin_health_pit.parquet"]:
    p = os.path.join(DC, f)
    if not os.path.exists(p):
        sec(f + " 缺失")
        continue
    df = pd.read_parquet(p)
    sec(f)
    print("shape", df.shape, "| cols", list(df.columns))
    cc = [c for c in ("code", "symbol", "ts_code") if c in df.columns]
    if cc:
        print("唯一代码", df[cc[0]].nunique())
    yc = [c for c in ("year", "report_date", "ann_date") if c in df.columns]
    if yc:
        print(yc[0], "范围", df[yc[0]].min(), "~", df[yc[0]].max())
    print(df.head(3).to_string(index=False))

ind = pd.read_csv(os.path.join(DC, "industry_map.csv"), sep="\t")
sec("industry_map")
print("shape", ind.shape, "| cols", list(ind.columns))
print(ind.head(3).to_string(index=False))

# 5) 财务代码 vs bin 股票交叉覆盖
scodes = set(stocks.code.str[2:])
fcodes = set(fcf.code.astype(str).str.zfill(6))
pcodes = set(prof.code.astype(str).str.zfill(6))
sec("覆盖交叉 (存活偏差关键项)")
print("fcf 代码不在 bin:", len(fcodes - scodes),
      "| bin 股票缺 fcf:", len(scodes - fcodes))
print("profit 代码不在 bin:", len(pcodes - scodes),
      "| bin 股票缺 profit:", len(scodes - pcodes))

# 6) tencent parquet 样本口径
fs = sorted(glob.glob(os.path.join(DC, "tencent", "*.parquet")))
if fs:
    d = pd.read_parquet(fs[0])
    sec("tencent样本 " + os.path.basename(fs[0]))
    print("cols", list(d.columns), "| 行", len(d),
          "| 日期", d.date.min(), "~", d.date.max())

# 7) akshare 可用性 (新数据抓取候选)
try:
    import akshare as ak
    sec("akshare 可用")
    print("版本", ak.__version__)
except Exception as e:
    sec("akshare 不可用")
    print(repr(e))
