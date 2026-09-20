#!/usr/bin/env python3
"""fetch_profit_patch —— 补齐 profit_cache_pit 2012-2015 年全市场净利润缺口
================================================================================
背景: 共享 profit_cache_pit.csv 2012-2015 年仅覆盖约250只 (上游 fetch_profit.py
的 TARGET_YEARS 从2016起才全市场), 而 fcf_cache_pit 同期是全市场覆盖
(2013年4017只) —— 交集太小导致 new_quant 质量池在 2019-2021H1 退化成个位数。

数据源: 东财业绩报表 stock_yjbb_em(date="YYYY1231") —— 一次请求返回全市场该
年报期归母净利润。已做两项实地验证:
  1. 金标准: 茅台 20131231 报表行净利润 151.37亿 = 真实2013年报归母净利润;
  2. 口径: 与主缓存 2016 年 4584 只重叠段对比, 相对误差中位数 0, 100% <0.5%
     (同为归母净利润, 单位元)。

已知坑 (实测): 东财对"该报告期无数据的股票"会塞入错位的历史报表
(如中芯国际 20131231 报表行实为2018年报数据), 故需三重过滤:
  1. A股代码前缀 (60/68/00/30);
  2. code ∈ fcf_cache_pit 当年有记录的代码集 —— fcf 早年为全市场覆盖,
     等价于"该年已上市且年报可得"名单, 2013年后上市的错位行自动排除;
  3. 最新公告日期 ∈ [Y+1-01-01, Y+2-12-31] —— 兜底剔除塞入的其他报告期。

产出: new_quant/data/profit_patch_pit.csv (Tab分隔: code/year/net_profit)
用法: /usr/bin/python3 tools/fetch_profit_patch.py   (按年幂等, 可中断续跑)
"""
import os
import sys
import time

import pandas as pd

NEW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(NEW_DIR))
FCF_CACHE = os.path.join(ROOT, "fcf_cache_pit.csv")
PROFIT_CACHE = os.path.join(ROOT, "profit_cache_pit.csv")
OUT = os.path.join(NEW_DIR, "data", "profit_patch_pit.csv")
YEARS = [2012, 2013, 2014, 2015]
A_PREFIX = ("60", "68", "00", "30")

# 工作区根目录含 akshare 源码包, 从根目录启动时会遮蔽正式安装版, 先剔除
sys.path = [p for p in sys.path
            if os.path.abspath(p or ".") != os.path.abspath(ROOT)]
import akshare as ak  # noqa: E402


def fetch_year(y, fcf_codes):
    """抓某年报期全市场业绩报表, 三重过滤后返回 code/year/net_profit"""
    df = ak.stock_yjbb_em(date=f"{y}1231")
    n_raw = len(df)
    df["code"] = df["股票代码"].astype(str).str.zfill(6)
    df["np"] = pd.to_numeric(df["净利润-净利润"], errors="coerce")
    df["ann"] = pd.to_datetime(df["最新公告日期"], errors="coerce")
    keep = (df["code"].str[:2].isin(A_PREFIX) & df["code"].isin(fcf_codes)
            & df["np"].notna()
            & df["ann"].between(f"{y + 1}-01-01", f"{y + 2}-12-31"))
    out = df.loc[keep, ["code", "np"]].copy()
    out["year"] = y
    return out[["code", "year", "np"]], n_raw - len(out)


def main():
    fcf = pd.read_csv(FCF_CACHE, sep="\t", dtype={"code": str})
    fcf["code"] = fcf["code"].str.zfill(6)
    fcf["year"] = pd.to_numeric(fcf["year"], errors="coerce")
    prof = pd.read_csv(PROFIT_CACHE, sep="\t", dtype={"code": str})
    prof["code"] = prof["code"].str.zfill(6)
    prof["year"] = pd.to_numeric(prof["year"], errors="coerce")

    parts, done = [], set()
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, sep="\t", dtype={"code": str})
        old["year"] = pd.to_numeric(old["year"], errors="coerce")
        parts.append(old)
        done = set(old["year"].dropna().astype(int).unique())
        print(f"[CACHE] 已有补丁 {len(old)} 条, 覆盖年份 {sorted(done)}", flush=True)

    for y in YEARS:
        if y in done:
            print(f"[{y}] 已缓存, 跳过", flush=True)
            continue
        fcf_codes = set(fcf.loc[fcf["year"] == y, "code"])
        rows, dropped = fetch_year(y, fcf_codes)
        # 金标准复核: 与主缓存该年已有的~250只对比
        p = prof[prof["year"] == y][["code", "net_profit"]].drop_duplicates("code")
        m = p.merge(rows, on="code", how="inner")
        msg = (f"[{y}] fcf名单 {len(fcf_codes)} | yjbb返回 {len(rows) + dropped} | "
               f"过滤后 {len(rows)} (剔错位/非名单 {dropped})")
        if len(m):
            rel = ((m["np"] - m["net_profit"]).abs()
                   / m["net_profit"].abs().clip(lower=1))
            msg += f" | 与主缓存重叠 {len(m)} 只 一致率 {(rel < 0.005).mean():.1%}"
        print(msg, flush=True)
        parts.append(rows.rename(columns={"np": "net_profit"}))
        pd.concat(parts, ignore_index=True).drop_duplicates(
            ["code", "year"]).to_csv(OUT, sep="\t", index=False)
        time.sleep(2)

    final = pd.read_csv(OUT, sep="\t", dtype={"code": str})
    print(f"\n[+] 补丁完成: {OUT} | {len(final)} 条 | "
          f"{final['year'].nunique()} 个年份 {sorted(final['year'].unique())}",
          flush=True)


if __name__ == "__main__":
    main()
