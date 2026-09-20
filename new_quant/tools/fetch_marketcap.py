#!/usr/bin/env python3
"""fetch_marketcap —— 百度个股历史总市值 → new_quant/data/marketcap_baidu.csv
================================================================================
用途: LOWATT 的"低关注"升级为"真小市值"因子 (原用20日均成交额代理)。
来源: ak.stock_zh_valuation_baidu(symbol, indicator='总市值', period='近十年')
  · 实测: 5天采样密度, 覆盖2016~今, 逐点是当时真实市值 (PIT安全, 最长滞后5天)
  · 速度: ~0.7s/只 (含限速0.3s)
范围: 质量池全集 (2017~2025 任一 usable_year 通过5年双正的代码) ∩ tencent行情库
      —— LOWATT 只可能从这些股票里选, 超集无浪费。
幂等: 已抓代码跳过, 可中断续跑。
"""
import glob
import os
import sys
import time

import pandas as pd

NEW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(NEW_DIR))
OUT = os.path.join(NEW_DIR, "data", "marketcap_baidu.csv")

sys.path.insert(0, NEW_DIR)
# 工作区根目录含 akshare 源码包, 剔除避免遮蔽正式安装版
sys.path = [p for p in sys.path
            if os.path.abspath(p or ".") != os.path.abspath(ROOT)]
import akshare as ak  # noqa: E402
from core import config, data  # noqa: E402


def universe():
    """质量池全集 ∩ tencent行情库 (行情侧用文件名快速判定)"""
    fin = data.load_financials()
    syms = set()
    for y in range(2017, 2026):
        win = fin[fin["year"].between(y - 4, y)]
        ok = win.groupby("sym").agg(
            n=("year", "nunique"),
            fcf_ok=("fcf", lambda s: bool(s.notna().all() and (s > 0).all())),
            np_ok=("net_profit", lambda s: bool(s.notna().all() and (s > 0).all())))
        syms |= set(ok.index[(ok["n"] == 5) & ok["fcf_ok"] & ok["np_ok"]])
    have_px = {os.path.splitext(os.path.basename(f))[0].upper()
               for f in glob.glob(os.path.join(config.TENCENT_DIR, "*.parquet"))}
    codes = sorted(s[2:] for s in syms if s in have_px)
    return codes


def fetch_one(code):
    for attempt in range(3):
        try:
            df = ak.stock_zh_valuation_baidu(
                symbol=code, indicator="总市值", period="近十年")
            if df is None or not len(df):
                return None
            df = df.rename(columns={"date": "date", "value": "mv"})
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= "2016-01-01"]
            df["code"] = code
            return df[["code", "date", "mv"]]
        except Exception:
            time.sleep(1.0 + attempt)
    return None


def main():
    codes = universe()
    done = set()
    parts = []
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, dtype={"code": str})
        done = set(old["code"].unique())
        parts = [old]
        print(f"[CACHE] 已有 {len(done)} 只", flush=True)
    todo = [c for c in codes if c not in done]
    print(f"[UNIVERSE] 质量池全集∩行情库: {len(codes)} 只 | 待抓 {len(todo)} 只 "
          f"(预计 {len(todo) * 1.0 / 60:.0f} 分钟)", flush=True)

    t0, rows = time.time(), []
    for i, code in enumerate(todo, 1):
        df = fetch_one(code)
        if df is not None:
            rows.append(df)
        if i % 50 == 0 or i == len(todo):
            if rows:
                parts.append(pd.concat(rows, ignore_index=True))
                rows = []
            pd.concat(parts, ignore_index=True).drop_duplicates(
                ["code", "date"]).to_csv(OUT, index=False)
            print(f"  进度 {i}/{len(todo)} ({i * 100 // len(todo)}%) "
                  f"耗时 {time.time() - t0:.0f}s", flush=True)
        time.sleep(0.3)

    final = pd.read_csv(OUT, dtype={"code": str})
    cov = final[final["date"] >= "2019-01-01"]["code"].nunique()
    print(f"\n[+] 完成: {OUT} | {final['code'].nunique()} 只 | "
          f"2019后有效覆盖 {cov} 只", flush=True)


if __name__ == "__main__":
    main()
