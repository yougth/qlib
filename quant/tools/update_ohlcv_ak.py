#!/usr/bin/env python3
"""
update_ohlcv_ak —— akshare 版行情增量更新 (东财接口)
================================================================================
腾讯 WAF 封了两个主机, 改用 akshare (eastmoney) 做增量追加。
每只股票只抓 hfq 日线 (一个请求), 追加到已有 parquet 尾部。

  · close = hfq 收盘 (收益/特征口径与 fetch_ohlcv 一致)
  · factor = 沿用旧文件尾部 factor (缺新增分红调整, 35 交易日误差 <1%)
  · 指数 (sh000300) 用 ak.stock_zh_index_daily

用法:
    python3 tools/update_ohlcv_ak.py                    # 更新池内全部
    python3 tools/update_ohlcv_ak.py --syms @file.txt   # 指定标的
    python3 tools/update_ohlcv_ak.py --limit 5          # 试点
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

QUANT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QLIB_REPO = os.path.dirname(QUANT_DIR)
OUT_DIR = os.path.join(QLIB_REPO, "data_cache", "tencent")
INDEX_SYMS = ["sh000300", "sh000905", "sh000001"]
TODAY = pd.Timestamp.now().strftime("%Y%m%d")


def fetch_akshare_hfq(code6, start_date, end_date, max_retries=3):
    """akshare 获取 A 股后复权日线"""
    import akshare as ak
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(symbol=code6, period="daily",
                                    start_date=start_date, end_date=end_date, adjust="hfq")
            if df is None or len(df) == 0:
                return None
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            for c in ["open", "close", "high", "low", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["date", "open", "close", "high", "low", "volume"]].dropna(
                subset=["close"]).drop_duplicates("date").sort_values("date")
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def fetch_akshare_index(symbol, start_date, end_date):
    """akshare 获取指数日线"""
    import akshare as ak
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        if df is None or len(df) == 0:
            return None
        df = df.rename(columns={"date": "date", "open": "open", "close": "close",
                                "high": "high", "low": "low", "volume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        sd = pd.Timestamp(start_date)
        ed = pd.Timestamp(end_date)
        df = df[(df["date"] >= sd) & (df["date"] <= ed)]
        for c in ["open", "close", "high", "low", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["date", "open", "close", "high", "low", "volume"]].dropna(
            subset=["close"]).drop_duplicates("date").sort_values("date")
    except Exception:
        return None


def update_one(sym):
    """增量更新一只标的"""
    path = os.path.join(OUT_DIR, f"{sym}.parquet")
    if not os.path.exists(path):
        return sym, f"ERR no parquet"

    try:
        old = pd.read_parquet(path)
    except Exception as e:
        return sym, f"ERR read: {e}"

    last_date = pd.Timestamp(str(old["date"].max())[:10])
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")

    is_index = sym in INDEX_SYMS
    code6 = sym[2:] if not is_index else None

    try:
        if is_index:
            new_data = fetch_akshare_index(sym, start, TODAY)
        else:
            new_data = fetch_akshare_hfq(code6, start, TODAY)

        if new_data is None or len(new_data) == 0:
            return sym, 0  # 无新行 (停牌等)

        # 只保留比旧文件更新的行
        new_data = new_data[new_data["date"] > last_date]
        if len(new_data) == 0:
            return sym, 0

        # factor: 沿用旧文件尾部 (缺新增分红, 误差可接受)
        tail_factor = float(old["factor"].iloc[-1])
        new_data["factor"] = tail_factor
        new_data["symbol"] = sym

        # 非正价检查
        bad = (new_data[["open", "close", "high", "low"]] <= 0).any(axis=1)
        if bad.any():
            if bad.mean() > 0.05:
                return sym, f"ERR NonPositivePrice {int(bad.sum())}/{len(new_data)}行"
            new_data = new_data[~bad]

        if len(new_data) == 0:
            return sym, 0

        out = pd.concat([old, new_data], ignore_index=True)
        out = out.drop_duplicates("date", keep="last").sort_values("date")
        out["symbol"] = sym
        out.to_parquet(path, index=False)
        return sym, len(new_data)
    except Exception as e:
        return sym, f"ERR {type(e).__name__}: {str(e)[:60]}"


def sync_csi300_cache():
    """从 sh000300.parquet 同步 csi300_cache.csv"""
    cache_path = os.path.join(QLIB_REPO, "csi300_cache.csv")
    bench = pd.read_parquet(os.path.join(OUT_DIR, "sh000300.parquet"))
    bench["d"] = pd.to_datetime(bench["date"]).dt.strftime("%Y-%m-%d")
    old = pd.read_csv(cache_path, sep="\t")
    old.columns = ["date", "close"]
    old_last = str(old["date"].max())[:10]
    new = bench[bench["d"] > old_last]
    if len(new) == 0:
        print(f"[csi300] 已最新 ({old_last})", flush=True)
        return
    with open(cache_path, "a") as f:
        for _, r in new.iterrows():
            f.write(f"{r['d']}\t{r['close']}\n")
    print(f"[csi300] 追加 {len(new)} 天: {new['d'].iloc[0]} ~ {new['d'].iloc[-1]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--syms", help="只更新指定标的 (@file 或逗号分隔)")
    a = ap.parse_args()

    syms = [f[:-8] for f in sorted(os.listdir(OUT_DIR)) if f.endswith(".parquet")]
    if a.syms:
        if a.syms.startswith("@"):
            with open(a.syms[1:]) as f:
                want = {ln.strip() for ln in f if ln.strip()}
        else:
            want = set(a.syms.split(","))
        syms = [s for s in syms if s in want]
        for ix in INDEX_SYMS:
            if ix not in syms:
                syms.append(ix)
    if a.limit:
        syms = syms[:a.limit]

    print(f"[update-ak] {len(syms)} 个 parquet, 增量到 {TODAY[:4]}-{TODAY[4:6]}-{TODAY[6:8]}", flush=True)
    # akshare 需要 chdir 到非源码目录
    os.chdir("/tmp")

    t0 = time.time()
    ok = skip = err = empty = 0
    errs, n_added = [], 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (sym, n) in enumerate(ex.map(update_one, syms), 1):
            if isinstance(n, str):
                err += 1
                errs.append((sym, n))
            elif n == 0:
                empty += 1
            else:
                ok += 1
                n_added += n
            if i % 50 == 0 or i == len(syms):
                el = time.time() - t0
                print(f"  {i}/{len(syms)} ok={ok} empty={empty} err={err} "
                      f"+{n_added}行 {el:.0f}s (eta {el/i*(len(syms)-i):.0f}s)", flush=True)

    print(f"[update-ak] 完成: ok={ok} empty={empty} err={err} 新增 {n_added} 行, "
          f"用时 {time.time()-t0:.0f}s", flush=True)
    for s, e in errs[:20]:
        print("   ERR", s, e, flush=True)
    if err > len(syms) * 0.02:
        sys.exit(f"[CHECK] 失败率 {err/len(syms):.1%} > 2%!")

    sync_csi300_cache()


if __name__ == "__main__":
    main()
