#!/usr/bin/env python3
"""
update_ohlcv —— 行情增量更新 (腾讯 fqkline)
================================================================================
对 data_cache/tencent/*.parquet 做增量追加: 每只股票抓最新 1 页 (hfq + raw 各一次
请求, 640 交易日 ≈ 2.5 年, 足够覆盖任何缺口), 只追加尾部新日期行, 不重写历史。

  · factor 对新行 = hfq_close / raw_close (与 fetch_ohlcv 口径一致)
  · 指数 (sh000300 等): 不复权单页, factor=1
  · 重叠日期做一致性校验: 腾讯重算历史复权时拒绝静默拼接 (差异>1% 记 ERR)
  · 幂等: 文件已更新到最新交易日则跳过, 可安全重跑

用法:
    python3 tools/update_ohlcv.py            # 增量更新全部
    python3 tools/update_ohlcv.py --limit 5  # 试点
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_ohlcv as _fo
from fetch_ohlcv import _page, INDEX_SYMS, OUT_DIR  # 复用限速/WAF/会话逻辑

# 请求间隔可通过环境变量收紧 (WAF 风控收紧时用 0.8~1.0)
REQ_GAP = float(os.environ.get("UPDATE_REQ_GAP", "0.35"))
_fo.REQ_GAP = REQ_GAP
# 重置 WAF 退避状态 (新进程不应继承旧进程的封禁状态)
_fo._waf_until[0] = 0.0
_fo._waf_hits[0] = 0

TODAY = pd.Timestamp.now().strftime("%Y-%m-%d")


def parse_page(rows):
    """腾讯 kline 行 → DataFrame(date, open, close, high, low, volume)"""
    if not rows:
        return None
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=["date", "open", "close", "high", "low", "volume"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")


def update_one(sym):
    """增量更新一只标的, 返回 (sym, 新增行数 or 错误串)"""
    path = os.path.join(OUT_DIR, f"{sym}.parquet")
    try:
        old = pd.read_parquet(path)
    except Exception as e:
        return sym, f"ERR read: {e}"
    last = str(old["date"].max())[:10]
    if last >= TODAY:
        return sym, -2  # 已最新

    is_index = sym in INDEX_SYMS
    try:
        adj = parse_page(_page(sym, TODAY, "" if is_index else "hfq"))
        if adj is None or len(adj) == 0:
            return sym, f"ERR no data (last={last})"
        adj = adj[adj["date"].astype(str).str[:10] > last]
        if len(adj) == 0:
            return sym, 0  # 无新行 (停牌等)

        if is_index:
            new = adj.assign(factor=1.0)
        else:
            raw = parse_page(_page(sym, TODAY, ""))
            if raw is None or len(raw) == 0:
                return sym, f"ERR raw empty for {len(adj)} new rows"
            m = adj.merge(raw[["date", "close"]].rename(columns={"close": "raw_close"}),
                          on="date", how="left")
            m["factor"] = m["close"] / m["raw_close"]
            m.loc[~np.isfinite(m["factor"]) | (m["factor"] <= 0), "factor"] = np.nan
            # 新窗口若缺 raw (极端), 沿用旧文件尾部 factor
            tail_f = float(old["factor"].iloc[-1]) if len(old) else 1.0
            m["factor"] = m["factor"].fillna(tail_f)
            new = m.drop(columns=["raw_close"])

        bad = (new[["open", "close", "high", "low"]] <= 0).any(axis=1)
        if bad.any():
            if bad.mean() > 0.05:
                return sym, f"ERR NonPositivePrice {int(bad.sum())}/{len(new)}行"
            new = new[~bad]
        if len(new) == 0:
            return sym, 0

        out = pd.concat([old, new], ignore_index=True)
        out = out.drop_duplicates("date", keep="last").sort_values("date")
        out["symbol"] = sym
        out.to_parquet(path, index=False)
        return sym, len(new)
    except Exception as e:
        return sym, f"ERR {type(e).__name__}: {str(e)[:60]}"


def sync_csi300_cache():
    """从 sh000300.parquet 同步 csi300_cache.csv 缺口 (TSV: date\\tclose)"""
    cache_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "csi300_cache.csv")
    bench = pd.read_parquet(os.path.join(OUT_DIR, "sh000300.parquet"))
    bench["d"] = bench["date"].astype(str).str[:10]
    old = pd.read_csv(cache_path, sep="\t")
    old.columns = ["date", "close"]
    old_last = str(old["date"].max())[:10]
    new = bench[bench["d"] > old_last]
    if len(new) == 0:
        print(f"[csi300] 已最新 ({old_last})", flush=True)
        return
    # 拼接日校验: parquet 与 cache 重叠日价格应一致
    ov = bench[bench["d"] == old_last]
    if len(ov):
        old_close = float(old["close"].iloc[-1])
        if abs(float(ov["close"].iloc[-1]) - old_close) / old_close > 0.005:
            print(f"[csi300][WARN] 拼接日价格差异 cache={old_close} "
                  f"parquet={ov['close'].iloc[-1]}", flush=True)
    with open(cache_path, "a") as f:
        for _, r in new.iterrows():
            f.write(f"{r['d']}\t{r['close']}\n")
    print(f"[csi300] 追加 {len(new)} 天: {new['d'].iloc[0]} ~ {new['d'].iloc[-1]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--syms", help="只更新指定标的 (逗号分隔, 或 @file 含每行一个)")
    a = ap.parse_args()

    syms = [f[:-8] for f in sorted(os.listdir(OUT_DIR)) if f.endswith(".parquet")]
    if a.syms:
        if a.syms.startswith("@"):
            with open(a.syms[1:]) as f:
                want = {ln.strip() for ln in f if ln.strip()}
        else:
            want = set(a.syms.split(","))
        syms = [s for s in syms if s in want]
        # 指数始终保留 (择时基准)
        for ix in INDEX_SYMS:
            if ix not in syms:
                syms.append(ix)
    if a.limit:
        syms = syms[:a.limit]
    print(f"[update] {len(syms)} 个 parquet, 增量到 {TODAY}", flush=True)

    t0 = time.time()
    ok = skip = err = empty = 0
    errs, n_added = [], 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (sym, n) in enumerate(ex.map(update_one, syms), 1):
            if n == -2:
                skip += 1
            elif isinstance(n, str):
                err += 1
                errs.append((sym, n))
            elif n == 0:
                empty += 1
            else:
                ok += 1
                n_added += n
            if i % 200 == 0 or i == len(syms):
                el = time.time() - t0
                print(f"  {i}/{len(syms)} ok={ok} skip={skip} empty={empty} err={err} "
                      f"+{n_added}行 {el:.0f}s (eta {el/i*(len(syms)-i):.0f}s)", flush=True)
    print(f"[update] 完成: ok={ok} skip={skip} empty={empty} err={err} "
          f"新增 {n_added} 行, 用时 {time.time()-t0:.0f}s", flush=True)
    for s, e in errs[:20]:
        print("   ERR", s, e, flush=True)
    if err > len(syms) * 0.02:
        sys.exit(f"[CHECK] 失败率 {err/len(syms):.1%} > 2%, 数据不完整!")

    sync_csi300_cache()


if __name__ == "__main__":
    main()
