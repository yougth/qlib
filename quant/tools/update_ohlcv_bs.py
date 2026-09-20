#!/usr/bin/env python3
"""
update_ohlcv_bs —— baostock 版行情增量更新
================================================================================
腾讯 WAF + 东财 eastmoney 双双被封, 改用 baostock 做增量追加。
baostock 后复权基准与腾讯不同, 通过重叠日比例换算保持序列连续。

每只股票 2 个请求:
  1. baostock adjustflag=1 (后复权) 从重叠日到今天 → 换算到腾讯口径
  2. baostock adjustflag=3 (不复权) 同期 → 取 raw price 计算 factor

用法:
    python3 tools/update_ohlcv_bs.py                     # 更新池内全部 (需 /tmp/pool_syms.txt)
    python3 tools/update_ohlcv_bs.py --syms @file.txt    # 指定标的
    python3 tools/update_ohlcv_bs.py --limit 5           # 试点
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

QUANT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QLIB_REPO = os.path.dirname(QUANT_DIR)
OUT_DIR = os.path.join(QLIB_REPO, "data_cache", "tencent")
INDEX_SYMS = ["sh000300", "sh000905", "sh000001"]
TODAY = pd.Timestamp.now().strftime("%Y-%m-%d")

_bs_logged_in = False


def bs_login():
    global _bs_logged_in
    if not _bs_logged_in:
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")
        _bs_logged_in = True


def bs_query(code_bs, fields, start, end, adjustflag=None):
    import baostock as bs
    kwargs = {"start_date": start, "end_date": end, "frequency": "d"}
    if adjustflag:
        kwargs["adjustflag"] = adjustflag
    rs = bs.query_history_k_data_plus(code_bs, fields, **kwargs)
    data = []
    while (rs.error_code == "0") & rs.next():
        data.append(rs.get_row_data())
    return pd.DataFrame(data, columns=rs.fields) if data else pd.DataFrame()


def bs_to_qlib_sym(bs_code):
    """sz.000001 → sz000001"""
    return bs_code.replace(".", "")


def qlib_to_bs_sym(qlib_sym):
    """sz000001 → sz.000001, sh000300 → sh.000300"""
    return qlib_sym[:2] + "." + qlib_sym[2:]


def update_one(sym):
    """增量更新一只标的 (baostock → tencent 口径换算)"""
    path = os.path.join(OUT_DIR, f"{sym}.parquet")
    if not os.path.exists(path):
        return sym, f"ERR no parquet"

    try:
        old = pd.read_parquet(path)
    except Exception as e:
        return sym, f"ERR read: {e}"

    old["date"] = pd.to_datetime(old["date"])
    last_date = old["date"].iloc[-1]
    last_close_hfq = float(old["close"].iloc[-1])
    old_factor = float(old["factor"].iloc[-1])
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if start > TODAY:
        return sym, -2  # 已最新

    is_index = sym in INDEX_SYMS
    if is_index:
        # 指数: 不需要复权, 直接取 raw
        bs_sym = qlib_to_bs_sym(sym)
        df = bs_query(bs_sym, "date,open,close,high,low,volume",
                       last_date.strftime("%Y-%m-%d"), TODAY)
        if len(df) == 0:
            return sym, 0
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "close", "high", "low", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df["date"] > last_date]
        if len(df) == 0:
            return sym, 0
        df["factor"] = 1.0
        df["symbol"] = sym
        new = df[["date", "open", "close", "high", "low", "volume", "factor", "symbol"]]
    else:
        bs_sym = qlib_to_bs_sym(sym)
        # 1) 取后复权 (含重叠日, 用于算 ratio)
        hfq = bs_query(bs_sym, "date,open,close,high,low,volume",
                        last_date.strftime("%Y-%m-%d"), TODAY, adjustflag="1")
        if len(hfq) == 0:
            return sym, 0
        hfq["date"] = pd.to_datetime(hfq["date"])
        for c in ["open", "close", "high", "low", "volume"]:
            hfq[c] = pd.to_numeric(hfq[c], errors="coerce")

        # 2) 取不复权 (含重叠日, 用于算 factor)
        raw = bs_query(bs_sym, "date,close",
                        last_date.strftime("%Y-%m-%d"), TODAY, adjustflag="3")
        if len(raw) > 0:
            raw["date"] = pd.to_datetime(raw["date"])
            raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
            raw = raw.rename(columns={"close": "raw_close"})
        else:
            raw = pd.DataFrame({"date": [], "raw_close": []})

        # 3) 重叠日 ratio = tencent_hfq / baostock_hfq
        overlap = hfq[hfq["date"] == last_date]
        if len(overlap) > 0 and overlap["close"].iloc[0] > 0:
            bs_hfq_overlap = float(overlap["close"].iloc[0])
            ratio = last_close_hfq / bs_hfq_overlap
        else:
            ratio = 1.0  # 无法校准, 用 1:1 (误差大但不会崩)

        # 4) 新行: baostock_hfq × ratio → tencent 口径 hfq
        new = hfq[hfq["date"] > last_date].copy()
        if len(new) == 0:
            return sym, 0
        for c in ["open", "close", "high", "low"]:
            new[c] = new[c] * ratio

        # 5) factor: raw_close 有则算, 无则沿用
        if len(raw) > 0:
            raw_new = raw[raw["date"] > last_date]
            new = new.merge(raw_new[["date", "raw_close"]], on="date", how="left")
            new["factor"] = np.where(
                new["raw_close"].notna() & (new["raw_close"] > 0),
                new["close"] / new["raw_close"],
                old_factor
            )
            new = new.drop(columns=["raw_close"])
        else:
            new["factor"] = old_factor

        new["symbol"] = sym
        new = new[["date", "open", "close", "high", "low", "volume", "factor", "symbol"]]

    # 非正价检查
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


def sync_csi300_cache():
    """从 sh000300.parquet 同步 csi300_cache.csv"""
    cache_path = os.path.join(QLIB_REPO, "..", "csi300_cache.csv")
    cache_path = os.path.normpath(cache_path)
    bench = pd.read_parquet(os.path.join(OUT_DIR, "sh000300.parquet"))
    bench["d"] = pd.to_datetime(bench["date"]).dt.strftime("%Y-%m-%d")
    if not os.path.exists(cache_path):
        bench[["d", "close"]].to_csv(cache_path, sep="\t", index=False, header=False)
        print(f"[csi300] 新建 {len(bench)} 天", flush=True)
        return
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
            if ix not in syms and os.path.exists(os.path.join(OUT_DIR, f"{ix}.parquet")):
                syms.append(ix)
    if a.limit:
        syms = syms[:a.limit]

    print(f"[update-bs] {len(syms)} 个 parquet, 增量到 {TODAY}", flush=True)
    bs_login()

    t0 = time.time()
    ok = skip = err = empty = 0
    errs, n_added = [], 0
    for i, sym in enumerate(syms, 1):
        try:
            sym, n = update_one(sym)
        except Exception as e:
            sym, n = sym, f"ERR {type(e).__name__}: {str(e)[:60]}"
        if isinstance(n, str):
            err += 1
            errs.append((sym, n))
        elif n == 0 or n == -2:
            empty += 1
        else:
            ok += 1
            n_added += n
        if i % 20 == 0 or i == len(syms):
            el = time.time() - t0
            eta = el / i * (len(syms) - i) if i > 0 else 0
            print(f"  {i}/{len(syms)} ok={ok} empty={empty} err={err} "
                  f"+{n_added}行 {el:.0f}s (eta {eta:.0f}s)", flush=True)
        time.sleep(0.05)  # baostock 轻量限速

    import baostock as bs
    bs.logout()
    _bs_logged_in = False

    print(f"[update-bs] 完成: ok={ok} empty={empty} err={err} 新增 {n_added} 行, "
          f"用时 {time.time()-t0:.0f}s", flush=True)
    for s, e in errs[:20]:
        print("   ERR", s, e, flush=True)
    if err > len(syms) * 0.02:
        sys.exit(f"[CHECK] 失败率 {err/len(syms):.1%} > 2%!")

    sync_csi300_cache()


if __name__ == "__main__":
    main()
