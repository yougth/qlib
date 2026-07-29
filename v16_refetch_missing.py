"""
V16 上线数据补齐: 为 2025+2026 动态池中所有缺行情的股票补 2020-09 至今日频数据
========================================================================
复用 update_qlib_data.py 的 qlib bin 追加机制 (append_stock_to_qlib):
  - 已有数据(截至2020-09-25)的股票 → 从2020-09-28追加
  - 从未入库的次新股 → 按其IPO首个交易日建新bin
补数对象 = build_dynamic_universe(2025)∪(2026) 中 D.features 拉不到2025后数据的股票
"""
import os, sys
import warnings, logging
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from update_qlib_data import (fetch_stock_data, append_stock_to_qlib,
                              QLIB_DATA_DIR, FETCH_START, FETCH_END, MAX_WORKERS)

DATA = "/Users/11164591/Documents/Qoder目录"
CACHE = Path(f"{DATA}/qlib/data_cache/ohlcv_missing_cache.parquet")


def get_missing_stocks():
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    logging.getLogger('qlib.data.data').setLevel(logging.ERROR)
    from v5_xgb_turnover_comparative import build_dynamic_universe, format_qlib_code

    fcf = pd.read_csv(f"{DATA}/fcf_cache.csv", sep='\t'); fcf["code"] = fcf["code"].astype(str).str.zfill(6)
    prof = pd.read_csv(f"{DATA}/profit_cache.csv", sep='\t'); prof["code"] = prof["code"].astype(str).str.zfill(6)
    name_map = {format_qlib_code(c): n for c, n in fcf.drop_duplicates("code")[["code", "name"]].values}
    allu = set()
    for y in [2025, 2026]:
        allu |= set(format_qlib_code(c) for c in build_dynamic_universe(y, fcf, prof))
    px = D.features(sorted(allu), ["$close"], start_time="2025-01-01", end_time="2026-07-31")
    have = set(px.reset_index()["instrument"].unique()) if px is not None and len(px) else set()
    missing_qlib = sorted(allu - have)
    # qlib_code(SH600000) → (纯数字code, name)
    out = []
    for qc in missing_qlib:
        code = qc[2:]
        out.append((code, name_map.get(qc, "")))
    return out, name_map


def fetch_missing(stock_list):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        cached = pd.read_parquet(CACHE)
        cached_codes = set(cached["code"].unique())
        todo = [s for s in stock_list if s[0] not in cached_codes]
        print(f"  缓存已有{len(cached_codes)}只, 还需拉取{len(todo)}只", flush=True)
        if not todo:
            return cached
        all_list = [cached]
    else:
        todo = stock_list
        all_list = []
    print(f"  多线程拉取 {len(todo)} 只 (workers={MAX_WORKERS})...", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_stock_data, c, n): (c, n) for c, n in todo}
        done = 0
        for fu in as_completed(futs):
            code, name, df = fu.result()
            done += 1
            if df is not None and len(df):
                df["code"] = code; df["name"] = name
                all_list.append(df)
            if done % 20 == 0 or done == len(todo):
                print(f"    进度 {done}/{len(todo)}", flush=True)
    data = pd.concat(all_list, ignore_index=True)
    data.to_parquet(CACHE)
    print(f"  拉取完成, 覆盖 {data['code'].nunique()} 只", flush=True)
    return data


def main():
    from update_qlib_data import format_qlib_code as fq2
    print("=" * 60 + "\n  V16 上线数据补齐\n" + "=" * 60, flush=True)
    stock_list, name_map = get_missing_stocks()
    print(f"\n缺行情股票: {len(stock_list)} 只", flush=True)

    ohlcv = fetch_missing(stock_list)

    # 读取完整日历 (已由此前静态名单补数扩展到2026)
    cal_path = QLIB_DATA_DIR / "calendars" / "day.txt"
    with open(cal_path) as f:
        all_cal = [ln.strip() for ln in f.readlines()]
    print(f"\nqlib日历: {all_cal[0]} ~ {all_cal[-1]} ({len(all_cal)}天)", flush=True)
    # 2020-09-25 之前的日历长度 (brand-new股票用不到, 仅传参)
    existing_cal_len = all_cal.index("2020-09-25") + 1 if "2020-09-25" in all_cal else len(all_cal)

    print("\n--- 追加到 qlib bin ---", flush=True)
    ok = fail = 0
    fail_list = []
    for i, (code, name) in enumerate(stock_list):
        qc = fq2(code)
        sdf = ohlcv[ohlcv["code"] == code].copy()
        if len(sdf) == 0:
            fail += 1; fail_list.append((qc, name, "无akshare数据")); continue
        success, msg = append_stock_to_qlib(code, qc, sdf, all_cal, existing_cal_len)
        if success:
            ok += 1
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(stock_list)}] {qc} {name}: {msg}", flush=True)
        else:
            fail += 1; fail_list.append((qc, name, msg))
    print(f"\n补数完成: 成功 {ok}, 失败 {fail}", flush=True)
    if fail_list:
        print("失败清单:", flush=True)
        for qc, name, msg in fail_list:
            print(f"  {qc} {name}: {msg}", flush=True)


if __name__ == "__main__":
    main()
