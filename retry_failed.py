"""重试获取失败的50只股票数据"""
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

QLIB_DATA_DIR = Path(os.path.expanduser("~/.qlib/qlib_data/cn_data"))
CSV_PATH = "/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv"
FETCH_START = "2020-09-28"
FETCH_END = "2026-07-23"
FIELDS = ["open", "close", "high", "low", "volume", "factor", "change"]
CACHE_DIR = Path("/Users/11164591/Documents/Qoder目录/qlib/data_cache")

def format_qlib_code(code):
    code_str = str(code).zfill(6)
    return f"SH{code_str}" if code_str.startswith("6") else f"SZ{code_str}"

def fetch_stock_data(code, name):
    import akshare as ak
    ak_code = str(code).zfill(6)
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(
                symbol=ak_code, period="daily",
                start_date=FETCH_START.replace("-", ""),
                end_date=FETCH_END.replace("-", ""),
                adjust="qfq"
            )
            if df is None or len(df) == 0:
                return code, name, None
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "涨跌幅": "change_pct",
            })
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df = df[["date", "open", "close", "high", "low", "volume", "change_pct"]].copy()
            df["change"] = df["change_pct"] / 100.0
            return code, name, df
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    return code, name, None

def append_stock_to_qlib(code, qlib_code, stock_df, all_calendar_dates, existing_cal_len):
    features_dir = QLIB_DATA_DIR / "features" / qlib_code
    has_existing = features_dir.exists()
    if not has_existing:
        features_dir.mkdir(parents=True, exist_ok=True)

    stock_df = stock_df.sort_values("date").copy()
    date_to_idx = {d: i for i, d in enumerate(all_calendar_dates)}
    stock_df["cal_idx"] = stock_df["date"].apply(lambda d: date_to_idx.get(d, -1))
    stock_df = stock_df[stock_df["cal_idx"] >= 0]
    if len(stock_df) == 0:
        return False, "no valid dates"

    new_start_idx = int(stock_df["cal_idx"].min())
    new_end_idx = int(stock_df["cal_idx"].max())

    if has_existing:
        close_bin_path = features_dir / "close.day.bin"
        with open(close_bin_path, "rb") as f:
            existing_data = np.frombuffer(f.read(), dtype="<f4")
        existing_start_idx = int(existing_data[0])
        existing_data_values = existing_data[1:]
        existing_end_idx = existing_start_idx + len(existing_data_values) - 1
        last_close = existing_data_values[-1]
        
        factor_bin_path = features_dir / "factor.day.bin"
        if factor_bin_path.exists():
            with open(factor_bin_path, "rb") as f:
                factor_data = np.frombuffer(f.read(), dtype="<f4")
            last_factor_val = float(factor_data[-1])
        else:
            last_factor_val = 1.0
    else:
        existing_end_idx = new_start_idx - 1
        last_close = None
        last_factor_val = 1.0

    # FIX: Always start from existing_end_idx + 1 to ensure continuity
    append_start = existing_end_idx + 1
    append_end = new_end_idx
    if append_start > append_end:
        return False, "no new data to append"
    append_len = append_end - append_start + 1

    field_arrays = {}
    for field in FIELDS:
        field_arrays[field] = np.full(append_len, np.nan, dtype=np.float32)

    for _, row in stock_df.iterrows():
        idx = int(row["cal_idx"]) - append_start
        if idx < 0 or idx >= append_len:
            continue
        field_arrays["open"][idx] = float(row["open"])
        field_arrays["close"][idx] = float(row["close"])
        field_arrays["high"][idx] = float(row["high"])
        field_arrays["low"][idx] = float(row["low"])
        field_arrays["volume"][idx] = float(row["volume"])
        field_arrays["change"][idx] = float(row["change"])
        field_arrays["factor"][idx] = last_factor_val

    # 连续性校正
    if has_existing and last_close is not None:
        first_valid = np.where(~np.isnan(field_arrays["close"]))[0]
        if len(first_valid) > 0:
            first_new_close = field_arrays["close"][first_valid[0]]
            if first_new_close > 0 and last_close > 0:
                scale = last_close / first_new_close
                if 0.5 < scale < 2.0 and abs(scale - 1.0) > 0.01:
                    for field in ["open", "close", "high", "low"]:
                        field_arrays[field] *= scale

    for field in FIELDS:
        bin_path = features_dir / f"{field}.day.bin"
        if has_existing and bin_path.exists():
            with open(bin_path, "ab") as f:
                field_arrays[field].astype("<f4").tofile(f)
        else:
            with open(bin_path, "wb") as f:
                np.hstack([np.array([append_start], dtype="<f4"),
                           field_arrays[field]]).astype("<f4").tofile(f)
    return True, f"appended {append_len} days"

if __name__ == "__main__":
    # 读取日历
    cal_path = QLIB_DATA_DIR / "calendars" / "day.txt"
    with open(cal_path) as f:
        all_calendar_dates = [line.strip() for line in f.readlines()]
    existing_cal_len = 4943  # 原始日历长度

    # 读取股票列表
    df_stocks = pd.read_csv(CSV_PATH)

    # 找出需要更新的股票（bin文件长度不够的）
    to_retry = []
    for _, row in df_stocks.iterrows():
        code = str(row["code"]).zfill(6)
        qlib_code = format_qlib_code(code)
        bin_path = QLIB_DATA_DIR / "features" / qlib_code / "close.day.bin"
        if not bin_path.exists():
            to_retry.append((code, row["name"]))
            continue
        with open(bin_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype="<f4")
        start_idx = int(data[0])
        values = data[1:]
        expected_len = len(all_calendar_dates) - start_idx
        if len(values) < expected_len:
            to_retry.append((code, row["name"]))

    print(f"需要重试: {len(to_retry)} 只股票")
    
    # 多线程拉取（降低并发避免连接错误）
    cache_file = CACHE_DIR / "ohlcv_cache.parquet"
    cached = pd.read_parquet(cache_file) if cache_file.exists() else pd.DataFrame()
    cached_codes = set(cached["code"].astype(str).str.zfill(6).unique()) if len(cached) > 0 else set()
    
    missing_to_fetch = [(c, n) for c, n in to_retry if c not in cached_codes]
    print(f"缓存中已有: {len(to_retry) - len(missing_to_fetch)}, 需要拉取: {len(missing_to_fetch)}")

    if missing_to_fetch:
        print(f"开始拉取 (workers=3)...")
        new_data_list = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(fetch_stock_data, c, n): (c, n) for c, n in missing_to_fetch}
            done = 0
            for future in as_completed(futures):
                code, name, df = future.result()
                done += 1
                if df is not None:
                    df["code"] = code
                    df["name"] = name
                    new_data_list.append(df)
                else:
                    print(f"  FAILED: {code} {name}")
                if done % 10 == 0 or done == len(missing_to_fetch):
                    print(f"  进度: {done}/{len(missing_to_fetch)}")

        if new_data_list:
            new_data = pd.concat(new_data_list, ignore_index=True)
            if len(cached) > 0:
                cached = pd.concat([cached, new_data], ignore_index=True)
            else:
                cached = new_data
            cached.to_parquet(cache_file)
            print(f"缓存已更新: {cached['code'].nunique()} 只股票")

    # 追加到 qlib
    print("\n追加到 qlib...")
    success = 0
    failed = 0
    for code, name in to_retry:
        qlib_code = format_qlib_code(code)
        stock_df = cached[cached["code"].astype(str).str.zfill(6) == code].copy() if len(cached) > 0 else pd.DataFrame()
        if len(stock_df) == 0:
            failed += 1
            continue
        ok, msg = append_stock_to_qlib(code, qlib_code, stock_df, all_calendar_dates, existing_cal_len)
        if ok:
            success += 1
        else:
            failed += 1
            print(f"  FAILED: {qlib_code} {name}: {msg}")

    print(f"\n重试完成: 成功 {success}, 失败 {failed}")
