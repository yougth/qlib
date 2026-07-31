"""
Qlib 数据更新脚本 — 从 akshare 拉取 A 股日频数据并追加到 qlib 二进制存储
=====================================================================
功能：
1. 获取 A 股交易日历（追加到 day.txt）
2. 多线程拉取 277 只股票的日频 OHLCV 前复权数据
3. 将数据追加到 qlib 的 .day.bin 文件中
4. 验证数据完整性

qlib bin 格式：[float32 start_index, float32 data_0, float32 data_1, ...]
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================== 配置 ========================
QLIB_DATA_DIR = Path(os.path.expanduser("~/.qlib/qlib_data/cn_data"))
CSV_PATH = "/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv"
FETCH_START = "2020-09-28"   # qlib 现有数据截至 2020-09-25, 从下一个交易日开始
FETCH_END = "2026-07-23"
FIELDS = ["open", "close", "high", "low", "volume", "factor", "change"]
MAX_WORKERS = 8
CACHE_DIR = Path("/Users/11164591/Documents/Qoder目录/qlib/data_cache")


def format_qlib_code(code):
    code_str = str(code).zfill(6)
    return f"SH{code_str}" if code_str.startswith("6") else f"SZ{code_str}"


def format_akshare_code(code):
    """akshare 使用纯数字代码"""
    return str(code).zfill(6)


# ======================== Step 1: 获取交易日历 ========================
def fetch_trading_calendar():
    """从 akshare 获取 A 股交易日历"""
    import akshare as ak
    print("正在获取 A 股交易日历...")
    trade_dates = ak.tool_trade_date_hist_sina()
    all_dates = pd.to_datetime(trade_dates["trade_date"]).dt.strftime("%Y-%m-%d").tolist()

    # 读取现有 qlib 日历
    cal_path = QLIB_DATA_DIR / "calendars" / "day.txt"
    with open(cal_path) as f:
        existing_dates = [line.strip() for line in f.readlines()]

    existing_set = set(existing_dates)
    print(f"  现有 qlib 日历: {existing_dates[0]} ~ {existing_dates[-1]} ({len(existing_dates)} 天)")

    # 找出需要追加的新日期
    new_dates = [d for d in all_dates if d not in existing_set and d >= FETCH_START and d <= FETCH_END]
    new_dates.sort()
    print(f"  需要追加的新交易日: {len(new_dates)} 天")
    if new_dates:
        print(f"  新日期范围: {new_dates[0]} ~ {new_dates[-1]}")

    return existing_dates, new_dates


def update_calendar(new_dates):
    """将新交易日追加到 day.txt"""
    if not new_dates:
        print("  无需更新日历")
        return
    cal_path = QLIB_DATA_DIR / "calendars" / "day.txt"
    with open(cal_path, "a") as f:
        for d in new_dates:
            f.write(d + "\n")
    print(f"  日历已追加 {len(new_dates)} 天")


# ======================== Step 2: 拉取 OHLCV 数据 ========================
def fetch_stock_data(code, name):
    """拉取单只股票的日频数据（前复权）"""
    import akshare as ak
    ak_code = format_akshare_code(code)

    try:
        # 前复权数据
        df_qfq = ak.stock_zh_a_hist(
            symbol=ak_code, period="daily",
            start_date=FETCH_START.replace("-", ""),
            end_date=FETCH_END.replace("-", ""),
            adjust="qfq"
        )
        if df_qfq is None or len(df_qfq) == 0:
            return code, name, None

        df_qfq = df_qfq.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "涨跌幅": "change_pct",
        })
        df_qfq["date"] = pd.to_datetime(df_qfq["date"]).dt.strftime("%Y-%m-%d")
        df_qfq = df_qfq[["date", "open", "close", "high", "low", "volume", "change_pct"]].copy()
        df_qfq["change"] = df_qfq["change_pct"] / 100.0

        return code, name, df_qfq
    except Exception as e:
        print(f"  [ERROR] {code} {name}: {e}")
        return code, name, None


def fetch_all_stock_data(stock_list):
    """多线程拉取所有股票数据"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "ohlcv_cache.parquet"

    # 检查缓存
    if cache_file.exists():
        print("发现缓存文件，正在加载...")
        cached = pd.read_parquet(cache_file)
        cached_codes = set(cached["code"].unique())
        missing = [s for s in stock_list if s[0] not in cached_codes]
        if not missing:
            print(f"  缓存完整 ({len(cached_codes)} 只股票)")
            return cached
        print(f"  缓存有 {len(cached_codes)} 只，还需拉取 {len(missing)} 只")
        stock_list = missing
        all_data = cached
    else:
        all_data = pd.DataFrame()
        all_data_list = []

    print(f"开始多线程拉取 {len(stock_list)} 只股票数据 (workers={MAX_WORKERS})...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_stock_data, code, name): (code, name)
            for code, name in stock_list
        }
        done = 0
        for future in as_completed(futures):
            code, name, df = future.result()
            done += 1
            if df is not None:
                df["code"] = code
                df["name"] = name
                all_data_list.append(df)
            if done % 50 == 0 or done == len(stock_list):
                print(f"  进度: {done}/{len(stock_list)}")

    if all_data_list:
        new_data = pd.concat(all_data_list, ignore_index=True)
        if len(all_data) > 0:
            all_data = pd.concat([all_data, new_data], ignore_index=True)
        else:
            all_data = new_data
        all_data.to_parquet(cache_file)
        print(f"  缓存已保存: {cache_file}")

    print(f"  总计: {all_data['code'].nunique()} 只股票, {len(all_data)} 条记录")
    return all_data


# ======================== Step 3: 追加到 qlib 二进制存储 ========================
def append_stock_to_qlib(code, qlib_code, stock_df, all_calendar_dates, existing_cal_len):
    """将单只股票的数据追加到 qlib bin 文件"""
    features_dir = QLIB_DATA_DIR / "features" / qlib_code
    if not features_dir.exists():
        # 股票在 qlib 中不存在，创建目录并从新数据开始
        features_dir.mkdir(parents=True, exist_ok=True)
        start_index = existing_cal_len  # 从新数据开始的日历索引
        has_existing = False
    else:
        has_existing = True

    # 将股票数据按日期对齐到完整日历
    stock_df = stock_df.sort_values("date").copy()
    stock_df["cal_idx"] = stock_df["date"].apply(
        lambda d: all_calendar_dates.index(d) if d in all_calendar_dates else -1
    )
    stock_df = stock_df[stock_df["cal_idx"] >= 0]

    if len(stock_df) == 0:
        return False, "no valid dates"

    # 获取新数据在日历中的范围
    new_start_idx = int(stock_df["cal_idx"].min())
    new_end_idx = int(stock_df["cal_idx"].max())

    # 读取现有数据的边界价格（用于连续性校正）
    if has_existing:
        close_bin_path = features_dir / "close.day.bin"
        with open(close_bin_path, "rb") as f:
            existing_data = np.frombuffer(f.read(), dtype="<f4")
        existing_start_idx = int(existing_data[0])
        existing_data_values = existing_data[1:]
        existing_end_idx = existing_start_idx + len(existing_data_values) - 1
        last_close = existing_data_values[-1]
        last_factor_val = None

        # 读取最后一个 factor 值
        factor_bin_path = features_dir / "factor.day.bin"
        if factor_bin_path.exists():
            with open(factor_bin_path, "rb") as f:
                factor_data = np.frombuffer(f.read(), dtype="<f4")
            last_factor_val = float(factor_data[-1])
    else:
        existing_end_idx = new_start_idx - 1
        last_close = None
        last_factor_val = 1.0

    # 构建新数据数组（对齐到日历）
    new_cal_len = len(all_calendar_dates)
    # 新数据覆盖的范围: max(existing_end_idx+1, new_start_idx) 到 new_end_idx
    append_start = max(existing_end_idx + 1, new_start_idx)
    append_end = new_end_idx

    if append_start > append_end:
        return False, "no new data to append"

    append_len = append_end - append_start + 1

    # 为每个字段创建追加数组
    field_arrays = {}
    for field in FIELDS:
        field_arrays[field] = np.full(append_len, np.nan, dtype=np.float32)

    # 填充数据
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
        field_arrays["factor"][idx] = last_factor_val if last_factor_val else 1.0

    # 连续性校正：如果新旧数据在边界处的价格差异较大，进行缩放
    if has_existing and last_close is not None and last_factor_val:
        # 找到新数据中第一个有效收盘价
        first_valid_idx = np.where(~np.isnan(field_arrays["close"]))[0]
        if len(first_valid_idx) > 0:
            first_new_close = field_arrays["close"][first_valid_idx[0]]
            if first_new_close > 0 and last_close > 0:
                scale = last_close / first_new_close
                if 0.5 < scale < 2.0 and abs(scale - 1.0) > 0.01:
                    # 价格有跳变，可能是复权基准不同，进行缩放
                    for field in ["open", "close", "high", "low"]:
                        field_arrays[field] *= scale

    # 写入 bin 文件
    for field in FIELDS:
        bin_path = features_dir / f"{field}.day.bin"
        if has_existing and bin_path.exists():
            # 追加模式
            with open(bin_path, "ab") as f:
                field_arrays[field].astype("<f4").tofile(f)
        else:
            # 新建模式
            start_idx = append_start
            with open(bin_path, "wb") as f:
                np.hstack([np.array([start_idx], dtype="<f4"),
                           field_arrays[field]]).astype("<f4").tofile(f)

    return True, f"appended {append_len} days ({all_calendar_dates[append_start]} ~ {all_calendar_dates[append_end]})"


def update_all_stocks(ohlcv_data, all_calendar_dates, existing_cal_len):
    """更新所有股票的 qlib 数据"""
    df_stocks = pd.read_csv(CSV_PATH)
    stock_codes = df_stocks["code"].tolist()

    success = 0
    failed = 0

    for i, code in enumerate(stock_codes):
        code_str = str(code).zfill(6)
        qlib_code = format_qlib_code(code)
        name = df_stocks[df_stocks["code"] == code]["name"].values[0] if code in df_stocks["code"].values else ""

        stock_df = ohlcv_data[ohlcv_data["code"] == code].copy()
        if len(stock_df) == 0:
            # 尝试用字符串匹配
            stock_df = ohlcv_data[ohlcv_data["code"].astype(str).str.zfill(6) == code_str].copy()

        if len(stock_df) == 0:
            print(f"  [{i+1}/{len(stock_codes)}] {qlib_code} {name}: 无数据")
            failed += 1
            continue

        ok, msg = append_stock_to_qlib(code, qlib_code, stock_df, all_calendar_dates, existing_cal_len)
        if ok:
            success += 1
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(stock_codes)}] {qlib_code} {name}: {msg}")
        else:
            failed += 1
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(stock_codes)}] {qlib_code} {name}: FAILED - {msg}")

    print(f"\n更新完成: 成功 {success}, 失败 {failed}")
    return success, failed


# ======================== Step 4: 验证数据 ========================
def verify_data():
    """验证更新后的数据完整性"""
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN)
    from qlib.data import D

    cal = D.calendar(start_time="2020-09-01", end_time="2026-07-23")
    print(f"\n验证日历: {cal[0].date()} ~ {cal[-1].date()} ({len(cal)} 天)")

    # 检查几只股票
    test_codes = ["SH600519", "SH601318", "SZ000858"]
    for code in test_codes:
        df = D.features([code], fields=["$close", "$volume"], start_time="2024-01-01", end_time="2024-01-10")
        if len(df) > 0:
            print(f"  {code}: {len(df)} 条, close范围 {df['$close'].min():.2f} ~ {df['$close'].max():.2f}")
        else:
            print(f"  {code}: 无数据!")


# ======================== 主流程 ========================
if __name__ == "__main__":
    print("=" * 60)
    print("  Qlib 数据更新脚本")
    print(f"  追加范围: {FETCH_START} ~ {FETCH_END}")
    print("=" * 60)

    # Step 1: 交易日历
    print("\n--- Step 1: 获取交易日历 ---")
    existing_dates, new_dates = fetch_trading_calendar()
    if not new_dates:
        print("日历已是最新，无需更新")
    else:
        update_calendar(new_dates)

    # 读取完整日历
    cal_path = QLIB_DATA_DIR / "calendars" / "day.txt"
    with open(cal_path) as f:
        all_calendar_dates = [line.strip() for line in f.readlines()]
    existing_cal_len = len(existing_dates)
    print(f"  完整日历: {len(all_calendar_dates)} 天")

    # Step 2: 拉取 OHLCV
    print("\n--- Step 2: 拉取 OHLCV 数据 ---")
    df_stocks = pd.read_csv(CSV_PATH)
    stock_list = [(str(row["code"]).zfill(6), row["name"]) for _, row in df_stocks.iterrows()]
    ohlcv_data = fetch_all_stock_data(stock_list)

    # Step 3: 追加到 qlib
    print("\n--- Step 3: 追加到 qlib 二进制存储 ---")
    update_all_stocks(ohlcv_data, all_calendar_dates, existing_cal_len)

    # Step 4: 验证
    print("\n--- Step 4: 验证数据 ---")
    verify_data()

    print("\n" + "=" * 60)
    print("  数据更新完成!")
    print("=" * 60)
