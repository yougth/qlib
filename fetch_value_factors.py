"""
拉取价值因子原始数据 — ROE/EPS/BPS（用于计算 PE/PB/ROE 因子）
================================================================
数据源: akshare stock_financial_analysis_indicator（新浪财经财务指标）
输出:   value_factors_cache.csv (code, year, quarter, roe, eps, bps)
"""
import sys
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

CSV_PATH = "/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv"
OUTPUT_PATH = "/Users/11164591/Documents/Qoder目录/value_factors_cache.csv"
START_YEAR = "2014"  # 多拉几年用于计算历史分位数


def fetch_one(code, name):
    """拉取单只股票的财务指标"""
    import akshare as ak
    ak_code = str(code).zfill(6)
    try:
        df = ak.stock_financial_analysis_indicator(symbol=ak_code, start_year=START_YEAR)
        if df is None or len(df) == 0:
            return code, name, None
        # 提取关键字段
        cols_map = {
            "日期": "date",
            "净资产收益率(%)": "roe",
            "摊薄每股收益(元)": "eps",
            "每股净资产_调整前(元)": "bps",
            "股息发放率(%)": "div_payout",
            "净利润增长率(%)": "profit_growth",
            "主营业务收入增长率(%)": "revenue_growth",
        }
        keep = {k: v for k, v in cols_map.items() if k in df.columns}
        df = df[list(keep.keys())].rename(columns=keep).copy()
        df["code"] = ak_code
        df["name"] = name
        # 解析日期 -> year, quarter
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["quarter"] = df["date"].dt.month  # 3/6/9/12
        # 转数值
        for col in ["roe", "eps", "bps", "div_payout", "profit_growth", "revenue_growth"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return code, name, df
    except Exception as e:
        print(f"  [ERROR] {ak_code} {name}: {e}")
        return code, name, None


if __name__ == "__main__":
    df_stocks = pd.read_csv(CSV_PATH)
    stock_list = [(str(row["code"]).zfill(6), row["name"]) for _, row in df_stocks.iterrows()]
    print(f"股票池: {len(stock_list)} 只")

    # 检查缓存
    import os
    if os.path.exists(OUTPUT_PATH):
        cached = pd.read_csv(OUTPUT_PATH)
        cached_codes = set(cached["code"].astype(str).str.zfill(6).unique())
        missing = [(c, n) for c, n in stock_list if c not in cached_codes]
        if not missing:
            print(f"缓存完整 ({len(cached_codes)} 只), 无需重新拉取")
            print(f"数据: {len(cached)} 行, 列: {cached.columns.tolist()}")
            sys.exit(0)
        print(f"缓存有 {len(cached_codes)} 只, 还需 {len(missing)} 只")
        all_dfs = [cached]
        stock_list = missing
    else:
        all_dfs = []

    print(f"开始拉取 {len(stock_list)} 只股票财务指标 (workers=4)...")
    done = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_one, c, n): (c, n) for c, n in stock_list}
        for future in as_completed(futures):
            code, name, df = future.result()
            done += 1
            if df is not None:
                all_dfs.append(df)
            if done % 30 == 0 or done == len(stock_list):
                print(f"  进度: {done}/{len(stock_list)}")
            time.sleep(0.3)  # 限流

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        # 去重保留最新
        result = result.drop_duplicates(subset=["code", "date"], keep="last")
        result.to_csv(OUTPUT_PATH, index=False)
        print(f"\n已保存: {OUTPUT_PATH}")
        print(f"  股票数: {result['code'].nunique()}")
        print(f"  总行数: {len(result)}")
        print(f"  年份范围: {result['year'].min()} ~ {result['year'].max()}")
        print(f"  列: {result.columns.tolist()}")
    else:
        print("无数据获取!")
