"""
V14 数据抓取: 估值因子(日频) + 深度基本面因子(季频)
- 估值: 东财 stock_value_em → PE-TTM/PB/PS/PCF/PEG (2018-01起有历史)
- 财务: 新浪 stock_financial_analysis_indicator → ROE/毛利率/增速/负债率/应收周转
- 支持断点续传: 已抓过的code跳过
"""
import os, sys, time
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")
import pandas as pd
import akshare as ak
from v5_xgb_turnover_comparative import build_dynamic_universe

WORK = "/Users/11164591/Documents/Qoder目录"
VAL_CACHE = f"{WORK}/valuation_cache.csv"
FIN_CACHE = f"{WORK}/finind_cache.csv"

FIN_COLS = {
    "日期": "report_date",
    "加权净资产收益率(%)": "roe",
    "销售毛利率(%)": "gross_margin",
    "主营业务收入增长率(%)": "rev_growth",
    "净利润增长率(%)": "profit_growth",
    "资产负债率(%)": "debt_ratio",
    "应收账款周转率(次)": "ar_turnover",
}
VAL_COLS = {
    "数据日期": "date", "PE(TTM)": "pe_ttm", "市净率": "pb",
    "市销率": "ps_ttm", "市现率": "pcf", "PEG值": "peg",
}

def get_union_universe():
    fcf = pd.read_csv(f"{WORK}/fcf_cache.csv", sep='\t')
    prof = pd.read_csv(f"{WORK}/profit_cache.csv", sep='\t')
    fcf["code"] = fcf["code"].astype(str).str.zfill(6)
    prof["code"] = prof["code"].astype(str).str.zfill(6)
    union = set()
    for y in [2019, 2021, 2022, 2023, 2024, 2025, 2026]:
        union |= build_dynamic_universe(y, fcf, prof)
    return sorted(union)

def load_done(path, col="code"):
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t', dtype={col: str})
        return df, set(df[col].unique())
    return None, set()

def main():
    codes = get_union_universe()
    print(f"[+] union股票池: {len(codes)}只", flush=True)

    # ---------- 估值 ----------
    val_df, val_done = load_done(VAL_CACHE)
    todo = [c for c in codes if c not in val_done]
    print(f"[估值] 待抓 {len(todo)}/{len(codes)}", flush=True)
    buf = []
    for i, code in enumerate(todo):
        for attempt in range(3):
            try:
                df = ak.stock_value_em(symbol=code)
                df = df.rename(columns=VAL_COLS)[list(VAL_COLS.values())]
                df["code"] = code
                buf.append(df)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [估值] {code} 失败: {str(e)[:60]}", flush=True)
                time.sleep(1.5)
        time.sleep(0.35)
        if (i + 1) % 25 == 0 or i == len(todo) - 1:
            if buf:
                new = pd.concat(buf, ignore_index=True)
                if os.path.exists(VAL_CACHE):
                    new.to_csv(VAL_CACHE, sep='\t', index=False, mode='a', header=False)
                else:
                    new.to_csv(VAL_CACHE, sep='\t', index=False)
                buf = []
            print(f"  [估值] 进度 {i+1}/{len(todo)}", flush=True)

    # ---------- 财务指标 ----------
    fin_df, fin_done = load_done(FIN_CACHE)
    todo = [c for c in codes if c not in fin_done]
    print(f"[财务] 待抓 {len(todo)}/{len(codes)}", flush=True)
    buf = []
    for i, code in enumerate(todo):
        for attempt in range(3):
            try:
                df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2013")
                keep = [c for c in FIN_COLS if c in df.columns]
                df = df[keep].rename(columns=FIN_COLS)
                df["code"] = code
                buf.append(df)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [财务] {code} 失败: {str(e)[:60]}", flush=True)
                time.sleep(1.5)
        time.sleep(0.35)
        if (i + 1) % 25 == 0 or i == len(todo) - 1:
            if buf:
                new = pd.concat(buf, ignore_index=True)
                if os.path.exists(FIN_CACHE):
                    new.to_csv(FIN_CACHE, sep='\t', index=False, mode='a', header=False)
                else:
                    new.to_csv(FIN_CACHE, sep='\t', index=False)
                buf = []
            print(f"  [财务] 进度 {i+1}/{len(todo)}", flush=True)

    # 汇总
    for path, name in [(VAL_CACHE, "估值"), (FIN_CACHE, "财务")]:
        df = pd.read_csv(path, sep='\t', dtype={"code": str})
        print(f"[✓] {name}缓存: {df.shape}, {df['code'].nunique()}只", flush=True)

if __name__ == "__main__":
    main()
