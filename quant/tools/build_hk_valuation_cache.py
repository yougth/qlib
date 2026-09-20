#!/usr/bin/env python3
"""
build_hk_valuation_cache.py — 港股估值缓存构建 (修复版)
================================================================================
从 akshare 获取港股利润表+资产负债表, 用 qfq 前复权价(真实价) 计算 PE/PB/PCF/PS,
生成与 A 股 valuation_cache.csv 同口径的港股估值数据。

关键修复:
  1. 真实价: 用腾讯 qfq(前复权)价 —— qfq 以最新价为基准, 历史qfq价 = 当时真实价
     (腾讯 2024-06-03 qfq=365.2, 真实价合理; 而 hfq 后复权价=2104 虚高, 不能用于估值)
  2. PIT: 年报Y 必须 Y+1年5月1日后才可用 (严格 no-lookahead)
     - 每年只有 12-31 年报, 次年 5月1日 起可用
     - 上半年(1-4月)用 Y-2 年报, 5月起用 Y-1 年报

公式:
  PE  = qfq_price / EPS       (股东应占溢利 / 总股本)
  PB  = qfq_price / BPS       (总权益 / 总股本)
  PCF = qfq_price / OCFPS     (经营业务现金净额 / 总股本)
  PS  = qfq_price / SPS       (营业额 / 总股本)
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import urllib.request

QUANT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("QUANT_DATA_DIR") or os.path.join(QUANT_DIR, "..", "..")
HK_VAL_CACHE = os.path.join(DATA_DIR, "valuation_cache_hk.csv")

HEADERS = {"Referer": "https://gu.qq.com", "User-Agent": "Mozilla/5.0"}

# 从 fetch_hk_data.py 导入港股列表
sys.path.insert(0, QUANT_DIR)
from tools.fetch_hk_data import HK_STOCKS, clean_code

def fetch_qfq_kline(code, start="2014-01-01", end=None):
    """腾讯港股前复权(qfq)K线, 用于估值真实价"""
    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
    code5 = clean_code(code)
    from datetime import datetime, timedelta
    all_rows = []
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    seg_start = s
    while seg_start < e:
        seg_end = min(seg_start + timedelta(days=730), e)
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?"
               f"param=hk{code5},day,{seg_start.strftime('%Y-%m-%d')},"
               f"{seg_end.strftime('%Y-%m-%d')},640,qfq")
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            inner = data.get("data", {}).get(f"hk{code5}", {})
            kline = inner.get("qfqday", [])
            if kline:
                all_rows.extend(kline)
        except Exception:
            pass
        time.sleep(0.2)
        seg_start = seg_end + timedelta(days=1)
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=["date", "open", "close", "high", "low",
                                          "volume", "_e", "_f", "_g"])
    df = df.drop_duplicates(subset=["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()

def fetch_hk_financials(code):
    """获取港股利润表+资产负债表+现金流量表年度数据"""
    import akshare as ak
    result = {}
    try:
        df_p = ak.stock_financial_hk_report_em(stock=code, symbol="利润表", indicator="年度")
        for _, r in df_p.iterrows():
            rd = str(r["REPORT_DATE"])[:10]
            if r["STD_ITEM_NAME"] == "股东应占溢利":
                result.setdefault(rd, {})["net_profit"] = float(r["AMOUNT"])
            if r["STD_ITEM_NAME"] == "营业额":
                result.setdefault(rd, {})["revenue"] = float(r["AMOUNT"])
        df_b = ak.stock_financial_hk_report_em(stock=code, symbol="资产负债表", indicator="年度")
        for _, r in df_b.iterrows():
            rd = str(r["REPORT_DATE"])[:10]
            if r["STD_ITEM_NAME"] == "总权益":
                result.setdefault(rd, {})["total_equity"] = float(r["AMOUNT"])
        df_c = ak.stock_financial_hk_report_em(stock=code, symbol="现金流量表", indicator="年度")
        for _, r in df_c.iterrows():
            rd = str(r["REPORT_DATE"])[:10]
            if r["STD_ITEM_NAME"] == "经营业务现金净额":
                result.setdefault(rd, {})["ocf"] = float(r["AMOUNT"])
    except Exception as e:
        print(f"  [FAIL] {code}: {e}", flush=True)
    return result

def get_hk_shares(code):
    """获取港股总股本(股)"""
    import akshare as ak
    try:
        df = ak.stock_hk_financial_indicator_em(symbol=code)
        if df is not None and len(df) > 0:
            for col in df.columns:
                if '已发行股本' in col and 'H股' not in col:
                    val = df.iloc[0][col]
                    if pd.notna(val) and float(val) > 0:
                        return float(val)
    except Exception:
        pass
    return None

def latest_available_report(fins, date):
    """PIT: 找到 date 时最新可用的年报.
    年报Y 在 Y+1年5月1日起可用. 严格 no-lookahead."""
    d = pd.Timestamp(date)
    best = None
    best_year = 0
    for rd, data in fins.items():
        rd_date = pd.Timestamp(rd)
        rd_year = rd_date.year
        # 年报截止12-31, 次年5月1日披露
        available_from = pd.Timestamp(f"{rd_year+1}-05-01")
        if d >= available_from and rd_year > best_year:
            best = data
            best_year = rd_year
    return best

def build_hk_valuation():
    stocks = [(clean_code(c), n) for c, n in HK_STOCKS]
    seen = set()
    stocks = [(c, n) for c, n in stocks if c not in seen and not seen.add(c)]
    print(f"[港股估值] {len(stocks)} 只港股", flush=True)

    # 日历
    qlib_data = os.path.join(os.path.dirname(QUANT_DIR), "data_cache", "qlib_cn_tencent")
    cal = open(os.path.join(qlib_data, "calendars", "day.txt")).read().strip().split()
    cal_dates = pd.to_datetime(cal)

    all_rows = []
    ok = 0
    for i, (code, name) in enumerate(stocks):
        fins = fetch_hk_financials(code)
        shares = get_hk_shares(code)
        price_series = fetch_qfq_kline(code)
        if not fins or not shares or shares <= 0 or price_series is None:
            continue

        code_rows = []
        for date_str in cal:
            d = pd.Timestamp(date_str)
            if d not in price_series.index:
                continue
            real_price = price_series.loc[d]
            if not np.isfinite(real_price) or real_price <= 0:
                continue

            data = latest_available_report(fins, d)
            if not data:
                continue

            np_val = data.get("net_profit")
            eq_val = data.get("total_equity")
            ocf_val = data.get("ocf")
            rev_val = data.get("revenue")

            eps = np_val / shares if np_val and shares > 0 else None
            bps = eq_val / shares if eq_val and shares > 0 else None
            ocfps = ocf_val / shares if ocf_val and shares > 0 else None
            sps = rev_val / shares if rev_val and shares > 0 else None

            pe = real_price / eps if eps and eps != 0 else np.nan
            pb = real_price / bps if bps and bps != 0 else np.nan
            pcf = real_price / ocfps if ocfps and ocfps != 0 else np.nan
            ps = real_price / sps if sps and sps != 0 else np.nan

            if any(np.isfinite([pe, pb, ps, pcf])):
                code_rows.append({
                    "date": date_str,
                    "pe_ttm": round(pe, 4) if np.isfinite(pe) else "",
                    "pb": round(pb, 4) if np.isfinite(pb) else "",
                    "ps_ttm": round(ps, 4) if np.isfinite(ps) else "",
                    "pcf": round(pcf, 4) if np.isfinite(pcf) else "",
                    "peg": "",
                    "code": code,
                })

        if code_rows:
            all_rows.extend(code_rows)
            ok += 1
            if (i + 1) % 10 == 0 or i == len(stocks) - 1:
                print(f"  [{i+1}/{len(stocks)}] ok={ok} rows={len(all_rows)}", flush=True)
        time.sleep(0.2)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(HK_VAL_CACHE, sep="\t", index=False)
        print(f"\n[完成] valuation_cache_hk.csv: {df.shape}, "
              f"{df['code'].nunique()}只, {df['date'].min()}~{df['date'].max()}", flush=True)
    else:
        print("\n[WARNING] 无有效港股估值数据", flush=True)

if __name__ == "__main__":
    build_hk_valuation()
