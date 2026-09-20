#!/usr/bin/env python3
"""tools.fetch_lof —— 方向2 LOF折溢价 数据抓取
================================================================================
数据源: 2026-09-18 东财push2行情域名对本机断连, 全面切换新浪源。
  1. fund_etf_category_sina("LOF基金") 列表(含成交额) → data/lof_list.csv
  2. 挑成交额前N只, 逐只抓:
     · fund_etf_hist_sina 场内日线(全历史) → data/lof/{code}_px.csv
     · fund_open_fund_info_em 单位净值(东财fund域名正常) → data/lof/{code}_nav.csv
PIT注意: 净值T日晚披露 → 用T-1净值算折价, 回测按T+1交易。
幂等: 已有文件跳过; 每股限速0.3s。
"""
import os
import sys
import time

sys.path = [p for p in sys.path if "Qoder" not in p]
import akshare as ak  # noqa: E402

DATA = "/Users/11164591/Documents/Qoder目录/qlib/new_quant/data"
LOF_DIR = os.path.join(DATA, "lof")
TOP_N = 80


def fetch_list():
    p = os.path.join(DATA, "lof_list.csv")
    if os.path.exists(p) and os.path.getsize(p) > 5000:
        print("[skip] LOF列表已存在", flush=True)
        return
    last = None
    for i in range(5):
        try:
            df = ak.fund_etf_category_sina(symbol="LOF基金")
            df.to_csv(p, index=False)
            print(f"[+] LOF列表 {df.shape} → {p}", flush=True)
            return
        except Exception as e:
            last = e
            print(f"  [retry {i+1}] {e!r}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"列表抓取失败: {last!r}")


def fetch_one(sym):
    """sym 为带前缀代码, 如 sz161725"""
    code = sym[-6:]
    px_p = os.path.join(LOF_DIR, f"{code}_px.csv")
    nav_p = os.path.join(LOF_DIR, f"{code}_nav.csv")
    if not (os.path.exists(px_p) and os.path.getsize(px_p) > 200):
        h = ak.fund_etf_hist_sina(symbol=sym)
        h.to_csv(px_p, index=False)
        time.sleep(0.3)
    if not (os.path.exists(nav_p) and os.path.getsize(nav_p) > 200):
        n = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        n.to_csv(nav_p, index=False)
        time.sleep(0.3)


def main():
    os.makedirs(LOF_DIR, exist_ok=True)
    fetch_list()
    import pandas as pd
    lst = pd.read_csv(os.path.join(DATA, "lof_list.csv"), dtype={"代码": str})
    amt_col = next((c for c in lst.columns if "成交额" in c), None)
    lst = lst.sort_values(amt_col, ascending=False)
    syms = lst["代码"].astype(str).head(TOP_N).tolist()
    print(f"[LOF] 抓取成交额前{len(syms)}只: {syms[:8]}...", flush=True)
    ok = 0
    for i, s in enumerate(syms):
        try:
            fetch_one(s)
            ok += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(syms)}] ok={ok}", flush=True)
        except Exception as e:
            print(f"  [WARN] {s}: {e!r}", flush=True)
    print(f"[DONE] LOF抓取完成 {ok}/{len(syms)}", flush=True)


if __name__ == "__main__":
    main()