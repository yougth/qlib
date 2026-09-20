#!/usr/bin/env python3
"""fetch_industry —— 拉取池内全部股票行业分类, 用于金融股豁免识别
输出: data_cache/industry_map.csv (code, industry, name)"""
import os
import sys
import json
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

from core.universe import build_windows, build_dynamic_universe, load_pit_caches

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "data_cache", "industry_map.csv")
HDR = {"Referer": "https://emweb.securities.eastmoney.com/",
       "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def get_json(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            time.sleep(1.0 + i)
    return None


def fetch_a_industry(code):
    exch = "SH" if str(code).startswith(("6", "9")) else "SZ"
    sec = f"{code}.{exch}"
    d = get_json(f"https://datacenter.eastmoney.com/securities/api/data/get?"
                 f"type=RPT_F10_BASIC_ORGINFO&sty=ALL"
                 f"&filter=(SECUCODE%3D%22{sec}%22)&p=1&ps=10&source=HSF10&client=PC") or {}
    data = ((d.get("result") or {}).get("data") or [{}])[0]
    return data.get("EM2016", "") or "", data.get("SECURITY_NAME_ABBR", "") or ""


def fetch_hk_industry(code):
    sec = f"{int(code):05d}.HK"
    d = get_json(f"https://datacenter.eastmoney.com/securities/api/data/get?"
                f"type=RPT_HKF10_INFO_ORGPROFILE&sty=ALL"
                f"&filter=(SECUCODE%3D%22{sec}%22)&p=1&ps=10&source=HSF10&client=PC") or {}
    data = ((d.get("result") or {}).get("data") or [{}])[0]
    return data.get("BELONG_INDUSTRY", "") or "", data.get("SECURITY_NAME_ABBR", "") or ""


def main():
    fcf_df, profit_df = load_pit_caches()
    codes = set()
    for win in build_windows():
        codes.update(build_dynamic_universe(win["year"], fcf_df, profit_df))
    a = sorted(c for c in codes if len(str(c)) == 6)
    hk = sorted(c for c in codes if len(str(c)) == 5 and str(c).isdigit())
    print(f"A股 {len(a)} + 港股 {len(hk)}", flush=True)

    rows = []
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, dtype={"code": str})
        rows = old.to_dict("records")
        done = set(old["code"])
        print(f"已有 {len(done)} 只, 增量拉取", flush=True)
    else:
        done = set()

    for i, c in enumerate(a + hk):
        if c in done:
            continue
        try:
            ind, name = (fetch_a_industry(c) if len(c) == 6 else fetch_hk_industry(c))
        except Exception:
            ind, name = "", ""
        rows.append({"code": c, "industry": ind, "name": name})
        time.sleep(0.08)
        if (len(rows)) % 100 == 0:
            print(f"  {len(rows)}/{len(a)+len(hk)}", flush=True)

    df = pd.DataFrame(rows).drop_duplicates(subset=["code"])
    df.to_csv(OUT, index=False)
    print(f"[+] {OUT}: {len(df)} 只")
    fin_kw = ("银行", "保险", "证券", "信托", "多元金融", "金融")
    fin = df[df["industry"].fillna("").str.contains("|".join(fin_kw))]
    print(f"识别金融股 {len(fin)} 只:")
    print(fin.to_string(index=False))


if __name__ == "__main__":
    main()
