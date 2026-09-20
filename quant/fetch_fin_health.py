#!/usr/bin/env python3
"""
fetch_fin_health —— 拉取资产负债表/利润表关键科目 (PIT, 含公告日)
================================================================================
目的: 为"十年双正"底池增加财务健康过滤 (ROIC / 利息保障 / 资产负债率) 提供数据。
数据源 (与项目 tencent/东财缓存一致的公开接口):
  - A股: 东财 F10 zcfzbAjaxNew / lrbAjaxNew (宽表, 含 NOTICE_DATE 公告日)
  - 港股: 东财数据中心 RPT_HKF10_FN_BALANCE / RPT_HKF10_FN_INCOME (长表)
输出: data_cache/fin_health_pit.parquet
  code | year | notice_date | ta | tl | te | int_debt | ebit | tax | ni | int_exp | fin_type
  (ta=总资产 tl=总负债 te=股东权益 int_debt=有息负债 ebit=息税前利润
   tax=所得税 ni=归母净利 int_exp=利息/融资费用 fin_type: 0非金融 2银行 3保险 1证券)
PIT 原则: A股用 NOTICE_DATE 精确公告日; 港股无公告日, 用保守规则 Y年报→Y+1-05-01 可用。
"""
import os
import sys
import json
import time
import urllib.request
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from core.universe import build_windows, build_dynamic_universe, load_pit_caches

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "data_cache", "fin_health_pit.parquet")
YEARS = list(range(2016, 2026))          # 2016~2025 年报 (过滤仅需 Y-4..Y-2)
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


def f(x):
    return float(x) if x not in (None, "", "None") else None


# ---------------- A股 ----------------
def parse_a_pair(code, bs, inc):
    """一对 (资产负债表列表, 利润表列表) → 记录列表"""
    out = []
    inc_by_date = {r["REPORT_DATE"][:4]: r for r in inc}
    for b in bs:
        yr = b["REPORT_DATE"][:4]
        i = inc_by_date.get(yr, {})
        ta, tl = f(b.get("TOTAL_ASSETS")), f(b.get("TOTAL_LIABILITIES"))
        te = f(b.get("TOTAL_EQUITY")) or f(b.get("TOTAL_PARENT_EQUITY"))
        int_debt = sum(v for v in (f(b.get("SHORT_LOAN")), f(b.get("LONG_LOAN")),
                                   f(b.get("BOND_PAYABLE")),
                                   f(b.get("NONCURRENT_LIAB_1YEAR"))) if v)
        tp, it = f(i.get("TOTAL_PROFIT")), f(i.get("INCOME_TAX"))
        ie = f(i.get("FE_INTEREST_EXPENSE")) or f(i.get("INTEREST_EXPENSE")) \
             or f(i.get("FINANCE_EXPENSE"))
        ebit = (tp + ie) if (tp is not None and ie) else tp
        out.append({"code": str(code), "year": int(yr),
                    "notice_date": b.get("NOTICE_DATE", "")[:10] or None,
                    "ta": ta, "tl": tl, "te": te, "int_debt": int_debt,
                    "ebit": ebit, "tax": it,
                    "ni": f(i.get("NETPROFIT")), "int_exp": ie,
                    "fin_type": None})
    return out


def fetch_a_stock(code, years, ctype_hint=None):
    """东财 F10 宽表: companyType 4通用→3保险→2银行→1证券, 每批最多5年"""
    exch = "SH" if str(code).startswith(("6", "9")) else "SZ"
    em = f"{exch}{code}"
    out = []
    ctypes = [ctype_hint] if ctype_hint else [4, 3, 2, 1]
    got_ctype = None
    for ctype in ctypes:
        ok_any = False
        for y0 in range(0, len(years), 5):          # 接口每请求最多返回5期
            batch = years[y0:y0 + 5]
            dates = ",".join(f"{y}-12-31" for y in batch)
            rows = []
            for rep in ("zcfzb", "lrb"):
                url = (f"https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/"
                       f"{rep}AjaxNew?companyType={ctype}&reportDateType=1&reportType=1"
                       f"&dates={urllib.parse.quote(dates)}&code={em}")
                d = get_json(url)
                data = (d or {}).get("data") or []
                if rep == "zcfzb" and not data:
                    break                # 该 companyType 无效
                rows.append(data)
                time.sleep(0.12)
            if len(rows) == 2:
                recs = parse_a_pair(code, rows[0], rows[1])
                for r in recs:
                    r["fin_type"] = 0 if ctype == 4 else ctype
                out.extend(recs)
                ok_any = True
                got_ctype = ctype
            if not ok_any:
                break
        if ok_any:
            break                        # 找到有效 companyType
    return out, got_ctype


# ---------------- 港股 ----------------
HK_BS = {"总资产": "ta", "总负债": "tl", "总权益": "te", "短期贷款": "s_loan",
         "长期贷款": "l_loan"}
HK_INC = {"除税前溢利": "ebt", "税项": "tax", "股东应占溢利": "ni", "融资成本": "fin_cost"}


def fetch_hk_stock(code):
    """东财数据中心长表 → 透视; 年报(REPORT_DATE=12-31); 无公告日用保守 Y+1-05-01"""
    sec = f"{int(code):05d}.HK"
    out = {}
    for typ, items in (("RPT_HKF10_FN_BALANCE", HK_BS),
                       ("RPT_HKF10_FN_INCOME", HK_INC)):
        url = (f"https://datacenter.eastmoney.com/securities/api/data/get?type={typ}"
               f"&sty=ALL&filter=(SECUCODE%3D%22{sec}%22)&p=1&ps=3000"
               f"&source=HSF10&client=PC")
        d = get_json(url)
        data = ((d or {}).get("result") or {}).get("data") or []
        for r in data:
            yr = r["REPORT_DATE"][:4]
            if r["REPORT_DATE"][5:10] != "12-31" or r["ITEM_NAME"] not in items:
                continue
            out.setdefault(yr, {})[items[r["ITEM_NAME"]]] = f(r.get("AMOUNT"))
        time.sleep(0.12)
    rows = []
    for yr, v in out.items():
        te = v.get("te")
        int_debt = sum(x for x in (v.get("s_loan"), v.get("l_loan")) if x) or None
        ebt, fc = v.get("ebt"), v.get("fin_cost")
        ebit = (ebt + fc) if (ebt is not None and fc) else ebt
        rows.append({"code": str(code), "year": int(yr), "notice_date": None,
                     "ta": v.get("ta"), "tl": v.get("tl"), "te": te,
                     "int_debt": int_debt, "ebit": ebit, "tax": v.get("tax"),
                     "ni": v.get("ni"), "int_exp": fc, "fin_type": 0})
    return rows


def main():
    fcf_df, profit_df = load_pit_caches()
    windows = build_windows()
    all_codes = set()
    for win in windows:
        all_codes.update(build_dynamic_universe(win["year"], fcf_df, profit_df))
    a_codes = sorted(c for c in all_codes if len(str(c)) == 6)
    hk_codes = sorted(c for c in all_codes if len(str(c)) == 5 and str(c).isdigit())
    print(f"池并集: A股 {len(a_codes)} + 港股 {len(hk_codes)}", flush=True)

    # 增量模式: 已有缓存的 (code, year) 跳过; A股只补缺失年份
    old = pd.read_parquet(OUT) if os.path.exists(OUT) else pd.DataFrame(
        columns=["code", "year"])
    have = {(str(r.code), int(r.year)) for r in old.itertuples()}
    ctype_map = (old.dropna(subset=["fin_type"]).groupby("code")["fin_type"]
                 .last().to_dict()) if len(old) else {}

    records = []
    need_a = {c: [y for y in YEARS if (c, y) not in have] for c in a_codes}
    need_a = {c: ys for c, ys in need_a.items() if ys}
    print(f"A股需补拉 {len(need_a)} 只 (缺年份总计 "
          f"{sum(len(v) for v in need_a.values())} 年)", flush=True)
    for i, c in enumerate(a_codes):
        ys = need_a.get(c)
        if not ys:
            continue
        rec, ct = fetch_a_stock(c, ys, ctype_hint=ctype_map.get(c))
        records.extend(rec)
        if (i + 1) % 50 == 0:
            print(f"  A股 {i+1}/{len(a_codes)}, 补拉累计 {len(records)} 行", flush=True)
    n_a = len(records)
    print(f"A股补拉完成: {n_a} 行", flush=True)

    for i, c in enumerate(hk_codes):
        if any((c, y) in have for y in YEARS):
            continue                       # 港股长表一次拉全, 已有则跳过
        records.extend(fetch_hk_stock(c))
        if (i + 1) % 20 == 0:
            print(f"  港股 {i+1}/{len(hk_codes)}, 累计 {len(records)} 行", flush=True)
    print(f"港股完成: +{len(records)-n_a} 行", flush=True)

    df = pd.concat([old, pd.DataFrame(records)], ignore_index=True)
    df = df.drop_duplicates(subset=["code", "year"], keep="first")
    df = df.sort_values(["code", "year"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    cov = df.groupby("code").size()
    print(f"\n[+] 保存 {OUT}: {len(df)} 行, {df['code'].nunique()} 只, "
          f"年报数 中位 {cov.median():.0f} (2010~2025)", flush=True)
    # 覆盖率体检
    miss_a = [c for c in a_codes if c not in set(df['code'])]
    miss_hk = [c for c in hk_codes if c not in set(df['code'])]
    print(f"缺失: A股 {len(miss_a)} 只, 港股 {len(miss_hk)} 只", flush=True)
    if miss_a:
        print("  A股缺:", miss_a[:20], flush=True)
    if miss_hk:
        print("  港股缺:", miss_hk[:20], flush=True)


if __name__ == "__main__":
    main()
