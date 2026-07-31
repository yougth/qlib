"""
存活偏差定量检查
================
问题: fcf_cache.csv 只含当前上市公司, 2019-2026年间退市的股票被系统性排除。
方法: 拉全部退市股(沪+深), 用与 build_dynamic_universe 完全相同的规则
      (FCF=经营现金流-资本开支, 10年窗口[Y-11,Y-2], 覆盖率>=80%且全为正,
       净利润2016起全为正), 检查每个回测年有多少退市股"本应入池"。
输出: 每个回测年被遗漏的股票清单 → 若接近0, 则存活偏差对该策略影响可忽略;
      若数量可观, 需补行情重算E0。
"""
import sys, time
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")
import pandas as pd
import numpy as np
import akshare as ak

WORK = "/Users/11164591/Documents/Qoder目录"
YEARS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]

def get_delist_list():
    sh = ak.stock_info_sh_delist()
    sh = sh.rename(columns={"公司代码": "code", "公司简称": "name",
                            "上市日期": "list_date", "暂停上市日期": "delist_date"})
    sz = ak.stock_info_sz_delist(symbol="终止上市公司")
    sz = sz.rename(columns={"证券代码": "code", "证券简称": "name",
                            "上市日期": "list_date", "终止上市日期": "delist_date"})
    df = pd.concat([sh[["code", "name", "delist_date"]], sz[["code", "name", "delist_date"]]])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["delist_date"] = pd.to_datetime(df["delist_date"], errors="coerce")
    df = df.dropna(subset=["delist_date"])
    # 只关心2018年后退市的 (更早退市不影响2019+回测池: 2019池要求2008-2017双正+在市)
    df = df[df["delist_date"] >= "2018-01-01"].drop_duplicates(subset=["code"])
    # 只要主板/创业板A股 (0/3/6开头)
    df = df[df["code"].str[0].isin(["0", "3", "6"])]
    return df.reset_index(drop=True)

def fetch_annual_fin(code):
    """返回 {year: (net_profit, fcf)}; 数据不可得返回None"""
    prefix = "sh" if code.startswith("6") else "sz"
    try:
        inc = ak.stock_financial_report_sina(stock=f"{prefix}{code}", symbol="利润表")
        time.sleep(0.3)
        cf = ak.stock_financial_report_sina(stock=f"{prefix}{code}", symbol="现金流量表")
        time.sleep(0.3)
    except Exception:
        return None
    if inc is None or cf is None or len(inc) == 0 or len(cf) == 0:
        return None
    out = {}
    inc = inc.copy(); cf = cf.copy()
    inc["报告日"] = inc["报告日"].astype(str)
    cf["报告日"] = cf["报告日"].astype(str)
    inc_y = inc[inc["报告日"].str.endswith("1231")]
    cf_y = cf[cf["报告日"].str.endswith("1231")]
    for _, r in inc_y.iterrows():
        y = int(r["报告日"][:4])
        np_v = pd.to_numeric(r.get("净利润"), errors="coerce")
        out.setdefault(y, [np.nan, np.nan])
        out[y][0] = np_v
    for _, r in cf_y.iterrows():
        y = int(r["报告日"][:4])
        ocf = pd.to_numeric(r.get("经营活动产生的现金流量净额"), errors="coerce")
        capex = pd.to_numeric(r.get("购建固定资产、无形资产和其他长期资产所支付的现金"), errors="coerce")
        if pd.isna(capex):
            capex = 0.0
        out.setdefault(y, [np.nan, np.nan])
        out[y][1] = (ocf - capex) if pd.notna(ocf) else np.nan
    return {y: (v[0], v[1]) for y, v in out.items()}

def passes_rule(fin, bt_year):
    """与build_dynamic_universe完全一致的规则"""
    fcf_years = list(range(bt_year - 11, bt_year - 1))  # [Y-11, Y-2]
    fcf_vals = [fin[y][1] for y in fcf_years if y in fin and pd.notna(fin[y][1])]
    if len(fcf_vals) < len(fcf_years) * 0.8:
        return False
    if any(v <= 0 for v in fcf_vals):
        return False
    profit_start = max(bt_year - 11, 2016)
    profit_years = list(range(profit_start, bt_year - 1))
    if profit_years:
        p_vals = [fin[y][0] for y in profit_years if y in fin and pd.notna(fin[y][0])]
        if len(p_vals) < len(profit_years) * 0.8:
            return False
        if any(v <= 0 for v in p_vals):
            return False
    return True

def main():
    delist = get_delist_list()
    print(f"[+] 2018年后退市A股: {len(delist)}只", flush=True)

    missed = {y: [] for y in YEARS}
    checked = 0
    for _, row in delist.iterrows():
        code, name, ddate = row["code"], row["name"], row["delist_date"]
        fin = fetch_annual_fin(code)
        checked += 1
        if checked % 20 == 0:
            print(f"  进度 {checked}/{len(delist)}", flush=True)
        if fin is None or len(fin) == 0:
            continue
        for y in YEARS:
            # 该股在回测年y还在上市(年初未退市)才有意义
            if ddate < pd.Timestamp(f"{y}-01-01"):
                continue
            if passes_rule(fin, y):
                missed[y].append(f"{code}({name})")

    print(f"\n{'='*70}", flush=True)
    print("存活偏差定量检查结果 (本应入池但被fcf_cache遗漏的退市股):", flush=True)
    total = 0
    for y in YEARS:
        print(f"  {y}年池: {len(missed[y])}只  {missed[y]}", flush=True)
        total += len(missed[y])
    print(f"{'='*70}", flush=True)
    rows = [{"year": y, "n_missed": len(missed[y]), "codes": ";".join(missed[y])} for y in YEARS]
    pd.DataFrame(rows).to_csv(f"{WORK}/qlib/survivorship_check.csv", sep='\t', index=False)
    print("[+] 已保存 survivorship_check.csv", flush=True)

if __name__ == "__main__":
    main()
