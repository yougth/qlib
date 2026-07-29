"""
PIT阶段1 (只读, 不写qlib): 修正股票池存活偏差
=====================================================
1. 拉沪+深退市清单, 保留 终止上市日>=2018 的
2. 新浪财报补退市股 年度FCF(经营现金流净额-购建固定资产) 与 归母净利润 (与东财口径一致)
3. 生成PIT增广缓存 fcf_cache_pit.csv / profit_cache_pit.csv (= 存量东财 + 退市新浪)
4. 逐年用 build_dynamic_universe(window=year-11..year-2) 计算:
   本应进池 = 双正通过 且 该年首次调仓日时仍在市(终止上市日 >= 调仓日)
   → 打印各年新增的退市候选, 量化存活偏差范围
"""
import os, sys, warnings, time
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA = "/Users/11164591/Documents/Qoder目录"
DELIST_CACHE = f"{DATA}/qlib/data_cache/delist_financials.parquet"
os.makedirs(f"{DATA}/qlib/data_cache", exist_ok=True)


def get_delist_list():
    frames = []
    # 深市
    try:
        sz = ak.stock_info_sz_delist(symbol="终止上市公司")
        sz = sz.rename(columns={"证券代码": "code", "证券简称": "name",
                                "上市日期": "list_date", "终止上市日期": "delist_date"})
        frames.append(sz[["code", "name", "list_date", "delist_date"]])
        print(f"  深市退市 {len(sz)}", flush=True)
    except Exception as e:
        print(f"  深市失败 {e}", flush=True)
    # 沪市: 尝试多种签名
    sh = None
    for kw in [{}, {"symbol": "全部"}, {"symbol": "沪市"}]:
        try:
            sh = ak.stock_info_sh_delist(**kw)
            if sh is not None and len(sh):
                break
        except Exception:
            continue
    if sh is not None and len(sh):
        ren = {}
        for c in sh.columns:
            if "代码" in c: ren[c] = "code"
            elif "简称" in c or "名称" in c: ren[c] = "name"
            elif "终止" in c or "暂停" in c or "摘牌" in c: ren[c] = "delist_date"
            elif "上市日期" in c: ren[c] = "list_date"
        sh = sh.rename(columns=ren)
        print(f"  沪市原始列映射 {ren}", flush=True)
        keep = [c for c in ["code", "name", "list_date", "delist_date"] if c in sh.columns]
        frames.append(sh[keep])
        print(f"  沪市退市 {len(sh)} 列{list(sh.columns)}", flush=True)
    else:
        print("  沪市退市清单获取失败(仅深市)", flush=True)
    d = pd.concat(frames, ignore_index=True)
    d["code"] = d["code"].astype(str).str.zfill(6)
    d["delist_date"] = pd.to_datetime(d["delist_date"], errors="coerce")
    d = d.dropna(subset=["delist_date"])
    d = d[d["delist_date"] >= pd.Timestamp("2018-01-01")].drop_duplicates("code")
    return d.reset_index(drop=True)


def sina_prefix(code):
    c = str(code).zfill(6)
    return ("sh" if c.startswith("6") else "sz") + c


def fetch_fin(code, name):
    stock = sina_prefix(code)
    rows = []
    try:
        cf = ak.stock_financial_report_sina(stock=stock, symbol="现金流量表")
        pf = ak.stock_financial_report_sina(stock=stock, symbol="利润表")
    except Exception:
        return code, name, None
    if cf is None or pf is None or "报告日" not in cf.columns or "报告日" not in pf.columns:
        return code, name, None

    def annual(df):
        df = df.copy(); df["报告日"] = df["报告日"].astype(str)
        return df[df["报告日"].str.endswith("1231")]
    cf, pf = annual(cf), annual(pf)
    OP, CX = "经营活动产生的现金流量净额", "购建固定资产、无形资产和其他长期资产所支付的现金"
    NP = "归属于母公司所有者的净利润"
    pf_map = {str(r["报告日"])[:4]: r.get(NP) for _, r in pf.iterrows()}
    for _, r in cf.iterrows():
        y = str(r["报告日"])[:4]
        try:
            op = float(r.get(OP)); 
        except (TypeError, ValueError):
            continue
        try:
            cx = float(r.get(CX))
        except (TypeError, ValueError):
            cx = 0.0
        try:
            np_ = float(pf_map.get(y)) if pf_map.get(y) is not None else None
        except (TypeError, ValueError):
            np_ = None
        rows.append({"code": code, "name": name, "year": int(y),
                     "netcash_operate": op, "capex": cx, "fcf": op - cx, "net_profit": np_})
    return code, name, rows


def main():
    print("=" * 60 + "\n  PIT阶段1: 退市股财务补齐 (存活偏差修正)\n" + "=" * 60, flush=True)
    delist = get_delist_list()
    print(f"\n2018年后退市: {len(delist)} 只", flush=True)

    if os.path.exists(DELIST_CACHE):
        fin = pd.read_parquet(DELIST_CACHE)
        print(f"  已有退市财务缓存 {fin['code'].nunique()} 只", flush=True)
    else:
        allrows = []
        todo = list(delist[["code", "name"]].itertuples(index=False, name=None))
        print(f"  新浪财报拉取 {len(todo)} 只...", flush=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(fetch_fin, c, n): c for c, n in todo}
            done = ok = 0
            for fu in as_completed(futs):
                c, n, rows = fu.result(); done += 1
                if rows: allrows.extend(rows); ok += 1
                if done % 20 == 0 or done == len(todo):
                    print(f"    {done}/{len(todo)} (有财务{ok})", flush=True)
        fin = pd.DataFrame(allrows)
        fin.to_parquet(DELIST_CACHE)
        print(f"  退市股有财务 {fin['code'].nunique()} 只, {len(fin)} 行", flush=True)

    # 构建增广缓存
    fcf0 = pd.read_csv(f"{DATA}/fcf_cache.csv", sep='\t'); fcf0["code"] = fcf0["code"].astype(str).str.zfill(6)
    prof0 = pd.read_csv(f"{DATA}/profit_cache.csv", sep='\t'); prof0["code"] = prof0["code"].astype(str).str.zfill(6)
    fin["code"] = fin["code"].astype(str).str.zfill(6)
    fcf_add = fin[["code", "name", "year", "netcash_operate", "capex", "fcf"]].dropna(subset=["fcf"])
    prof_add = fin[["code", "name", "year", "net_profit"]].dropna(subset=["net_profit"])
    # 去重: 退市股不应已在存量(存量是在市), 直接并
    fcf_pit = pd.concat([fcf0, fcf_add], ignore_index=True).drop_duplicates(["code", "year"], keep="first")
    prof_pit = pd.concat([prof0, prof_add], ignore_index=True).drop_duplicates(["code", "year"], keep="first")
    fcf_pit.to_csv(f"{DATA}/fcf_cache_pit.csv", sep='\t', index=False)
    prof_pit.to_csv(f"{DATA}/profit_cache_pit.csv", sep='\t', index=False)
    print(f"\n增广缓存: fcf {fcf0['code'].nunique()}→{fcf_pit['code'].nunique()}只, "
          f"profit {prof0['code'].nunique()}→{prof_pit['code'].nunique()}只", flush=True)

    # 逐年: 本应进池的退市股
    from v5_xgb_turnover_comparative import build_dynamic_universe, format_qlib_code
    delist_codes = set(fin["code"].unique())
    dmap = delist.set_index("code")["delist_date"].to_dict()
    nmap = delist.set_index("code")["name"].to_dict()
    rebal_first = {2019: "2019-01-02", 2021: "2021-01-04", 2022: "2022-01-04", 2023: "2023-01-03",
                   2024: "2024-01-02", 2025: "2025-01-02", 2026: "2026-01-05"}
    print("\n" + "=" * 60 + "\n  各年【本应进池但被存活偏差剔除】的退市股\n" + "=" * 60, flush=True)
    total_hits = set()
    for y in [2019, 2021, 2022, 2023, 2024, 2025, 2026]:
        pool_old = set(format_qlib_code(c) for c in build_dynamic_universe(y, fcf0, prof0))
        pool_new = set(format_qlib_code(c) for c in build_dynamic_universe(y, fcf_pit, prof_pit))
        added = pool_new - pool_old
        # 仅保留 该年首调仓时仍在市 的退市股 (delist_date >= 首调仓)
        rb = pd.Timestamp(rebal_first[y])
        added_delist = [(a, nmap.get(a[2:], a), dmap.get(a[2:])) for a in added
                        if a[2:] in delist_codes and dmap.get(a[2:], pd.Timestamp("1900")) >= rb]
        total_hits |= set(a for a, _, _ in added_delist)
        print(f"\n[{y}] 存量池{len(pool_old)} → PIT池含退市候选{len(added_delist)}只:", flush=True)
        for a, nm, dd in sorted(added_delist, key=lambda x: str(x[2])):
            print(f"    {a} {nm}  (退市日 {dd.date() if pd.notna(dd) else '?'})", flush=True)
    print(f"\n>>> 全期共 {len(total_hits)} 只退市股本应进入历史池, 之前被系统性剔除", flush=True)
    print("[+] 已存 fcf_cache_pit.csv / profit_cache_pit.csv", flush=True)


if __name__ == "__main__":
    main()
