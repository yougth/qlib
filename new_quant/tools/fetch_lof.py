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

用法:
    python3 tools/fetch_lof.py                     # 首次: 抓成交额前80只 (已有跳过)
    python3 tools/fetch_lof.py --update SZ161725 SZ160632 ...
        # 增量更新持仓 LOF 的场内日线 (重抓覆盖, 新数据晚于本地末行才落盘)
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
        h = ak.fund_etf_hist_sina(symbol=sym.lower())   # 新浪只认小写
        h.to_csv(px_p, index=False)
        time.sleep(0.3)
    if not (os.path.exists(nav_p) and os.path.getsize(nav_p) > 200):
        n = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        n.to_csv(nav_p, index=False)
        time.sleep(0.3)


def update_px(sym):
    """增量更新单只 LOF 的场内日线 (重抓全历史, 仅当比本地更新才覆盖落盘)"""
    import pandas as pd
    code = sym[-6:]
    px_p = os.path.join(LOF_DIR, f"{code}_px.csv")
    old_last = None
    if os.path.exists(px_p):
        try:
            old = pd.read_csv(px_p)
            if len(old):
                old_last = str(old["date"].iloc[-1])
        except Exception:
            old_last = None
    h = ak.fund_etf_hist_sina(symbol=sym.lower())   # 新浪只认小写
    if h is None or not len(h):
        return f"{sym}: 抓取为空, 保留旧文件"
    h = h.drop_duplicates("date").sort_values("date")
    new_last = str(h["date"].iloc[-1])
    if old_last and new_last <= old_last:
        return f"{sym}: 无新数据 (止于 {old_last})"
    h.to_csv(px_p, index=False)
    time.sleep(0.3)
    return f"{sym}: {old_last or '新建'} → {new_last}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", nargs="*", metavar="SYM", default=None,
                    help="增量更新指定 LOF (sz161725 带前缀); 空列表=更新 data/lof 全部已有 px")
    a = ap.parse_args()
    os.makedirs(LOF_DIR, exist_ok=True)
    if a.update is not None:
        syms = list(a.update)
        if not syms:  # 无参数: data/lof 下全部已有 px 文件
            import glob
            syms = ["sz" + os.path.basename(p).split("_")[0]
                    for p in glob.glob(os.path.join(LOF_DIR, "*_px.csv"))]
        print(f"[LOF] 增量更新 {len(syms)} 只...", flush=True)
        for s in sorted(syms):
            try:
                print(f"  {update_px(s)}", flush=True)
            except Exception as e:
                print(f"  [WARN] {s}: {e!r}", flush=True)
        return
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