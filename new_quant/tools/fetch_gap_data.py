#!/usr/bin/env python3
"""tools.fetch_gap_data —— 缺口方向数据抓取 (方向1/2/8/9 数据层)
================================================================================
· 方向8 波动率: index_option_50etf_qvix / index_option_50etf_greeks(QVIX同源),
  上证50ETF期权隐含波动率指数 2015-02-09起 → data/qvix.csv
· 方向9 回购: stock_repurchase_em 当期快照 5515条 (含历史起始时间),
  → data/repurchase.csv (注意: 覆盖偏差=久远已完成回购可能缺席, 报告须披露)
· 方向1 期货: futures_main_sina 主力连续 (RB/HC/CU/AL/AU/AG 6品种)
  → data/futures_main/{sym}.csv
· 方向2 LOF: 先抓场内LOF列表, 个股行情+净值由 fetch_lof.py 二阶段抓
幂等: 已存在且行数>0 则跳过。akshare 从 /tmp 启动避免源码包遮蔽。
"""
import os
import sys
import time

sys.path = [p for p in sys.path if "Qoder" not in p]
import akshare as ak  # noqa: E402

DATA = "/Users/11164591/Documents/Qoder目录/qlib/new_quant/data"


def save(df, path, note):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[+] {note}: {path} ({len(df)}行)", flush=True)


def fetch_qvix():
    for tag, fn in [("QVIX上证50ETF", "index_option_50etf_qvix"),
                    ("QVIX300ETF", "index_option_300etf_qvix")]:
        p = os.path.join(DATA, f"{fn}.csv")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            print(f"[skip] {tag} 已存在", flush=True)
            continue
        try:
            df = getattr(ak, fn)()
            save(df, p, tag)
        except Exception as e:
            print(f"[WARN] {tag} 失败: {e!r}", flush=True)


def fetch_repurchase():
    p = os.path.join(DATA, "repurchase.csv")
    if os.path.exists(p) and os.path.getsize(p) > 10000:
        print("[skip] 回购已存在", flush=True)
        return
    df = ak.stock_repurchase_em()
    save(df, p, "回购公告快照")


def fetch_futures():
    syms = {"RB0": "螺纹钢主力", "HC0": "热卷主力", "CU0": "沪铜主力",
            "AL0": "沪铝主力", "AU0": "沪金主力", "AG0": "沪银主力"}
    for sym, name in syms.items():
        p = os.path.join(DATA, "futures_main", f"{sym}.csv")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            print(f"[skip] {name} 已存在", flush=True)
            continue
        try:
            df = ak.futures_main_sina(symbol=sym, start_date="20140101",
                                      end_date="20260918")
            save(df, p, f"期货主力连续 {name}")
            time.sleep(0.5)
        except Exception as e:
            print(f"[WARN] {name} 失败: {e!r}", flush=True)


if __name__ == "__main__":
    fetch_qvix()
    fetch_repurchase()
    fetch_futures()
    print("[DONE] 方向8/9/1 数据层抓取完成", flush=True)
