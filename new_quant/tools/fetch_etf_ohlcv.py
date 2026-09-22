#!/usr/bin/env python3
"""tools.fetch_etf_ohlcv —— ETF 日线抓取 (腾讯 fqkline) → new_quant/data/etf/

沿用 qlib/quant/tools/fetch_ohlcv.py 的口径与限速纪律 (单线程, REQ_GAP=0.5s):
  · 优先 hfq(后复权, 含分红再投资): close=hfq, factor=hfq/raw, 真实价=close/factor
  · 接口对某 ETF 不支持 hfq 时回退 raw(factor=1, mode=raw) 并显式记录
    —— 有分红的 ETF 回退口径会低估收益, 回测报告中披露
  · volume 单位手 (本策略不用 volume, 仅留存)

用法:
    python3 tools/fetch_etf_ohlcv.py            # 已存在则跳过
    python3 tools/fetch_etf_ohlcv.py --refresh  # 强制重抓
    python3 tools/fetch_etf_ohlcv.py --update   # 增量追加新交易日 (看板/盯市日用)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config  # noqa: E402

HOSTS = ["https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
         "https://ifzq.gtimg.cn/appstock/app/fqkline/get"]
START = "2014-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")   # 动态: 刷新/增量都以今天为上界
PAGE = 640
MAX_PAGES = 14
REQ_GAP = 0.5

_last = [0.0]


def _throttle():
    while True:
        wait = _last[0] + REQ_GAP - time.time()
        if wait <= 0:
            _last[0] = time.time()
            return
        time.sleep(min(wait, 2.0))


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X "
                                    "10_15_7) AppleWebKit/537.36 Chrome/120"})
    return s


def _page(sess, sym, end, fq):
    for attempt in range(len(HOSTS) * 2 + 1):
        host = HOSTS[attempt % len(HOSTS)]
        _throttle()
        try:
            r = sess.get(f"{host}?param={sym},day,,{end},{PAGE},{fq}",
                         timeout=(5, 20))
            if r.status_code != 200 or not r.text.lstrip().startswith("{"):
                time.sleep(1.0 + attempt)
                continue
            j = r.json()
            d = (j.get("data") or {}).get(sym) or {}
            return d.get(fq + "day" if fq else "day") or []
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return []


def _series(sess, sym, fq):
    out, end, seen = [], END, set()
    for _ in range(MAX_PAGES):
        rows = _page(sess, sym, end, fq)
        if not rows:
            break
        rows = [r for r in rows if r[0] not in seen]
        if not rows:
            break
        for r in rows:
            seen.add(r[0])
        out.extend(rows)
        first = min(r[0] for r in rows)
        if first <= START or len(rows) < PAGE:
            break
        end = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if end < START:
            break
    if not out:
        return None
    df = pd.DataFrame([r[:6] for r in out],
                      columns=["date", "open", "close", "high", "low", "volume"])
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")


def fetch_one(sess, sym, refresh=False):
    path = os.path.join(config.ETF_DIR, f"{sym}.parquet")
    if os.path.exists(path) and not refresh:
        return "skip"
    hfq = _series(sess, sym, "hfq")
    raw = _series(sess, sym, "")
    if hfq is None or not len(hfq):
        if raw is None or not len(raw):
            return "EMPTY"
        m = raw.assign(factor=1.0, mode="raw")
    elif raw is None or not len(raw):
        m = hfq.assign(factor=1.0, mode="hfq_only")   # close 已是后复权
    else:
        m = hfq.merge(raw[["date", "close"]].rename(columns={"close": "raw_close"}),
                      on="date", how="left")
        m["factor"] = m["close"] / m["raw_close"]
        m.loc[~np.isfinite(m["factor"]), "factor"] = np.nan
        m["factor"] = m["factor"].ffill().bfill().fillna(1.0)
        m["mode"] = "hfq"
        m = m.drop(columns=["raw_close"])
    bad = (m[["open", "close", "high", "low"]] <= 0).any(axis=1)
    if bad.any():
        if bad.mean() > 0.05:
            return f"ERR NonPositivePrice {int(bad.sum())}/{len(m)}"
        m = m[~bad]
    if not len(m):
        return "EMPTY"
    m["symbol"] = sym
    m.to_parquet(path, index=False)
    return f"ok({len(m)}行,{m['mode'].iloc[0]})"


def _series_since(sess, sym, fq, since):
    """抓 since (含, 重叠一日用于衔接) 之后的日线; 返回 DataFrame 或 None"""
    out, end, seen = [], END, set()
    since_ts = pd.Timestamp(since)
    for _ in range(MAX_PAGES):
        rows = _page(sess, sym, end, fq)
        if not rows:
            break
        rows = [r for r in rows if r[0] not in seen]
        if not rows:
            break
        for r in rows:
            seen.add(r[0])
        out.extend(rows)
        first = pd.Timestamp(min(r[0] for r in rows))
        if first <= since_ts or len(rows) < PAGE:
            break
        end = (first - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if end < START:
            break
    if not out:
        return None
    df = pd.DataFrame([r[:6] for r in out],
                      columns=["date", "open", "close", "high", "low", "volume"])
    df = df[df["date"] >= since]          # 重叠一日, merge 后 drop_duplicates 去重
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")


def update_one(sess, sym):
    """增量更新已有 parquet: 只追加本地末行之后的新交易日 (行情源 hfq 重述不影响,
    factor 按行重算 = hfq/raw, 真实价 close/factor 始终 = 不复权价, 序列连续)"""
    path = os.path.join(config.ETF_DIR, f"{sym}.parquet")
    if not os.path.exists(path):
        return fetch_one(sess, sym, refresh=True)     # 无文件 → 全量
    old = pd.read_parquet(path)
    if not len(old):
        return fetch_one(sess, sym, refresh=True)
    last = str(old["date"].iloc[-1])
    if last >= END:
        return "skip(最新)"
    hfq = _series_since(sess, sym, "hfq", last)
    raw = _series_since(sess, sym, "", last)
    if hfq is None or not len(hfq):
        if raw is None or not len(raw):
            return f"ok(无新行, 仍止于 {last})"
        add = raw.assign(factor=1.0, mode="raw")
    elif raw is not None and len(raw):
        add = hfq.merge(raw[["date", "close"]].rename(columns={"close": "raw_close"}),
                        on="date", how="left")
        add["factor"] = add["close"] / add["raw_close"]
        add.loc[~np.isfinite(add["factor"]), "factor"] = np.nan
        add["factor"] = add["factor"].ffill().bfill().fillna(1.0)
        add["mode"] = "hfq"
        add = add.drop(columns=["raw_close"])
    else:
        add = hfq.assign(factor=1.0, mode="hfq_only")
    add = add.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")
    add["symbol"] = sym
    m = pd.concat([old, add[list(old.columns)]], ignore_index=True)
    m = m.drop_duplicates("date", keep="last").sort_values("date")
    m.to_parquet(path, index=False)
    return f"ok(+{len(add)}行, 至 {m['date'].iloc[-1]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--update", action="store_true", help="增量追加新交易日 (已存在文件)")
    a = ap.parse_args()
    os.makedirs(config.ETF_DIR, exist_ok=True)
    sess = _session()
    print(f"[fetch] ETF {len(config.ETF_SYMS)} 只, {START}~{END}, 输出 {config.ETF_DIR}"
          f"{' (增量)' if a.update else ''}", flush=True)
    t0 = time.time()
    for sym, name in config.ETF_SYMS.items():
        r = update_one(sess, sym) if a.update else fetch_one(sess, sym, a.refresh)
        print(f"  {sym} {name}: {r} ({time.time()-t0:.0f}s)", flush=True)
        if r.startswith("ERR"):
            print(f"  [WARN] {sym} 异常: {r}", flush=True)


if __name__ == "__main__":
    main()
