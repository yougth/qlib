#!/usr/bin/env python3
"""
fetch_ohlcv —— 全市场日线抓取 (腾讯 fqkline), 含已退市股票
================================================================================
为什么重写数据源:
  原 ~/.qlib/qlib_data/cn_data(_fixed) 只有 358 只股票有 2020-09-25 之后的行情,
  而这 358 只恰好是"用今天的财务数据算出来的池子" → 整个 2020-2026 回测的可交易
  宇宙是按未来信息挑出来的 (数据获取层前视 + 存活偏差)。akshare 走的 eastmoney
  接口在本机被拒 (Empty reply), 腾讯 web.ifzq.gtimg.cn 可用且**含退市股**, 因此
  用它把 2012 年至今的全市场行情整体重抓一遍, 不再做拼接。

口径:
  · 主序列 = hfq(后复权): 收益/特征口径正确, 且**恒为正**。
    绝不能用 qfq: 腾讯/东财的前复权是"逐次减现金红利", 长历史下会把价格压成负数
    —— 实测 sh600519 的 qfq 在 2012 年为 -160, 原 cn_data 里 sh600256 有 14 个
    负收盘价就是这么来的 (负价格→收益率 -142%/+600%, 直接污染标签与特征)。
  · 同时抓不复权序列, 存 factor = hfq_close / raw_close
    → 真实成交价 = $close / $factor  (流动性口径与实盘下单价都必须用它)
  · volume 单位 = 手 (成交额 = 真实价 * volume * 100)
  · 指数无复权概念, 用不复权序列, factor = 1

用法:
    python3 tools/fetch_ohlcv.py                 # 全量(自动续跑, 已有文件跳过)
    python3 tools/fetch_ohlcv.py --refresh       # 忽略已有文件重抓
    python3 tools/fetch_ohlcv.py --limit 50      # 试点
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

BASE = "https://ifzq.gtimg.cn/appstock/app/fqkline/get"
# 备用主机: 腾讯 WAF 按主机名封 —— 8 线程跑 30 只就把 web.ifzq.gtimg.cn 封了
# (全部返回 501 + waf.tencent.com 跳转页, 二十分钟不恢复), 而同源的 ifzq.gtimg.cn
# 当时仍然 200。所以多主机轮换, 并在识别到 WAF 时整体退避而不是硬重试。
HOSTS = ["https://ifzq.gtimg.cn/appstock/app/fqkline/get",
         "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"]
START = "2012-01-01"
END = "2026-07-23"
PAGE = 640                      # 接口单次上限
MAX_PAGES = 14                  # 640*14 ≈ 8960 交易日, 远超 2012 至今
REQ_GAP = 0.35                 # 每请求最小间隔(秒) — 全局共享, 多线程也只快不了一点
                                # 0.12×4线程=8.3req/s 在~1800只后触发 ifzq WAF;
                                # 0.35=2.86req/s (3倍安全余量), 0.5=2req/s 太慢
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data_cache", "tencent")
CODE_SRC = os.environ.get("CODE_SRC", "/Users/11164591/Documents/Qoder目录/fcf_cache_pit.csv")
INDEX_SYMS = ["sh000300", "sh000905", "sh000001"]

_local = None
_lock = __import__("threading").Lock()
_last_req = [0.0]               # 全局上次请求时刻 (限速用)
_waf_until = [0.0]              # 全局 WAF 退避截止时刻
_waf_hits = [0]


def _throttle():
    """全局限速 + WAF 退避: 所有线程共用一个节流阀。

    并发本身不是问题, 单位时间请求数才是 —— 8 线程无间隔跑 30 只就被封了。
    """
    while True:
        with _lock:
            now = time.time()
            wait = max(_waf_until[0] - now, _last_req[0] + REQ_GAP - now)
            if wait <= 0:
                _last_req[0] = now
                return
        time.sleep(min(wait, 5.0))


def _waf_backoff(status):
    """识别到 WAF 拦截: 全局退避 (每次翻倍, 上限 120s), 让所有线程一起停"""
    with _lock:
        _waf_hits[0] += 1
        delay = min(120.0, 5.0 * (2 ** min(_waf_hits[0] - 1, 5)))
        _waf_until[0] = max(_waf_until[0], time.time() + delay)
        if _waf_hits[0] in (1, 5, 20, 100) or _waf_hits[0] % 500 == 0:
            print(f"  [WAF] 第{_waf_hits[0]}次拦截 (HTTP {status}), "
                  f"全局退避 {delay:.0f}s", flush=True)


def _session():
    global _local
    import threading
    if _local is None:
        _local = threading.local()
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X "
                                        "10_15_7) AppleWebKit/537.36 Chrome/120"})
        _local.s = s
    return s


def _page(sym, end, fq):
    """取 sym 在 end 及之前的最多 PAGE 根日线. fq: 'hfq' 或 '' (不复权)

    三类失败必须区分对待, 否则要么把数据悄悄抓漏, 要么把自己彻底封死:
      · WAF 拦截 (501 + 非 JSON 正文): 换主机 + 全局退避, 期间所有线程一起等。
        如果按普通错误硬重试, 只会让封禁更久。
      · 普通网络错误: 短退避后重试。
      · 该窗口确实没数据: 返回空页, 由 _series 收尾。
    重试耗尽返回 [] 而不抛异常 —— 抛异常会被上层记成 err 并放大成长时间挂起。
    """
    for attempt in range(len(HOSTS) * 2 + 1):
        host = HOSTS[attempt % len(HOSTS)]
        _throttle()
        try:
            r = _session().get(f"{host}?param={sym},day,,{end},{PAGE},{fq}",
                               timeout=(5, 20))
            if r.status_code != 200 or not r.text.lstrip().startswith("{"):
                _waf_backoff(r.status_code)
                continue
            j = r.json()
            d = (j.get("data") or {}).get(sym) or {}
            return d.get(fq + "day" if fq else "day") or []
        except Exception:
            time.sleep(0.3 * (attempt + 1) + random.random() * 0.3)
    return []


def _series(sym, fq):
    """向后翻页取全历史 [START, END]"""
    out, end, seen = [], END, set()
    for _ in range(MAX_PAGES):
        rows = _page(sym, end, fq)
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
    df = df[(df.date >= START) & (df.date <= END)]
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")


def fetch_one(code):
    """code: 6位数字 或 sh/sz 前缀符号. 返回 (code, 行数 or 0 or 'ERR..')"""
    sym = code if code[:2] in ("sh", "sz") else \
        ("sh" + code if code[0] in "69" else "sz" + code)
    path = os.path.join(OUT_DIR, f"{sym}.parquet")
    if os.path.exists(path) and not FORCE:
        return code, -2                                  # 跳过
    is_index = sym in INDEX_SYMS
    try:
        adj = _series(sym, "" if is_index else "hfq")
        if adj is None or len(adj) == 0:
            return code, 0
        if is_index:
            m = adj.assign(factor=1.0)
        else:
            raw = _series(sym, "")
            if raw is not None and len(raw):
                m = adj.merge(raw[["date", "close"]].rename(columns={"close": "raw_close"}),
                              on="date", how="left")
            else:
                m = adj.assign(raw_close=adj["close"])
            m["factor"] = m["close"] / m["raw_close"]
            m.loc[~np.isfinite(m["factor"]), "factor"] = np.nan
            m["factor"] = m["factor"].ffill().bfill().fillna(1.0)
            m = m.drop(columns=["raw_close"])
        # ---- 非正价: hfq 下不该出现 (qfq 才会, 见模块头)。少量停牌行可容忍剔除,
        #      大面积异常说明这只股的数据不可信, 报错让人看见, 不静默写盘 ----
        ohlc = ["open", "close", "high", "low"]
        bad = (m[ohlc] <= 0).any(axis=1)
        if bad.any():
            if bad.mean() > 0.05:
                return code, f"ERR NonPositivePrice {int(bad.sum())}/{len(m)}行"
            m = m[~bad]
        if len(m) == 0:
            return code, 0
        m["symbol"] = sym
        m.to_parquet(path, index=False)
        return code, len(m)
    except Exception as e:
        return code, f"ERR {type(e).__name__}: {str(e)[:60]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    global FORCE
    FORCE = a.refresh
    os.makedirs(OUT_DIR, exist_ok=True)

    codes = sorted(pd.read_csv(CODE_SRC, sep=None, engine="python",
                               dtype={"code": str})["code"].dropna().unique())
    codes = [c.zfill(6) for c in codes]
    codes = INDEX_SYMS + codes
    if a.limit:
        codes = codes[:a.limit]
    print(f"[fetch] 标的 {len(codes)} 个 (含 {len(INDEX_SYMS)} 指数), "
          f"区间 {START}~{END}, 输出 {OUT_DIR}", flush=True)

    t0 = time.time()
    done = skip = empty = err = 0
    errs = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (code, n) in enumerate(ex.map(fetch_one, codes), 1):
            if n == -2:
                skip += 1
            elif isinstance(n, str):
                err += 1
                errs.append((code, n))
            elif n == 0:
                empty += 1
            else:
                done += 1
            if i % 200 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(codes)} ok={done} skip={skip} empty={empty} "
                      f"err={err} {el:.0f}s (eta {el/i*(len(codes)-i):.0f}s)",
                      flush=True)
    print(f"[fetch] 完成: ok={done} skip={skip} empty={empty} err={err} "
          f"用时 {time.time()-t0:.0f}s", flush=True)
    for c, e in errs[:20]:
        print("   ERR", c, e, flush=True)
    if err > len(codes) * 0.02:
        sys.exit(f"[CHECK] 失败率 {err/len(codes):.1%} > 2%, 数据不完整, 拒绝继续!")


FORCE = False
if __name__ == "__main__":
    main()
