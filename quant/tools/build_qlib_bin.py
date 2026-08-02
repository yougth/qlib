"""
tools/build_qlib_bin.py —— 腾讯 parquet 行情 → qlib bin (带校验与对账)
================================================================================
用途: 把 tools/fetch_ohlcv.py 抓下来的全市场后复权行情转成 qlib 二进制库, 替换
原来那份"只有 358 只股票有 2020-09-25 之后行情"的 cn_data/cn_data_fixed。

为什么要自己写一层而不是直接调 dump_bin.py:
  1. **坏文件必须挡在日历之外**。qlib 的日历 = 所有输入文件日期的并集, 一个文件里
     混进周末/未来日期, 整个库的交易日历就废了, 而且后果极隐蔽 (多出来的"交易日"
     上全市场 NaN, 回测那天空仓)。所以先逐文件体检, 只把干净文件链进 staging。
  2. **口径必须在入库时锁死**: $close = 后复权价, $factor = 后复权/原始, 于是
     $close/$factor = 真实成交价。core/tradability.py 的流动性口径与 live 下单价
     都依赖这个恒等式 —— 老库里它只在 2019 年之后成立 (早年 factor 近乎常数,
     2013 年反推出的"真实价"只有实际的 36%), 这类错不校验就发现不了。
  3. **入库后必须对账**: 抽样逐行比对 bin 与 parquet, 并统计逐年有行情股票数。
     "逐年覆盖数"是前视选择偏差的唯一有效体检项 —— 老库 2013 年有 3000+ 只、
     2021 年只剩 358 只, 而那 358 只正是用今天的财务数据筛出来的名单。

用法:
    PYTHONPATH=. /usr/bin/python3 tools/build_qlib_bin.py            # 全流程
    PYTHONPATH=. /usr/bin/python3 tools/build_qlib_bin.py --stage-only
    PYTHONPATH=. /usr/bin/python3 tools/build_qlib_bin.py --verify-only
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

QUANT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QLIB_REPO = os.path.dirname(QUANT_DIR)
SRC_DIR = os.path.join(QLIB_REPO, "data_cache", "tencent")
STAGE_DIR = os.path.join(QLIB_REPO, "data_cache", "tencent_stage")
BIN_DIR = os.path.join(QLIB_REPO, "data_cache", "qlib_cn_tencent")
REPORT = os.path.join(QUANT_DIR, "outputs", "build_qlib_bin_report.txt")

FIELDS = "open,close,high,low,volume,factor"
NEED_COLS = ["date", "open", "close", "high", "low", "volume", "factor"]
MIN_ROWS = 60                      # 少于 60 个交易日的标的没有建模价值
DATE_LO, DATE_HI = pd.Timestamp("2012-01-01"), pd.Timestamp("2026-12-31")
SPOT_CHECK = ["sz000001", "sz000002", "sh000300", "sh600000", "sz300750"]

_log_lines = []


def log(s):
    print(s, flush=True)
    _log_lines.append(s)


# ================================================================================
# 1. 逐文件体检
# ================================================================================
def check_one(path):
    """返回 (ok, reason, info, clean_df). ok=False 的文件不进 staging。
    clean_df 非空 = 原文件被清洗过 (删行/截断), 需写新 parquet; None = 原样硬链接。"""
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return False, f"读取失败 {type(e).__name__}", {}, None
    miss = [c for c in NEED_COLS if c not in df.columns]
    if miss:
        return False, f"缺列 {miss}", {}, None
    if len(df) < MIN_ROWS:
        return False, f"仅{len(df)}行(<{MIN_ROWS})", {}, None

    d = pd.to_datetime(df["date"], errors="coerce")
    if d.isna().any():
        return False, "日期不可解析", {}, None
    if d.duplicated().any():
        return False, f"日期重复{int(d.duplicated().sum())}条", {}, None
    if (d.dt.dayofweek >= 5).any():
        return False, f"含周末日期{int((d.dt.dayofweek >= 5).sum())}条", {}, None
    if (d < DATE_LO).any() or (d > DATE_HI).any():
        return False, "日期越界", {}, None

    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    cleaned = False
    if not np.isfinite(df[["open", "close", "high", "low", "factor"]].to_numpy()).all():
        return False, "价格/因子含非有限值", {}, None
    if (df[["open", "close", "high", "low", "factor"]] <= 0).any().any():
        return False, "存在非正价格或非正因子", {}, None
    # 容差 1e-3 (0.1%): 腾讯 close 截断到 3 位小数, low/high 保留更多位, 精度差
    # 导致 low 比 close 大 0.00x。真正的高危数据错误 (high < open*0.99) 仍会被抓到。
    eps = 1e-3
    bad_ohlc = (h < np.maximum(o, c) * (1 - eps)) | (l > np.minimum(o, c) * (1 + eps))
    if bad_ohlc.any():
        if bad_ohlc.mean() > 0.05:
            return False, f"OHLC 不自洽{int(bad_ohlc.sum())}/{len(df)}行", {}, None
        df = df[~bad_ohlc].copy()      # 剔行不剔文件 (每只通常仅 1 行坏数据)
        cleaned = True
        o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    if (df["volume"] < 0).any():
        return False, "成交量为负", {}, None

    # ---- factor 大跳变截断: 退市/合并股的 hfq 与 raw 序列在特殊处理日
    # 会不匹配, factor 单日暴跌 >50%。跳变后的行 close 极小 (0.00x),
    # 混进日历会污染因子分布。截断到跳变前最后一个正常行 ----
    fac = df["factor"].to_numpy()
    fac_ret = fac[1:] / fac[:-1] - 1
    big_drop = np.where(fac_ret < -0.50)[0]
    if len(big_drop):
        cut = int(big_drop[0]) + 1
        df = df.iloc[:cut].copy()
        if len(df) < MIN_ROWS:
            return False, f"factor 跳变后仅剩{len(df)}行", {}, None
        cleaned = True

    # d 可能已过时 (清洗/截断后), 重取
    d = pd.to_datetime(df["date"])
    # ---- 以下只记录, 不否决 ----
    fac = df["factor"].to_numpy()
    fac_drop = int((fac[1:] < fac[:-1] * 0.99).sum())      # 后复权因子应单调不减
    ret = c.pct_change()
    jump = int((ret.abs() > 0.55).sum())                   # 单日 ±55% 以上
    return True, "", {"rows": len(df), "start": d.min(), "end": d.max(),
                      "fac_drop": fac_drop, "jump": jump,
                      "fac_min": float(fac.min()), "fac_max": float(fac.max())}, \
           (df if cleaned else None)


def stage():
    if not os.path.isdir(SRC_DIR):
        raise RuntimeError(f"[CHECK] 源目录不存在: {SRC_DIR}")
    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".parquet"))
    if not files:
        raise RuntimeError("[CHECK] 源目录没有 parquet!")
    if os.path.isdir(STAGE_DIR):
        shutil.rmtree(STAGE_DIR)
    os.makedirs(STAGE_DIR)

    ok, bad, infos = 0, [], []
    for f in files:
        good, reason, info, clean = check_one(os.path.join(SRC_DIR, f))
        if not good:
            bad.append((f, reason))
            continue
        if clean is not None:
            clean.to_parquet(os.path.join(STAGE_DIR, f), index=False)
        else:
            os.link(os.path.join(SRC_DIR, f), os.path.join(STAGE_DIR, f))
        ok += 1
        info["file"] = f
        infos.append(info)

    log(f"[stage] 源 {len(files)} 个文件 → 合格 {ok}, 剔除 {len(bad)}")
    for f, r in bad[:40]:
        log(f"    [剔除] {f}: {r}")
    if len(bad) > 40:
        log(f"    ... 另有 {len(bad) - 40} 个")
    if files and len(bad) / len(files) > 0.02:
        raise RuntimeError(f"[CHECK] 体检剔除率 {len(bad)/len(files):.1%} > 2%, 数据源不可信!")

    inf = pd.DataFrame(infos)
    log(f"[stage] 行数 中位数={int(inf['rows'].median())} 最小={int(inf['rows'].min())} "
        f"最大={int(inf['rows'].max())}")
    log(f"[stage] 因子倒退(>1%)标的 {int((inf['fac_drop'] > 0).sum())} 个; "
        f"单日|收益|>55% 合计 {int(inf['jump'].sum())} 条, 涉及 {int((inf['jump'] > 0).sum())} 个标的")
    worst = inf.sort_values("jump", ascending=False).head(8)
    for _, r in worst.iterrows():
        if r["jump"] > 0:
            log(f"    [跳变] {r['file']} {int(r['jump'])}条")
    return ok


# ================================================================================
# 2. dump
# ================================================================================
def dump(workers):
    sys.path.insert(0, os.path.join(QLIB_REPO, "scripts"))
    from dump_bin import DumpDataAll
    if os.path.isdir(BIN_DIR):
        shutil.rmtree(BIN_DIR)
    log(f"[dump] {STAGE_DIR} → {BIN_DIR} (fields={FIELDS})")
    DumpDataAll(data_path=STAGE_DIR, qlib_dir=BIN_DIR, freq="day",
                max_workers=workers, date_field_name="date",
                file_suffix=".parquet", symbol_field_name="symbol",
                include_fields=FIELDS).dump()


def check_calendar():
    p = os.path.join(BIN_DIR, "calendars", "day.txt")
    cal = pd.to_datetime(pd.read_csv(p, header=None)[0])
    if (cal.dt.dayofweek >= 5).any():
        raise RuntimeError("[CHECK] 日历含周末!")
    if cal.duplicated().any() or not cal.is_monotonic_increasing:
        raise RuntimeError("[CHECK] 日历重复或未排序!")
    per_year = cal.groupby(cal.dt.year).size()
    log(f"[cal] {cal.iloc[0].date()} ~ {cal.iloc[-1].date()} 共 {len(cal)} 天")
    log("[cal] 逐年交易日: " + " ".join(f"{y}:{n}" for y, n in per_year.items()))
    full = per_year[(per_year.index >= 2012) & (per_year.index <= 2025)]
    bad = full[(full < 230) | (full > 255)]
    if len(bad):
        raise RuntimeError(f"[CHECK] 以下年份交易日数异常(应230~255): {dict(bad)}")
    # 指数应逐日有行情 → 用它交叉验证日历没有多余日子
    idx = os.path.join(SRC_DIR, "sh000300.parquet")
    if os.path.exists(idx):
        d = pd.to_datetime(pd.read_parquet(idx)["date"])
        extra = set(cal[(cal >= d.min()) & (cal <= d.max())]) - set(d)
        log(f"[cal] 日历中沪深300无行情的日子: {len(extra)} 天"
            + (f" 例:{sorted(extra)[:5]}" if extra else ""))
        if len(extra) > 20:
            raise RuntimeError(f"[CHECK] 日历比沪深300多出 {len(extra)} 天, 疑有脏日期!")
    return cal


# ================================================================================
# 3. 入库后对账
# ================================================================================
def verify():
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    cal = check_calendar()
    insts = pd.read_csv(os.path.join(BIN_DIR, "instruments", "all.txt"), sep="\t",
                        header=None, names=["symbol", "start", "end"])
    log(f"[verify] instruments {len(insts)} 只")

    qlib.init(provider_uri=BIN_DIR, region=REG_CN)

    # ---- 3.1 抽样逐行对账 (bin vs parquet) ----
    for sym in SPOT_CHECK:
        f = os.path.join(SRC_DIR, f"{sym}.parquet")
        if not os.path.exists(f):
            log(f"    [抽查] {sym} 源文件不存在, 跳过")
            continue
        src = pd.read_parquet(f)
        src["date"] = pd.to_datetime(src["date"])
        src = src.set_index("date").sort_index()
        got = D.features([sym.upper()], ["$close", "$factor", "$volume"],
                         start_time=str(src.index[0].date()), end_time=str(src.index[-1].date()))
        got = got.droplevel(0).sort_index()
        both = src.join(got, how="inner")
        if len(both) < len(src) * 0.99:
            raise RuntimeError(f"[CHECK] {sym} bin 行数 {len(both)} 明显少于源 {len(src)}!")
        for a, b in [("close", "$close"), ("factor", "$factor"), ("volume", "$volume")]:
            rel = ((both[b] - both[a]).abs() / both[a].abs().clip(lower=1e-9)).max()
            if rel > 1e-4:
                raise RuntimeError(f"[CHECK] {sym} {a} 与 bin 不一致, 最大相对误差 {rel:.2e}!")
        real = (both["$close"] / both["$factor"])
        log(f"    [抽查] {sym} n={len(both)} 对账通过, 真实价 {real.iloc[0]:.2f}→{real.iloc[-1]:.2f}")

    # ---- 3.2 逐年有行情股票数 (前视选择偏差体检) ----
    # 用 instruments 的 [start,end] 区间统计, 不逐年拉特征 (5400只×15年太慢);
    # 老库那种"3585 只全部截断在 2020-09-28"的病在 end_datetime 上一样看得见。
    idx_syms = {"SH000300", "SH000905", "SH000001"}
    ins = insts.copy()
    ins["symbol"] = ins["symbol"].str.upper()
    ins["start"] = pd.to_datetime(ins["start"])
    ins["end"] = pd.to_datetime(ins["end"])
    ins = ins[~ins["symbol"].isin(idx_syms)]
    stocks = ins["symbol"].tolist()
    log(f"[verify] 逐年有行情股票数 (剔指数, 共{len(stocks)}只):")
    counts = {}
    for y in range(2012, 2027):
        e = pd.Timestamp("2026-07-23" if y == 2026 else f"{y}-12-31")
        n = int(((ins["start"] <= e) & (ins["end"] >= pd.Timestamp(f"{y}-01-01"))).sum())
        counts[y] = n
        log(f"    {y}: {n}")
    # end_datetime 聚堆 = 系统性截断 (老库 3585 只齐刷刷停在同一天)
    tail = ins[ins["end"] < pd.Timestamp("2026-06-01")]
    if len(tail):
        top = tail["end"].value_counts().head(3)
        log("[verify] 提前终止(退市/停更)标的 end_datetime 最集中的 3 天: "
            + ", ".join(f"{d.date()}×{n}" for d, n in top.items()))
        if top.iloc[0] > 50:
            raise RuntimeError(f"[CHECK] 有 {top.iloc[0]} 只标的行情齐刷刷终止于 "
                               f"{top.index[0].date()}, 这是数据截断而非退市!")
    thin = {y: n for y, n in counts.items() if n < 1500}
    if thin:
        raise RuntimeError(f"[CHECK] 以下年份有行情股票数 <1500, 仍是残缺行情: {thin}")
    # 覆盖数应随年份大体递增(A股扩容); 断崖式下跌 = 数据被截断
    ys = sorted(counts)
    for a, b in zip(ys, ys[1:]):
        if b <= 2025 and counts[b] < counts[a] * 0.85:
            raise RuntimeError(f"[CHECK] {a}→{b} 有行情股票数从 {counts[a]} 掉到 "
                               f"{counts[b]}, 疑似数据截断(老库正是这样漏掉3585只)!")

    # ---- 3.3 全库口径自检: $close/$factor 应等于真实价, 且价格恒正 ----
    smp = stocks[::max(1, len(stocks) // 300)][:300]
    px = D.features(smp, ["$close", "$factor", "$volume"],
                    start_time="2015-01-01", end_time="2026-07-23")
    bad_close = int((px["$close"] <= 0).sum())
    bad_fac = int((px["$factor"] <= 0).sum())
    if bad_close or bad_fac:
        raise RuntimeError(f"[CHECK] 抽样中非正 close {bad_close} 条 / 非正 factor {bad_fac} 条!")
    real = px["$close"] / px["$factor"]
    log(f"[verify] 抽样{len(smp)}只: 真实价分位 "
        f"1%={real.quantile(0.01):.2f} 50%={real.median():.2f} 99%={real.quantile(0.99):.2f}")
    if not (0.3 < real.quantile(0.01) and real.quantile(0.99) < 3000):
        raise RuntimeError("[CHECK] 真实价分布不合理, $factor 口径可疑!")
    amt = real * px["$volume"] * 100
    log(f"[verify] 抽样成交额中位数 {amt.median()/1e8:.3f} 亿元 (A股个股日成交额应在千万~十亿量级)")
    if not (1e6 < amt.median() < 1e10):
        raise RuntimeError(f"[CHECK] 成交额量级异常: 中位数 {amt.median():.3g} 元!")
    log(f"[verify] 日历 {len(cal)} 天, 全部检查通过 ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    try:
        if a.verify_only:
            verify()
        elif a.stage_only:
            stage()
        else:
            stage()
            dump(a.workers)
            verify()
    finally:
        with open(REPORT, "w") as fp:
            fp.write("\n".join(_log_lines) + "\n")
        print(f"[report] {REPORT}", flush=True)


if __name__ == "__main__":
    main()
