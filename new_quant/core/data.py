"""new_quant.core.data —— 共享数据只读访问层
================================================================================
只读复用 tencent parquet 行情与根目录财务/基准缓存, 不写任何共享文件。

口径 (与 qlib/quant/tools/fetch_ohlcv.py 一致, 已实地核对该工具文档):
  · parquet close = 后复权价(恒为正); 真实价 = close/factor
  · volume 单位 = 手 → 成交额 = 真实价 * volume * 100
  · 财务 PIT: 年报Y 于 Y+1-05-01 起可用 (usable_year)
存活偏差披露: 价格库含部分退市股(以 last_date 判别), 财务库覆盖 5400+ 代码
(含大量无行情的退市股) —— 两者覆盖差在 load_panels 里显式打印, 不静默。
"""
import glob
import os

import numpy as np
import pandas as pd

from . import config

_IDX = {"sh000001", "sh000300", "sh000905"}
_A_PREFIX = ("60", "68", "00", "30")   # A股: 沪主板/科创板/深主板/创业板


def load_calendar(start=None, end=None):
    """以上证指数 parquet 的日期序列为交易日历"""
    f = os.path.join(config.TENCENT_DIR, "sh000001.parquet")
    if not os.path.exists(f):
        raise RuntimeError(f"[CHECK] 日历源缺失: {f}")
    cal = pd.to_datetime(pd.read_parquet(f)["date"]).sort_values().reset_index(drop=True)
    lo = pd.Timestamp(start or config.BT_START) - pd.Timedelta(days=400)
    hi = pd.Timestamp(end or config.BT_END)
    cal = cal[(cal >= lo) & (cal <= hi)].reset_index(drop=True)
    print(f"[CHECK] 交易日历: {cal.iloc[0].date()} ~ {cal.iloc[-1].date()} ({len(cal)}天)",
          flush=True)
    return cal


def load_panels(start, end):
    """全市场只读面板: adj_close(后复权,ffill)/amount20/real_close/active/last_date

    ffill 语义: 停牌缺行向前补价; 退市股按最后价冻结 (回测引擎按 last_date
    统计"持有期内退市"事件并披露, 不假装它能以冻结价卖出)。
    """
    lo = pd.Timestamp(start) - pd.Timedelta(days=60)
    hi = pd.Timestamp(end)
    frames = []
    for f in sorted(glob.glob(os.path.join(config.TENCENT_DIR, "*.parquet"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in _IDX or not stem.startswith(("sh", "sz")):
            continue
        if not stem[2:].startswith(_A_PREFIX):
            continue                                    # 剔B股/指数/其他
        d = pd.read_parquet(f, columns=["date", "open", "high", "low",
                                        "close", "volume", "factor"])
        d["date"] = pd.to_datetime(d["date"])
        d = d[(d["date"] >= lo) & (d["date"] <= hi)]
        if not len(d):
            continue
        d["sym"] = stem.upper()
        frames.append(d)
    if not frames:
        raise RuntimeError("[CHECK] tencent parquet 面板为空, 拒绝回测!")
    px = pd.concat(frames, ignore_index=True).sort_values(["sym", "date"])

    fac = px["factor"].where(px["factor"] > 0)
    valid = px["close"].notna()
    fac_miss = float(fac[valid].isna().mean())
    if fac_miss > 0.01:
        raise RuntimeError(f"[CHECK] factor 缺失率 {fac_miss:.1%} > 1%, 口径不可信!")
    px["real"] = px["close"] / fac
    px["amount"] = px["real"] * px["volume"] * 100
    g = px.groupby("sym")
    px["amount20"] = g["amount"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    px["ret1"] = g["close"].transform(lambda x: x.pct_change())
    one_line = ((px["open"] == px["high"]) & (px["high"] == px["low"]) &
                (px["low"] == px["close"]) & (px["ret1"] > 0.09) &
                (px["volume"] > 0))
    limit_up = set(zip(px.loc[one_line, "date"], px.loc[one_line, "sym"]))
    med = px["amount20"].dropna().median()
    if not (1e6 < med < 1e11):
        raise RuntimeError(f"[CHECK] 成交额量级异常: 中位数={med:.3g}")

    piv = lambda col: px.pivot(index="date", columns="sym", values=col).sort_index()
    last_date = px.groupby("sym")["date"].max()
    n_all = len(last_date)
    n_dead = int((last_date < hi - pd.Timedelta(days=30)).sum())
    print(f"[CHECK] 面板: {n_all} 只A股 | 疑似退市/停更(last_date<末-30d): {n_dead} 只 | "
          f"中位20日均额 {med/1e8:.2f}亿 | 一字涨停 {len(limit_up)} 条", flush=True)
    return {
        "adj_close": piv("close").ffill(),
        "amount20": piv("amount20"),
        "real_close": piv("real"),
        "active": piv("volume").notna() & (piv("volume") > 0) & piv("close").notna(),
        "limit_up": limit_up,
        "last_date": last_date,
    }


def load_financials():
    """年度财务 PIT 表: sym/code/year/fcf/net_profit (外连接, 缺失即 NaN)

    profit 2012-2015 主缓存仅覆盖约250只(退市股专用管道产物), 用
    profit_patch_pit.csv 补齐存续股缺口 (东财业绩报表, 口径已交叉验证:
    与主缓存 2016 年 4584 只重叠段 100% 一致); 合并时主缓存优先。
    """
    fcf = pd.read_csv(config.FCF_CACHE, sep="\t", dtype={"code": str})
    prof = pd.read_csv(config.PROFIT_CACHE, sep="\t", dtype={"code": str})
    if os.path.exists(config.PROFIT_PATCH):
        patch = pd.read_csv(config.PROFIT_PATCH, sep="\t", dtype={"code": str})
        patch = patch[["code", "year", "net_profit"]]
        have = set(zip(prof["code"], prof["year"]))
        patch = patch[[(c, y) not in have
                       for c, y in zip(patch["code"], patch["year"])]]
        prof = pd.concat([prof[["code", "year", "net_profit"]], patch],
                         ignore_index=True)
        print(f"[CHECK] profit补丁: 增补 {len(patch)} 条 (主缓存优先) | "
              f"2013年覆盖 {prof[prof['year'] == 2013]['code'].nunique()} 只 "
              f"(原250只均为退市股管道数据, 予以保留)", flush=True)
    m = fcf[["code", "year", "fcf"]].merge(
        prof[["code", "year", "net_profit"]], on=["code", "year"], how="outer")
    m["code"] = m["code"].str.zfill(6)
    keep = m["code"].str[:2].isin(["60", "68"]) | m["code"].str[:2].isin(["00", "30"])
    dropped = int((~keep).sum())
    m = m[keep].copy()
    m["sym"] = np.where(m["code"].str[:2].isin(["60", "68"]),
                        "SH" + m["code"], "SZ" + m["code"])
    print(f"[CHECK] 财务表: {m['sym'].nunique()} 只A股 | 年份 "
          f"{int(m['year'].min())}~{int(m['year'].max())} | 剔除非A股前缀行 {dropped}",
          flush=True)
    return m[["sym", "code", "year", "fcf", "net_profit"]]


def usable_year(t):
    """PIT: t 时点可用的最新年报年份 (年报Y于Y+1-05-01起可用)"""
    return t.year - 1 if (t.month, t.day) >= (5, 1) else t.year - 2


def quality_syms(fin, t, n_years=None):
    """连续 n_years 年 FCF 与净利润双正且逐年都有数据 (缺年=不通过, 不静默放宽)"""
    n_years = n_years or config.QUALITY_YEARS
    y = usable_year(t)
    win = fin[fin["year"].between(y - n_years + 1, y)]
    if not len(win):
        return set()
    ok = win.groupby("sym").agg(
        n=("year", "nunique"),
        fcf_ok=("fcf", lambda s: bool(s.notna().all() and (s > 0).all())),
        np_ok=("net_profit", lambda s: bool(s.notna().all() and (s > 0).all())))
    return set(ok.index[(ok["n"] == n_years) & ok["fcf_ok"] & ok["np_ok"]])


def growth_frame(fin, t):
    """最新可用年报的盈利/FCF 增速与加速度 (年度口径, 缺前置年份则 NaN)"""
    y = usable_year(t)
    w = fin[fin["year"].between(y - 2, y)]
    if not len(w):
        return pd.DataFrame()
    p = w.pivot_table(index="sym", columns="year",
                      values=["net_profit", "fcf"], aggfunc="first")
    try:
        npy = p[("net_profit", y)]
        np1 = p[("net_profit", y - 1)]
        np2 = p[("net_profit", y - 2)]
        fy = p[("fcf", y)]
        f1 = p[("fcf", y - 1)]
    except KeyError:
        return pd.DataFrame()
    out = pd.DataFrame(index=p.index)
    out["g_np"] = npy / np1 - 1             # 盈利同比 (要求 np1>0 才可比)
    out.loc[(np1 <= 0) | np1.isna() | npy.isna(), "g_np"] = np.nan
    g_prev = np1 / np2 - 1
    out["accel"] = out["g_np"] - g_prev     # 加速度 (要求 np2>0)
    out.loc[(np2 <= 0) | np2.isna() | g_prev.isna(), "accel"] = np.nan
    out["g_fcf"] = fy / f1 - 1              # FCF 确认
    out.loc[(f1 <= 0) | f1.isna() | fy.isna(), "g_fcf"] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def cash_conversion(fin, t):
    """近3年 FCF/净利润 均值 (现金转换质量, 盈利含金量)"""
    y = usable_year(t)
    w = fin[fin["year"].between(y - 2, y)]
    if not len(w):
        return pd.Series(dtype=float)
    a = w.groupby("sym").agg(f=("fcf", "sum"), n=("net_profit", "sum"),
                             k=("year", "nunique"))
    r = (a["f"] / a["n"].abs()) .where((a["k"] == 3) & (a["n"] > 0))
    return r.clip(0, 3)


def load_benchmark():
    """沪深300日收益: 主源 tencent sh000300.parquet (与价格同源同终点),
    与根目录 csi300_cache.csv 重叠段交叉校验"""
    f = os.path.join(config.TENCENT_DIR, "sh000300.parquet")
    if not os.path.exists(f):
        raise RuntimeError("[CHECK] sh000300.parquet 缺失!")
    b = pd.read_parquet(f)[["date", "close"]].copy()
    b["date"] = pd.to_datetime(b["date"])
    s = b.set_index("date")["close"].astype(float).sort_index()
    if s.index[-1] < pd.Timestamp(config.BT_END) - pd.Timedelta(days=7):
        raise RuntimeError(f"[CHECK] 基准终点过旧: {s.index[-1]}")
    if s.pct_change().abs().max() > 0.12:
        raise RuntimeError("[CHECK] 基准存在异常日收益, 数据可疑!")
    if os.path.exists(config.BENCH_CACHE):
        c = pd.read_csv(config.BENCH_CACHE, sep="\t")
        c.columns = ["date", "close"]
        c["date"] = pd.to_datetime(c["date"])
        cs = c.set_index("date")["close"].astype(float).sort_index()
        ov = s.index.intersection(cs.index)
        if len(ov) > 200:
            d = (s.reindex(ov).pct_change() - cs.reindex(ov).pct_change()).abs().max()
            if d > 1e-4:
                raise RuntimeError(f"[CHECK] 两基准源不一致 (重叠段最大日差 {d:.2e})!")
    print(f"[CHECK] 沪深300基准: {s.index[0].date()} ~ {s.index[-1].date()} "
          f"({len(s)}天, 已交叉校验)", flush=True)
    return s.pct_change().fillna(0)


def month_end_signals(cal, start, end):
    """月末信号日 + 次一交易日执行日 (T+1)"""
    ds = [d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    out = []
    for i, d in enumerate(ds):
        if i + 1 < len(ds) and ds[i + 1].month != d.month:
            out.append((d, ds[i + 1]))
    return out
