"""
core.universe —— 滚动窗口 + 十年双正 PIT 股票池 (无穿越核心之一)
================================================================================
- build_windows: 滚动5年 (4年train + 1年valid, embargo 12月留白) → 回测1年
- build_dynamic_universe: Y-11..Y-2 十年窗口 FCF全正 + 净利全正 (year-2 规则防前视)
"""
import pandas as pd

from . import config


def format_qlib_code(code):
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"


def build_windows():
    """滚动5年 (4年train+1年valid) → 回测1年; 信号段含上年12月末修复1月空仓"""
    wins = []
    for y in range(2020, 2027):
        bt_end = "2026-07-23" if y == 2026 else f"{y}-12-31"
        sig_end = "2026-06-30" if y == 2026 else f"{y}-11-30"
        wins.append({
            "year": y, "name": f"W{y}",
            "train": (f"{y-5}-01-01", f"{y-2}-12-31"),
            "valid": (f"{y-1}-01-01", f"{y-1}-11-30"),      # embargo: 12月留白
            "test":  (f"{y-1}-12-01", sig_end),              # 信号段(含上年12月末)
            "bt_end": bt_end,
        })
    return wins


def load_pit_caches():
    fcf = pd.read_csv(config.FCF_CACHE, sep="\t", dtype={"code": str})
    prof = pd.read_csv(config.PROFIT_CACHE, sep="\t", dtype={"code": str})
    # ---- check: 缓存完整性 ----
    if len(fcf) == 0 or len(prof) == 0:
        raise RuntimeError("[CHECK] PIT缓存为空, 拒绝继续!")
    if fcf.duplicated(["code", "year"]).sum() or prof.duplicated(["code", "year"]).sum():
        raise RuntimeError("[CHECK] PIT缓存存在重复(code,year)键!")
    for df, col in [(fcf, "fcf"), (prof, "net_profit")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df["year"] = df["year"].astype(int)
    print(f"[CHECK] fcf_pit: {fcf.shape}, {fcf['code'].nunique()}只, "
          f"{fcf['year'].min()}~{fcf['year'].max()} | profit_pit: {prof.shape}, "
          f"{prof['code'].nunique()}只, {prof['year'].min()}~{prof['year'].max()}", flush=True)
    return fcf, prof


def build_dynamic_universe(backtest_year, fcf_df, profit_df):
    """Y-11..Y-2 十年窗口, FCF全正 + 净利全正(净利数据2016年起才全覆盖);
    数据覆盖≥80%年份即可入池 (既有放宽条款)"""
    fcf_years = list(range(backtest_year - 11, backtest_year - 1))
    f = fcf_df[fcf_df["year"].isin(fcf_years)].dropna(subset=["fcf"])
    fcf_pos = f.groupby("code").filter(
        lambda g: len(g) >= len(fcf_years) * 0.8 and (g["fcf"] > 0).all())
    fcf_codes = set(fcf_pos["code"])
    p_start = max(backtest_year - 11, 2016)
    p_years = list(range(p_start, backtest_year - 1))
    if p_years:
        p = profit_df[profit_df["year"].isin(p_years)].dropna(subset=["net_profit"])
        prof_pos = p.groupby("code").filter(
            lambda g: len(g) >= len(p_years) * 0.8 and (g["net_profit"] > 0).all())
        codes = fcf_codes & set(prof_pos["code"])
    else:
        codes = fcf_codes
    if len(codes) < config.MIN_POOL:
        raise RuntimeError(f"[CHECK] {backtest_year}年股票池仅{len(codes)}只(<{config.MIN_POOL}), 数据异常!")
    return sorted(codes)
