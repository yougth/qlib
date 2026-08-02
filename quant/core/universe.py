"""
core.universe —— 滚动窗口 + 十年双正 PIT 股票池 (无穿越核心之一)
================================================================================
- build_windows: 滚动5年 (4年train + 10个月valid) → 回测1年; 两侧 purge label horizon
- build_dynamic_universe: Y-11..Y-2 十年窗口 FCF全正 + 净利全正 (year-2 规则防前视)

【2026-07-31 穿越修复】旧分段 train 末 (Y-2)-12-31 / valid 末 (Y-1)-11-30 紧贴下一段
起点, 而 label = Ref($close,-20) 需要样本日之后 20 个交易日的收盘价, 导致:
  · valid 末 20 个交易日(约9%样本)的 label 落在 test 段内 → early stopping 所用的
    valid RankIC 含 test 期价格信息, 模型选择间接偷看测试集 (真穿越)
  · train 末 20 个交易日的 label 落在 valid 段内 → 早停失去独立性 (方法论瑕疵)
修复: 两段末尾各回撤一个月 (purged split, 见 López de Prado). 旧的"12月留白"说法是
错的 —— 12月恰是 test 段第一个月, 留白从未存在。assert_purged 用真实交易日历校验,
不再依赖日期字符串的字面形状。
"""
import pandas as pd

from . import config


def format_qlib_code(code):
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"


def build_windows():
    """滚动5年 (4年train + 10个月valid) → 回测1年; 信号段含上年12月末修复1月空仓.
    train/valid 末尾各留一个月 purge gap, 覆盖 20 交易日 label horizon (见模块头)."""
    wins = []
    for y in range(2020, 2027):
        bt_end = "2026-07-23" if y == 2026 else f"{y}-12-31"
        sig_end = "2026-06-30" if y == 2026 else f"{y}-11-30"
        wins.append({
            "year": y, "name": f"W{y}",
            "train": (f"{y-5}-01-01", f"{y-2}-11-30"),      # purge: 末尾留白至 valid.start
            "valid": (f"{y-1}-01-01", f"{y-1}-10-31"),      # purge: 末尾留白至 test.start
            "test":  (f"{y-1}-12-01", sig_end),              # 信号段(含上年12月末)
            "bt_end": bt_end,
        })
    return wins


def assert_purged(wins, cal, horizon=None):
    """用真实交易日历校验 purge: 每段末尾样本的 label 结束日必须早于下一段起点。

    label = Ref($close,-H) 意味着 d 日样本的标签需要 d 之后第 H 个交易日的收盘价。
    若该日期 >= 下一段起点, 则下一段的价格信息已进入本段标签 → 穿越。
    这是对 embargo 的唯一有效检查方式; 只比对日期字符串(如 endswith("-11-30"))
    形同虚设 —— 历史上正是这个假检查放过了 valid→test 的 20 日侵入。
    """
    H = horizon if horizon is not None else config.LABEL_HORIZON
    cal = pd.DatetimeIndex(cal)
    bad = []
    for w in wins:
        for seg, nxt in [("train", "valid"), ("valid", "test")]:
            end = pd.Timestamp(w[seg][1])
            nxt_start = pd.Timestamp(w[nxt][0])
            i = int(cal.searchsorted(end, side="right")) - 1
            if i < 0:
                continue
            lbl_end = cal[min(i + H, len(cal) - 1)]
            if lbl_end >= nxt_start:
                n = int(cal.searchsorted(lbl_end, side="right")
                        - cal.searchsorted(nxt_start, side="left"))
                bad.append(f"{w['name']} {seg}.end={end.date()} 的 label 延伸至 "
                           f"{lbl_end.date()}, 侵入 {nxt}(起 {nxt_start.date()}) "
                           f"{n} 个交易日")
    if bad:
        raise RuntimeError("[CHECK] label horizon 穿越下一段, 拒绝继续!\n  " +
                           "\n  ".join(bad))
    return True


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
    数据覆盖≥80%年份即可入池 (既有放宽条款)

    B股(2/9开头)一律排除: 对境内普通投资者不可交易, 且 B 股财报口径
    与 A 股不同 (港币/美元计价), 混进池子会让等权组合和流动性口径失真。
    fcf_cache_pit.csv 里有 4 只 9 开头 + 11 只 2 开头的 B 股, 不过滤就会进去。"""
    fcf_years = list(range(backtest_year - 11, backtest_year - 1))
    f = fcf_df[fcf_df["year"].isin(fcf_years)].dropna(subset=["fcf"])
    f = f[~f["code"].str[0].isin(["2", "9"])]          # 排除 B 股
    fcf_pos = f.groupby("code").filter(
        lambda g: len(g) >= len(fcf_years) * 0.8 and (g["fcf"] > 0).all())
    fcf_codes = set(fcf_pos["code"])
    p_start = max(backtest_year - 11, 2016)
    p_years = list(range(p_start, backtest_year - 1))
    if p_years:
        p = profit_df[profit_df["year"].isin(p_years)].dropna(subset=["net_profit"])
        p = p[~p["code"].str[0].isin(["2", "9"])]     # 排除 B 股
        prof_pos = p.groupby("code").filter(
            lambda g: len(g) >= len(p_years) * 0.8 and (g["net_profit"] > 0).all())
        codes = fcf_codes & set(prof_pos["code"])
    else:
        codes = fcf_codes
    if len(codes) < config.MIN_POOL:
        raise RuntimeError(f"[CHECK] {backtest_year}年股票池仅{len(codes)}只(<{config.MIN_POOL}), 数据异常!")
    return sorted(codes)
