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
import os
import pandas as pd

from . import config


def format_qlib_code(code):
    """A股: 6位 → SH/SZ; 港股: 5位 → hk (港股代码5位, A股6位)"""
    c = str(code)
    if len(c) == 5 and c.isdigit():
        return f"hk{c}"
    c = c.zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"


def build_windows():
    """滚动5年 (4年train + 10个月valid) → 回测1年; 信号段含上年12月末修复1月空仓.
    train/valid 末尾各留一个月 purge gap, 覆盖 20 交易日 label horizon (见模块头).

    W2027 为 Paper Trading 专用窗口 (数据截止 2026-07-23):
      train: 2022-01-01 ~ 2025-10-31 (purge 至 valid.start)
      valid: 2025-12-01 ~ 2026-06-30 (purge gap; 信号段需预留 20 交易日 label)
      test:  2026-06-01 ~ 2026-07-23 (最新数据, 含 6 月末月末信号)
    """
    wins = []
    for y in range(2020, 2028):
        if y == 2026:
            bt_end = "2026-07-23"
            sig_end = "2026-06-30"
            wins.append({
                "year": y, "name": f"W{y}",
                "train": (f"{y-5}-01-01", f"{y-2}-11-30"),
                "valid": (f"{y-1}-01-01", f"{y-1}-10-31"),
                "test":  (f"{y-1}-12-01", sig_end),
                "bt_end": bt_end,
            })
        elif y == 2027:
            wins.append({
                "year": y, "name": f"W{y}",
                "train": ("2022-01-01", "2025-09-30"),
                "valid": ("2025-11-01", "2026-05-31"),
                "test":  ("2026-06-01", "2026-06-23"),
                "bt_end": "2026-07-23",
            })
        else:
            wins.append({
                "year": y, "name": f"W{y}",
                "train": (f"{y-5}-01-01", f"{y-2}-11-30"),
                "valid": (f"{y-1}-01-01", f"{y-1}-10-31"),
                "test":  (f"{y-1}-12-01", f"{y}-11-30"),
                "bt_end": f"{y}-12-31",
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
    """加载 A股 + 港股 PIT 基本面缓存, 合并返回"""
    fcf = pd.read_csv(config.FCF_CACHE, sep="\t", dtype={"code": str})
    prof = pd.read_csv(config.PROFIT_CACHE, sep="\t", dtype={"code": str})
    # ---- 合并港股基本面 (如果存在) ----
    hk_fcf_path = os.path.join(os.path.dirname(config.FCF_CACHE), "fcf_cache_hk.csv")
    hk_prof_path = os.path.join(os.path.dirname(config.PROFIT_CACHE), "profit_cache_hk.csv")
    if os.path.exists(hk_fcf_path):
        hk_fcf = pd.read_csv(hk_fcf_path, sep="\t", dtype={"code": str})
        hk_fcf = hk_fcf.drop_duplicates(["code", "year"], keep="last")  # 港股财报有多版本, 保留最新
        fcf = pd.concat([fcf, hk_fcf], ignore_index=True)
        print(f"[CHECK] 合并港股FCF: +{len(hk_fcf)}行, {hk_fcf['code'].nunique()}只", flush=True)
    if os.path.exists(hk_prof_path):
        hk_prof = pd.read_csv(hk_prof_path, sep="\t", dtype={"code": str})
        hk_prof = hk_prof.drop_duplicates(["code", "year"], keep="last")
        prof = pd.concat([prof, hk_prof], ignore_index=True)
        print(f"[CHECK] 合并港股利润: +{len(hk_prof)}行, {hk_prof['code'].nunique()}只", flush=True)
    # ---- check: 缓存完整性 ----
    if len(fcf) == 0 or len(prof) == 0:
        raise RuntimeError("[CHECK] PIT缓存为空, 拒绝继续!")
    if fcf.duplicated(["code", "year"]).sum() or prof.duplicated(["code", "year"]).sum():
        raise RuntimeError("[CHECK] PIT缓存存在重复(code,year)键!")
    for df, col in [(fcf, "fcf"), (prof, "net_profit")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df["year"] = df["year"].astype(int)
    print(f"[CHECK] fcf_pit合并后: {fcf.shape}, {fcf['code'].nunique()}只, "
          f"{fcf['year'].min()}~{fcf['year'].max()} | profit_pit: {prof.shape}, "
          f"{prof['code'].nunique()}只, {prof['year'].min()}~{prof['year'].max()}", flush=True)
    return fcf, prof


def build_dynamic_universe(backtest_year, fcf_df, profit_df):
    """Y-11..Y-2 十年窗口, FCF全正 + 净利全正(净利数据2016年起才全覆盖);
    数据覆盖≥80%年份即可入池 (既有放宽条款)

    B股(6位A代码中2/9开头)排除: 不可交易且财报口径不同。
    港股(5位代码)保留: 港币计价但可通港股通交易, 财报口径为港股标准。"""
    def is_b_share(code):
        c = str(code)
        return len(c) == 6 and c[0] in ("2", "9")  # 只排除A股B股, 港股5位不受影响

    def is_hk_share(code):
        return len(str(code)) == 5 and str(code).isdigit()  # 港股5位数字代码

    fcf_years = list(range(backtest_year - 11, backtest_year - 1))
    f = fcf_df[fcf_df["year"].isin(fcf_years)].dropna(subset=["fcf"])
    f = f[~f["code"].apply(is_b_share)]              # 排除 B 股 (保留港股5位)
    if not config.INCLUDE_HK:
        f = f[~f["code"].apply(is_hk_share)]         # 纯A股对照: 排除港股
    fcf_pos = f.groupby("code").filter(
        lambda g: len(g) >= len(fcf_years) * 0.8 and (g["fcf"] > 0).all())
    fcf_codes = set(fcf_pos["code"])

    # ---- 港股豁免通道: 连续5年 FCF + 净利 双正 ----
    # 解决"存活期截断": 阿里等成熟科技巨头仅上市10年/刚过资本开支期,
    # 不满足十年双正, 但已连续5年双正 (用户指定门槛, 替换原FCF复合增速版).
    # 对港股独有资产放宽财务初筛: 最近5年(Y-6..Y-2) FCF 与净利 全部为正.
    # PIT: 只用 Y-11..Y-2 的已披露数据, 不穿越.
    if config.INCLUDE_HK and config.HK_EXEMPT_CHANNEL:
        hk_f = f[f["code"].apply(is_hk_share)]
        hk5_years = list(range(backtest_year - 6, backtest_year - 1))
        for code, g in hk_f.groupby("code"):
            if code in fcf_codes:
                continue  # 已通过十年双正
            g = g.sort_values("year")
            recent5 = g[g["year"].isin(hk5_years)]
            if len(recent5) < 4:
                continue
            if (recent5["fcf"] > 0).all():
                fcf_codes.add(code)

    p_start = max(backtest_year - 11, 2016)
    p_years = list(range(p_start, backtest_year - 1))
    if p_years:
        p = profit_df[profit_df["year"].isin(p_years)].dropna(subset=["net_profit"])
        p = p[~p["code"].apply(is_b_share)]          # 排除 B 股 (保留港股5位)
        if not config.INCLUDE_HK:
            p = p[~p["code"].apply(is_hk_share)]     # 纯A股对照: 排除港股
        prof_pos = p.groupby("code").filter(
            lambda g: len(g) >= len(p_years) * 0.8 and (g["net_profit"] > 0).all())
        # 豁免通道港股: 最近5年净利全部为正
        if config.INCLUDE_HK and config.HK_EXEMPT_CHANNEL:
            hk_p = p[p["code"].apply(is_hk_share)]
            for code in list(fcf_codes):
                if code in prof_pos["code"].unique():
                    continue
                gp = hk_p[hk_p["code"] == code].sort_values("year")
                recent5p = gp[gp["year"].isin(list(range(backtest_year - 6, backtest_year - 1)))]
                if len(recent5p) >= 4 and (recent5p["net_profit"] > 0).all():
                    continue  # 连续5年净利正, 保留豁免
                else:
                    fcf_codes.discard(code)
        codes = fcf_codes & set(prof_pos["code"])
    else:
        codes = fcf_codes
    if len(codes) < config.MIN_POOL:
        raise RuntimeError(f"[CHECK] {backtest_year}年股票池仅{len(codes)}只(<{config.MIN_POOL}), 数据异常!")
    return sorted(codes)
