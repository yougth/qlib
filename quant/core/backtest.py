"""
core.backtest —— 组合回测引擎 (T+1, 全期连续NAV) + 指标 + 标准图1输出
================================================================================
- portfolio_backtest: 月频调仓, 单边换手×0.4% 成本, 返回 (日收益, 月均换手, 买入笔数)
- calc_metrics: 图1九指标之收益系 (年化/波动/Sharpe/回撤/Calmar/超额)
- annualized_since: 剔2020对齐口径的子区间年化
- print_standard_report: 分年收益表 + 图1整体指标表 (用户固定输出格式)
"""
import numpy as np
import pandas as pd

from . import config


def portfolio_backtest(rebalances, price_mat):
    """rebalances: [(exec_dt, [等权持仓]), ...] 时间升序; 成本=单边换手×0.4%.
    返回 (日收益, 月均单边换手, 总买入笔数)"""
    rebalances = sorted(rebalances, key=lambda x: x[0])
    dates = [d for d in price_mat.index if d >= rebalances[0][0]]
    shares, value = {}, 1.0
    navs, ds, turnovers = [], [], []
    n_buys = 0
    ri = 0
    for d in dates:
        px = price_mat.loc[d]
        if shares:
            value = sum(sh * px[i] for i, sh in shares.items() if pd.notna(px.get(i)))
        while ri < len(rebalances) and rebalances[ri][0] == d:
            targets = [t for t in rebalances[ri][1]
                       if pd.notna(px.get(t)) and px.get(t, 0) > 0]
            if targets:
                n_buys += len(set(targets) - set(shares.keys()))   # 新建仓笔数
                w_tgt = {t: 1.0 / len(targets) for t in targets}
                w_cur = {i: sh * px[i] / value for i, sh in shares.items()
                         if pd.notna(px.get(i))} if shares and value > 0 else {}
                to = 0.5 * sum(abs(w_tgt.get(i, 0) - w_cur.get(i, 0))
                               for i in set(w_tgt) | set(w_cur))
                value -= value * to * config.FEE_ROUNDTRIP
                shares = {t: w_tgt[t] * value / px[t] for t in targets}
                turnovers.append(to)
            ri += 1
        navs.append(value)
        ds.append(d)
    nav = pd.Series(navs, index=pd.DatetimeIndex(ds))
    return (nav.pct_change().fillna(0),
            (np.mean(turnovers) if turnovers else 0.0), n_buys)


def calc_metrics(returns, bench=None):
    if len(returns) == 0:
        return {}
    n_years = len(returns) / 244
    ar = (1 + returns).prod() ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(244)
    nav = (1 + returns).cumprod()
    mdd = ((nav / nav.cummax()) - 1).min()
    out = {"ar": ar, "vol": vol, "sharpe": ar / vol if vol > 0 else 0,
           "mdd": mdd, "calmar": ar / abs(mdd) if mdd < 0 else float("nan"),
           "total": nav.iloc[-1] - 1}
    if bench is not None:
        b = bench.reindex(returns.index).fillna(0)
        ex = (1 + returns).prod() ** (1 / n_years) - (1 + b).prod() ** (1 / n_years)
        out["excess_ar"] = ex
    return out


def annualized_since(returns, start):
    """从 start 起(含)子区间的年化收益, 用于对齐 value_comp 基线(跳过2020)口径"""
    sub = returns[returns.index >= pd.Timestamp(start)]
    if len(sub) < 60:
        return float("nan")
    return (1 + sub).prod() ** (244 / len(sub)) - 1


def attrib_returns(rets, bench, rets_untimed=None):
    """收益归因 (DFA/AQR 尽调第一问: 你赚的是什么钱?):
    总年化收益 ≈ beta贡献 + 择时贡献 + 选股/因子贡献

    - beta贡献: 日频 OLS 斜率 × 基准年化 (市场暴露部分)
    - 择时贡献: 择时策略 vs 同策略未择时的年化差 (DL_T/ICW_T/ICW_SW 专用,
      其余策略为 0); 复利交互项也归入择时, 保证三项之和≈总收益
    - 选股/因子贡献: 残差 (含质量/价值/动量因子暴露, 无法再拆因无因子收益序列)
    注: 算术拆分 CAGR 忽略复利交互, 是归因报表的标准近似。
    返回 dict 或 {} (样本不足)。
    """
    b = bench.reindex(rets.index).fillna(0)
    ok = rets.notna() & b.notna()
    x, y = b[ok], rets[ok]
    if len(x) < 60 or x.std() == 0:
        return {}
    beta = np.cov(y, x)[0, 1] / x.var()
    n_years = len(rets) / 244
    r_total = (1 + rets).prod() ** (1 / n_years) - 1
    r_bench = (1 + b).prod() ** (1 / n_years) - 1
    beta_contrib = beta * r_bench
    timing_contrib = 0.0
    if rets_untimed is not None:
        r_untimed = (1 + rets_untimed).prod() ** (1 / n_years) - 1
        timing_contrib = r_total - r_untimed
    return {"total": r_total, "beta": beta_contrib, "timing": timing_contrib,
            "alpha": r_total - beta_contrib - timing_contrib,
            "beta_coef": beta, "bench_ar": r_bench}


# ==================== 标准输出 (用户固定格式: 分年 + 图1九指标) ====================
def _fpp(x):
    return "  n/a  " if pd.isna(x) else f"{x*100:7.2f}%"


def _fn(x, d=3):
    return " n/a " if pd.isna(x) else f"{x:.{d}f}"


def print_yearly_table(yearly_all, order, labels, title="一、分年收益"):
    years = sorted({yr for m in yearly_all.values() for yr in m})
    print(f"\n{'='*96}\n  {title} (月频, T+1, 往返成本0.4%)\n{'='*96}", flush=True)
    print("策略".ljust(20) + "".join(f"{yr:>9}" for yr in years), flush=True)
    for tag in order:
        lab = labels.get(tag, tag)
        line = lab.ljust(18) + "".join(
            f"{yearly_all[tag].get(yr, float('nan'))*100:>8.1f}%"
            if yr in yearly_all.get(tag, {}) else f"{'-':>9}" for yr in years)
        print(line, flush=True)


def print_metric_table(mdf, title="二、整体指标 (图1口径)"):
    print(f"\n{'='*96}\n  {title}\n{'='*96}", flush=True)
    disp = pd.DataFrame({
        "策略": mdf["策略"],
        "年化收益": mdf["年化收益"].map(_fpp),
        "年化(剔20)": mdf["年化(剔2020)"].map(_fpp),
        "年化波动": mdf["年化波动"].map(_fpp),
        "Sharpe": mdf["Sharpe"].map(lambda x: _fn(x, 2)),
        "最大回撤": mdf["最大回撤"].map(_fpp),
        "Calmar": mdf["Calmar"].map(lambda x: _fn(x, 2)),
        "IC": mdf["IC"].map(lambda x: _fn(x, 4)),
        "RankIC": mdf["RankIC"].map(lambda x: _fn(x, 4)),
        "换手率": mdf["换手率"].map(_fpp),
        "交易次数": mdf["交易次数"],
    })
    print(disp.to_string(index=False), flush=True)
