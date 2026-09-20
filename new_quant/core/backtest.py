"""new_quant.core.backtest —— T+1 组合回测引擎 + 指标 + 标准报告
================================================================================
- run_portfolio: 目标权重调仓 (权重和可<1, 余为现金), 单边换手×费率计成本;
  退市股按冻结价持有 (last_date 事件计数披露, 不假装能卖出)
- calc_metrics: 年化/波动/Sharpe(CAGR/σ, 与项目旧口径一致)/回撤/Calmar
- RankIC/IC: 信号日截面打分 vs 后21个交易日真实前瞻收益 (OOS 构造)
- 标准输出: 分年收益表 + 图1九指标表 (用户固定格式)
"""
import numpy as np
import pandas as pd

from . import config


def run_portfolio(adj_close, rebalances, fee_rt, last_date=None):
    """rebalances: [(exec_dt, {sym: w})]; 权重和<=1, 余为现金(零收益)。
   空目标字典 = 清仓换现金 (不再解释为"跳过调仓")。

   返回 (日收益Series, 月均单边换手, 总买入笔数, 退市冻结事件数)
    """
    rebalances = sorted(rebalances, key=lambda x: x[0])
    px = adj_close.ffill()
    d0 = rebalances[0][0]
    dates = [d for d in px.index if d >= d0]
    shares, cash = {}, 1.0                    # 现金账户显式跟踪 (权重和<1的部分)
    navs, ds, tos, n_buys = [], [], [], 0
    frozen = set()
    ri = 0
    for d in dates:
        row = px.loc[d]
        pos = sum(sh * row[i] for i, sh in shares.items()
                  if pd.notna(row.get(i, np.nan)))
        value = pos + cash
        while ri < len(rebalances) and rebalances[ri][0] == d:
            tgt = {i: w for i, w in rebalances[ri][1].items()
                   if pd.notna(row.get(i, np.nan)) and row.get(i, 0) > 0}
            n_buys += len(set(tgt) - set(shares))
            w_cur = {i: sh * row[i] / value for i, sh in shares.items()
                     if pd.notna(row.get(i, np.nan))} if shares and value > 0 else {}
            to = 0.5 * sum(abs(tgt.get(i, 0) - w_cur.get(i, 0))
                           for i in set(tgt) | set(w_cur))
            value -= value * to * fee_rt
            shares = {i: w * value / row[i] for i, w in tgt.items()}
            cash = value - sum(sh * row[i] for i, sh in shares.items())
            tos.append(to)
            if last_date is not None:
                for i in shares:
                    if d > last_date.get(i, d):
                        frozen.add((i, str(d.date())))
            ri += 1
        navs.append(value)
        ds.append(d)
    nav = pd.Series(navs, index=pd.DatetimeIndex(ds), name="nav")
    return nav.pct_change().fillna(0), (np.mean(tos) if tos else 0.0), n_buys, frozen


def calc_metrics(returns, bench=None):
    n = len(returns)
    if n == 0:
        return {}
    yrs = n / 244
    ar = (1 + returns).prod() ** (1 / yrs) - 1
    vol = returns.std() * np.sqrt(244)
    nav = (1 + returns).cumprod()
    mdd = (nav / nav.cummax() - 1).min()
    out = {"年化收益": ar, "年化波动": vol,
           "Sharpe": ar / vol if vol > 0 else 0.0,
           "最大回撤": mdd, "Calmar": ar / abs(mdd) if mdd < 0 else float("nan"),
           "换手率": float("nan"), "交易次数": int("nan") if False else 0}
    if bench is not None:
        b = bench.reindex(returns.index).fillna(0)
        out["超额年化"] = ar - ((1 + b).prod() ** (1 / yrs) - 1)
    return out


def yearly_returns(returns):
    return (1 + returns).groupby(returns.index.year).prod() - 1


def rank_ic(adj_close, sig_scores, horizon=21):
    """sig_scores: {sig_dt: Series(score)}; 前瞻收益 = P[t+h]/P[t]-1 (后复权)"""
    px = adj_close
    ics, rics = [], []
    for t, s in sig_scores.items():
        if not len(s):
            continue
        pos = px.index.get_indexer([t])
        if pos[0] < 0 or pos[0] + horizon >= len(px):
            continue
        fwd = px.iloc[pos[0] + horizon] / px.iloc[pos[0]] - 1
        fwd = fwd.reindex(s.index)
        ok = s.notna() & fwd.notna()
        if ok.sum() < 10:
            continue
        ics.append(float(s[ok].corr(fwd[ok])))
        rics.append(float(s[ok].corr(fwd[ok], method="spearman")))
    return (np.mean(ics) if ics else float("nan"),
            np.mean(rics) if rics else float("nan"), len(ics))


# ==================== 标准输出 (用户固定格式) ====================
def _fpp(x):
    return "  n/a  " if pd.isna(x) else f"{x*100:7.2f}%"


def _fn(x, d=2):
    return " n/a " if pd.isna(x) else f"{x:.{d}f}"


def print_yearly_table(yearly_all, order, labels, title="一、分年收益"):
    years = sorted({y for m in yearly_all.values() for y in m})
    print(f"\n{'='*100}\n  {title} (月频, T+1)\n{'='*100}", flush=True)
    print("策略".ljust(22) + "".join(f"{y:>9}" for y in years), flush=True)
    for tag in order:
        line = labels.get(tag, tag).ljust(20)
        for y in years:
            v = yearly_all.get(tag, {}).get(y)
            line += f"{v*100:>8.1f}%" if v is not None and pd.notna(v) else f"{'-':>9}"
        print(line, flush=True)


def print_metric_table(rows, title="二、整体指标 (图1口径)"):
    print(f"\n{'='*100}\n  {title}\n{'='*100}", flush=True)
    cols = ["策略", "年化收益", "年化波动", "Sharpe", "最大回撤", "Calmar",
            "IC", "RankIC", "换手率", "交易次数"]
    print("策略".ljust(22) + "".join(f"{c:>10}" for c in cols[1:]), flush=True)
    for r in rows:
        line = r["策略"].ljust(20)
        line += f"{_fpp(r['年化收益']):>11}{_fpp(r['年化波动']):>10}"
        line += f"{_fn(r['Sharpe']):>10}{_fpp(r['最大回撤']):>10}"
        line += f"{_fn(r['Calmar']):>10}{_fn(r.get('IC'), 4):>10}"
        line += f"{_fn(r.get('RankIC'), 4):>10}{_fpp(r.get('换手率')):>10}"
        line += f"{str(r.get('交易次数', 'n/a')):>10}"
        print(line, flush=True)
