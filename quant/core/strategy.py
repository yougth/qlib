"""
core.strategy —— 信号层: 打分 → 可交易候选 → T+1 持仓 (回测与实盘唯一实现)
================================================================================
在此之前, "月末信号日 → 过滤 → 取TopK → T+1 执行" 这段逻辑在 run_rolling.py /
run_benchmark.py / live/monthly_signal.py 各写了一遍。三份实现只要有一处漏了
T+1 或漏了候选过滤, 回测就会凭空多出收益, 而且很难被发现 —— 所以统一收到这里,
任何口径改动只有一个落点。

不穿越的三条硬约束 (由本模块保证, 不允许调用方绕过):
  1. 信号日 sig_dt 的打分只能用 <= sig_dt 的信息 (由上游 dataset/valuation 保证),
     本模块不再触碰任何行情。
  2. 可交易过滤 (一字涨停/停牌/流动性) 用的是 sig_dt 当日状态, 不看未来。
  3. 执行日 = sig_dt 的**下一个**交易日 (exec_date), 绝不允许 sig_dt 当日成交
     —— 月末收盘后才拿到信号, 当天买入就是穿越。

strict 语义:
  · 回测主线 (run_rolling): strict=True, 候选不足 / 预测截面缺失直接抛错。宁可
    跑不出结果, 也不能静默跳过某几个月 —— 跳过等于偷偷改了回测区间。
  · 模型初筛 (run_benchmark): strict=False, 允许跳过并回报 n_ok, 由调用方判断
    信号条数是否足够 (跨 16 个模型横向比较时个别模型缺截面是正常的)。
"""
import pandas as pd

from .tradability import build_candidates, get_month_end_dates

# 全池等权对照系: 不取 TopK, 直接持有全部候选
POOL_STRATS = ("POOL_EW",)


def signal_days(cal, start, end):
    """月末交易日 = 信号日 (语义唯一入口, 便于将来改成周频/双周频)"""
    return get_month_end_dates(cal, start, end)


def exec_date(cal_idx, sig_dt):
    """T+1 执行日; 信号日已是日历末尾 (无下一交易日) 时返回 None → 调用方丢弃该信号"""
    pos = int(cal_idx.searchsorted(sig_dt)) + 1
    if pos >= len(cal_idx):
        return None
    return cal_idx[pos]


def eligible_candidates(base_idx, sig_dt, limit_up, susp, liq, topk,
                        strict=False, tag="", min_cand=None):
    """sig_dt 当日可交易候选集; 不足 topk 时 strict 抛错 / 否则返回 None"""
    cand = build_candidates(base_idx, sig_dt, limit_up, susp, liq)
    if len(cand) < topk:
        if strict:
            raise RuntimeError(
                f"[CHECK] {sig_dt.date()} {tag} 可交易候选仅{len(cand)}只(<{topk})!")
        return None
    if min_cand and len(cand) < min_cand:
        print(f"    [WARN] {sig_dt.date()} 可交易候选{len(cand)}只偏少", flush=True)
    return cand


def record_cross_ic(ic_records, win_name, model, sig_dt, score, fwd_mat, min_n=10):
    """该信号日截面 IC/RankIC (score vs 未来 LABEL_HORIZON 日真实收益)。

    fwd_mat 是**信号段之后**的真实收益, 只用于事后评估模型排序能力, 绝不回流到
    训练或选股 —— 调用方不得把它并进 score。
    """
    if ic_records is None or fwd_mat is None or sig_dt not in fwd_mat.index:
        return
    fwd = fwd_mat.loc[sig_dt].reindex(score.index)
    df = pd.concat([score, fwd], axis=1).dropna()
    if len(df) < min_n:
        return
    ic_records.append({"window": win_name, "model": model, "sig_date": sig_dt,
                       "ic": df.iloc[:, 0].corr(df.iloc[:, 1]),
                       "rank_ic": df.iloc[:, 0].corr(df.iloc[:, 1],
                                                    method="spearman")})


def top_picks(score, topk, name=""):
    """按分数降序取 TopK; POOL_EW 等全池策略返回全部候选。NaN 分数一律剔除。"""
    if name in POOL_STRATS:
        return list(score.index)
    return score.dropna().nlargest(topk).index.tolist()


def emit_window_signals(name, pred, universe, cal, val_piv, xs, xe,
                        limit_up, susp, liq, fwd_mat, topk,
                        ic_records=None, rebalances=None, win_name="",
                        strict=False, min_cand=None, holdings_log=None,
                        score_fn=None, ic_model=None, topk_pick=None):
    """单模型 单窗口: 遍历月末信号日, 把 (exec_dt, 持仓) 追加到 rebalances[name]。

    pred=None 且 score_fn=None 时走 value_comp 打分 (纯估值对照系)。
    topk    : 候选集门禁下限 (候选少于它就视为该月不可交易)
    topk_pick: 实际买入只数, 默认等于 topk (VAL20 取 20 而门禁仍按 topk 判定)
    返回成功产出信号的月份数。
    """
    from .valuation import value_comp_score          # 局部导入避免循环依赖

    cal_idx = pd.DatetimeIndex(cal)
    n_pick = topk if topk_pick is None else topk_pick
    n_ok = 0
    for sig_dt in signal_days(cal, xs, xe):
        if pred is not None:
            if sig_dt not in pred.index.get_level_values(0):
                if strict:
                    raise RuntimeError(f"[CHECK] {sig_dt.date()} 无{name}预测截面!")
                continue
            cross = pred.xs(sig_dt, level=0)
            base_idx = cross.index
        else:
            cross, base_idx = None, pd.Index(universe)

        cand = eligible_candidates(base_idx, sig_dt, limit_up, susp, liq, topk,
                                   strict=strict, tag=name, min_cand=min_cand)
        if cand is None:
            continue

        if score_fn is not None:
            score = score_fn(cand, sig_dt)
        elif cross is not None:
            score = cross.reindex(cand)
        else:
            score = value_comp_score(cand, sig_dt, val_piv).reindex(cand)

        record_cross_ic(ic_records, win_name, ic_model or name, sig_dt, score, fwd_mat)

        exec_dt = exec_date(cal_idx, sig_dt)
        if exec_dt is None:
            continue
        top = top_picks(score, n_pick, name)
        if not top:
            if strict:
                raise RuntimeError(f"[CHECK] {sig_dt.date()} {name} 无可买标的!")
            continue
        rebalances[name].append((exec_dt, top))
        if holdings_log is not None:
            holdings_log.append({"signal_date": sig_dt.date(),
                                 "exec_date": exec_dt.date(),
                                 "model": name, "holdings": ",".join(top)})
        n_ok += 1
    return n_ok
