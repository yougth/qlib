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
import numpy as np
import pandas as pd

from .tradability import build_candidates, get_month_end_dates, get_biweekly_dates

# 全池等权对照系: 不取 TopK, 直接持有全部候选
POOL_STRATS = ("POOL_EW",)


def signal_days(cal, start, end, freq="monthly"):
    """信号日入口: monthly=月末, biweekly=每10交易日 (双周频)"""
    if freq == "monthly":
        return get_month_end_dates(cal, start, end)
    elif freq == "biweekly":
        return get_biweekly_dates(cal, start, end)
    else:
        raise ValueError(f"Unsupported frequency: {freq}")


def exec_date(cal_idx, sig_dt):
    """T+1 执行日; 信号日已是日历末尾 (无下一交易日) 时返回 None → 调用方丢弃该信号"""
    pos = int(cal_idx.searchsorted(sig_dt)) + 1
    if pos >= len(cal_idx):
        return None
    return cal_idx[pos]


def eligible_candidates(base_idx, sig_dt, limit_up, susp, liq, topk,
                        strict=False, tag="", min_cand=None, px=None):
    """sig_dt 当日可交易候选集; 不足 topk 时 strict 抛错 / 否则返回 None"""
    cand = build_candidates(base_idx, sig_dt, limit_up, susp, liq, px=px)
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


def cascade_score(base_score, ml_pred, dt, base_topk):
    """级联打分: 基本面底仓 → 模型精排 (方向: value粗筛保底 + ML弹性增强)

    - 先按 base_score (value+盈利/VGH) 降序取 top base_topk 作为安全底仓
    - 底仓内按 ml_pred (ENS截面 rank) 精排 → 返回 ML rank 百分位
    - 底仓外的候选返回 NaN → 被 top_picks 剔除, 即最终持仓 ⊆ 底仓
    - ML 预测缺失的底仓成员给最低 rank (na_option=bottom), 仍可被选中
    """
    top_base = base_score.dropna().nlargest(base_topk).index
    ml = ml_pred.xs(dt, level=0).reindex(top_base)
    return ml.rank(pct=True, na_option="bottom").reindex(base_score.index)


def veto_score(base_score, ml_pred, dt, veto_frac=0.3):
    """排雷打分 (形态C): 模型只有否决权, 没有提名权 (V18 哲学延伸)。

    - 候选中模型分截面 rank 最低 veto_frac 比例 → base_score 置 NaN (top_picks 剔除)
    - 模型缺预测 → 最低 rank (na_option=bottom, 与 cascade_score 同口径, 保守排雷)
    - 其余候选保留原 value 分, 提名权完全在 value_comp 手里。
    """
    ml = ml_pred.xs(dt, level=0).reindex(base_score.index)
    r = ml.rank(pct=True, na_option="bottom")
    n_veto = int(np.ceil(len(r) * veto_frac))
    if n_veto <= 0:
        return base_score
    return base_score.mask(base_score.index.isin(r.nsmallest(n_veto).index))


def blend_score(base_score, ml_pred, dt, w):
    """融合打分 (形态A): (1-w)×value_rank + w×model_rank (V5 α=0.3 的受控重测)。

    - base_score 与 ml_pred 各自截面 rank 百分位后线性混合, 输入需同在 [0,1]
    - 模型缺预测 → 中性 0.5 (value 强的票不因模型无覆盖而被杀)
    - value 缺失 → 仍 NaN (无估值数据本就不该进 value_comp 候选)。
    """
    v = base_score.rank(pct=True)
    m = ml_pred.xs(dt, level=0).reindex(base_score.index).rank(pct=True).fillna(0.5)
    return (1 - w) * v + w * m


def emit_window_signals(name, pred, universe, cal, val_piv, xs, xe,
                        limit_up, susp, liq, fwd_mat, topk,
                        ic_records=None, rebalances=None, win_name="",
                        strict=False, min_cand=None, holdings_log=None,
                        score_fn=None, ic_model=None, topk_pick=None,
                        turnover_buffer=None, prev_holdings=None,
                        freq="monthly", hysteresis_band=None, px=None):
    """单模型 单窗口: 遍历月末信号日, 把 (exec_dt, 持仓) 追加到 rebalances[name]。

    pred=None 且 score_fn=None 时走 value_comp 打分 (纯估值对照系)。
    topk    : 候选集门禁下限 (候选少于它就视为该月不可交易)
    topk_pick: 实际买入只数, 默认等于 topk (VAL20 取 20 而门禁仍按 topk 判定)
    turnover_buffer: 换手缓冲比例 (0~1). 调仓时强制保留至少
        round(n_pick * turnover_buffer) 只上一期仍在候选中的持仓 (按得分从高到低保留),
        仅对新进入候选的股票换仓 → 显著降低换手/交易成本。
    hysteresis_band: 滞回阈值 (0~1). 排名变化不显著不调仓:
        上一期持仓中排名仍在 top n_pick*(1+band) 以内的, 即使跌出 topk_pick 也保留,
        只替换跌出 band 外的持仓。与 turnover_buffer 互斥 (滞回优先, 设置了滞回则缓冲不生效)。
        例: topk_pick=10, band=0.2 → 排名≤12的旧持仓保留, 只换掉跌出12名之后的。
    prev_holdings: 上一期持仓列表 (由调用方维护跨窗口状态, 传 None 则本窗口内累计)。
    返回成功产出信号的月份数。
    """
    from .valuation import value_comp_score          # 局部导入避免循环依赖

    cal_idx = pd.DatetimeIndex(cal)
    n_pick = topk if topk_pick is None else topk_pick
    n_ok = 0
    local_prev = prev_holdings
    for sig_dt in signal_days(cal, xs, xe, freq=freq):
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
                                   strict=strict, tag=name, min_cand=min_cand, px=px)
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

        # ---- 降换手机制 (滞回带宽优先; 二者不叠加, 缓冲的"无条件保留"会救回滞回已淘汰的持仓) ----
        # 滞回: 上一期持仓中当前排名仍在 top n_pick*(1+band) 以内的, 即使跌出 topk_pick 也保留
        if hysteresis_band and local_prev and 0 < hysteresis_band < 1:
            band_size = max(int(n_pick * (1 + hysteresis_band)), n_pick)
            ranked_all = score.dropna().nlargest(band_size).index.tolist()
            prev_set = set(local_prev)
            # 旧持仓仍在带宽内的, 按当前得分从高到低保留 (上限 n_pick)
            prev_keep = [s for s in ranked_all if s in prev_set][:n_pick]
            keep_set = set(prev_keep)
            # 新 top 中替补 (按排名从高到低)
            new_picks = [t for t in top if t not in keep_set]
            top = prev_keep + new_picks[: n_pick - len(prev_keep)]

        # ---- 换手缓冲: 保留上一期仍在候选中的高分持仓, 降低换手 ----
        elif turnover_buffer and local_prev and 0 < turnover_buffer < 1:
            n_keep = max(int(round(n_pick * turnover_buffer)), 0)
            # 上一期持仓中, 仍在当前候选里的 (按当前得分从高到低排)
            prev_in_cand = [s for s in local_prev if s in score.index]
            # 按分数降序保留
            ranked_prev = sorted(prev_in_cand, key=lambda s: (-(score[s] if pd.notna(score[s]) else -1e9)))[:n_keep]
            # 新 top 中, 去掉被保留的, 再补足到 n_pick
            keep_set = set(ranked_prev)
            rest = [t for t in top if t not in keep_set]
            top = ranked_prev + rest[: n_pick - len(keep_set)]

        rebalances[name].append((exec_dt, top))
        if holdings_log is not None:
            holdings_log.append({"signal_date": sig_dt.date(),
                                 "exec_date": exec_dt.date(),
                                 "model": name, "holdings": ",".join(top)})
        local_prev = top
        n_ok += 1
    return n_ok


def emit_vghx_signals(name, xgb_pred, universe, cal, val_piv, xs, xe,
                      limit_up, susp, liq, fwd_mat, topk,
                      vgh_score_fn, base_topk, ic_records=None, rebalances=None,
                      win_name="", strict=False, min_cand=None, holdings_log=None,
                      ic_model=None, blend_w=0.5, px=None):
    """VGHX: VGH 与 XGB 横向 Rank 融合 (不截断, 保留各自 Alpha).

    - VGH 和 XGB 分别对同一候选池打分 → 各自 Rank → Z-Score 标准化
    - 0.5 * Z_Rank_VGH + 0.5 * Z_Rank_XGB 加权 → 取 Top topk
    不再做底仓物理截断 (避免召回/精排特征空间错位):
    VGH 重绝对估值与财务基石, XGB 重非线性定价与动量截面, 相关性极低,
    横向融合在牺牲极少进攻性的前提下填平 XGB 极端风格切换下的回撤.
    base_topk 参数保留但不再用于截断 (横向融合对全候选打分).
    """
    cal_idx = pd.DatetimeIndex(cal)
    n_ok = 0
    for sig_dt in signal_days(cal, xs, xe):
        # ---- 1. XGB 预测截面作为候选基础 ----
        if sig_dt not in xgb_pred.index.get_level_values(0):
            if strict:
                raise RuntimeError(f"[CHECK] {sig_dt.date()} 无{name}预测截面!")
            continue
        cross = xgb_pred.xs(sig_dt, level=0)
        base_idx = cross.index

        cand = eligible_candidates(base_idx, sig_dt, limit_up, susp, liq, topk,
                                   strict=strict, tag=name, min_cand=min_cand, px=px)
        if cand is None:
            continue

        # ---- 2. VGH 与 XGB 各自打分 (同一候选池) ----
        vgh = vgh_score_fn(cand, sig_dt)
        xgb_cand = cross.reindex(cand).dropna()
        common = vgh.index.intersection(xgb_cand.index)
        if len(common) < topk:
            if strict:
                raise RuntimeError(f"[CHECK] {sig_dt.date()} {name} 融合可用{len(common)}只(<{topk})!")
            continue
        vgh = vgh[common]
        xgb_c = xgb_cand[common]

        # ---- 3. Rank → Z-Score 标准化 ----
        vgh_rank = vgh.rank(pct=True)
        xgb_rank = xgb_c.rank(pct=True)

        def zscore(s):
            m, sd = s.mean(), s.std()
            return (s - m) / sd if sd > 0 else (s - m)

        z_vgh = zscore(vgh_rank)
        z_xgb = zscore(xgb_rank)
        final_score = (1 - blend_w) * z_vgh + blend_w * z_xgb

        record_cross_ic(ic_records, win_name, ic_model or name, sig_dt,
                        final_score, fwd_mat)
        exec_dt = exec_date(cal_idx, sig_dt)
        if exec_dt is None:
            continue
        top = top_picks(final_score, topk, name)
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
