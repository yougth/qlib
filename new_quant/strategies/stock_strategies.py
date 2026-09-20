"""strategies.stock_strategies —— 方向4/5的可验证实现 (A股全市场, 质量门槛)

方向4 「无人关注的小公司复利」(代理版):
  数据缺口: 全市场无市值/股本/分析师覆盖数据 → "小/无人关注"用 20日均成交额
  分位代理 (已有2000万流动性红线兜底); 质量用 连续5年FCF+净利润双正 + 现金转换。
  这不是市值因子检验, 是"低关注度×质量"代理检验。
方向5 「盈利预期持续上修」(代理版):
  数据缺口: 无季度盈利预测/公告时点数据 → 用年度盈利同比+加速度+FCF确认代理。
  明确标注: 这是代理, 不是原文假设的季度预期修正。
基线 QUAL_EW: 质量池等权月度再平衡 (回答"选股规则是否打赢自己的池子")。
"""
import numpy as np
import pandas as pd

from core import config, data


def candidates_at(panels, fin, t):
    """可交易 × 质量候选集 (PIT): 有交易/流动性/股价红线/非一字涨停/5年双正"""
    if t not in panels["active"].index:
        return []
    act = panels["active"].loc[t]
    act = act[act.fillna(False)]
    if not len(act):
        return []
    amt = panels["amount20"].loc[t].reindex(act.index)
    real = panels["real_close"].loc[t].reindex(act.index)
    keep = (amt >= config.LIQ_THRESHOLD) & (real >= config.MIN_PRICE)
    cand = [s for s in keep[keep].index if (t, s) not in panels["limit_up"]]
    qual = data.quality_syms(fin, t)
    return [s for s in cand if s in qual]


def score_lowatt(panels, fin, t, cand):
    """0.6×低成交额分位 + 0.4×现金转换分位 (越不被关注+盈利含金量越高分越高)"""
    if not cand:
        return pd.Series(dtype=float)
    amt = panels["amount20"].loc[t].reindex(cand)
    cc = data.cash_conversion(fin, t).reindex(cand)
    return 0.6 * (-amt).rank(pct=True) + 0.4 * cc.rank(pct=True)


def score_accel(fin, t, cand):
    """0.4×盈利同比 + 0.35×加速度 + 0.25×FCF确认 (年度PIT代理)"""
    g = data.growth_frame(fin, t)
    if not len(g):
        return pd.Series(dtype=float)
    g = g.reindex(cand)
    r = g.rank(pct=True)
    return 0.4 * r["g_np"] + 0.35 * r["accel"] + 0.25 * r["g_fcf"]


def build_rebalances(panels, fin, signals, mode, topk=None):
    """mode: 'qual_ew'|'lowatt'|'accel' → (rebalances, sig_scores, picks)"""
    topk = topk or config.TOPK
    rebals, scores, picks = [], {}, []
    for t, e in signals:
        cand = candidates_at(panels, fin, t)
        if mode == "qual_ew":
            scores[t] = pd.Series(dtype=float)          # 无打分, IC 不适用
            picks.append(set(cand))
            tgt = {s: 1.0 / len(cand) for s in cand} if cand else {}
        else:
            if mode == "lowatt":
                s = score_lowatt(panels, fin, t, cand)
            elif mode == "accel":
                s = score_accel(fin, t, cand)
            else:
                raise ValueError(f"未知 mode: {mode}")
            s = s.dropna()
            scores[t] = s
            pick = list(s.sort_values(ascending=False).head(topk).index)
            picks.append(set(pick))
            tgt = {x: 1.0 / len(pick) for x in pick} if pick else {}
        rebals.append((e, tgt))
    return rebals, scores, picks
