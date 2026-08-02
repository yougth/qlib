"""
core.portfolio —— 组合层: sleeve 分仓 → 实际可下单股数 (回测与实盘共用)
================================================================================
研究层输出的是"买哪几只"(排序名单), 这里负责把它翻译成"各买多少股":
  · sleeve 分仓: DE 腿与 V20 腿资金各自独立, 不做信号融合 (重合股票自然加仓)
  · A 股一手 = 100 股, 小资金下"一手买不起"是硬约束 → 顺延下一名 (见 LOT_SKIP_RATIO)
  · 零股余钱留现金, 不强行凑单

与回测口径的关系: 回测按等权 + 单边换手 × FEE_ROUNDTRIP 摩擦近似; 本模块是实盘落地,
一手取整/顺延会带来与等权的偏差 (小资金下不可避免), 偏差量在输出里显式给出。
"""
import numpy as np
import pandas as pd

from . import config

# 上线组合: DE 腿(每月轮动, 5只) 40% + V20 腿(价值核心, 10只) 60%
# 依据: 15 只持仓 / 换手 46.2%/月 / 剔2020 年化 28.9% (见 README 上线方案)
SLEEVES = {
    "DE": {"weight": 0.40, "topk": 5},
    "V20": {"weight": 0.60, "topk": 10},
}

LOT = 100                # A股一手 = 100 股
LOT_SKIP_RATIO = 1.5     # 一手金额 > 目标仓位 × 此比例 → 买不起, 顺延下一名
MIN_TICKET = 3000        # 单笔低于此金额时佣金占比过高 (免5券商下 5/3000 = 0.17%)


def allocate(ranked_picks, prices, capital, sleeves=None,
             lot=LOT, skip_ratio=LOT_SKIP_RATIO):
    """把各 sleeve 的排序名单翻译成可下单股数。

    ranked_picks: {sleeve: [instrument, ...]}  按分数降序, 长度应 > topk 以支持顺延
    prices:       {instrument: 最新收盘价}      下单价基准 (T+1 收盘成交口径)
    capital:      总投入资金 (元)
    返回 (orders_df, summary_dict)
      orders_df: sleeve/rank/instrument/price/lots/shares/amount/weight_in_sleeve
      summary:   各 sleeve 实投/目标, 现金余额, 跳过的股票及原因
    """
    sleeves = sleeves or SLEEVES
    w_sum = sum(s["weight"] for s in sleeves.values())
    if abs(w_sum - 1.0) > 1e-6:
        raise ValueError(f"[CHECK] sleeve 权重合计={w_sum:.4f} != 1.0")

    rows, skipped = [], []
    for sname, scfg in sleeves.items():
        picks = list(ranked_picks.get(sname, []))
        if not picks:
            raise RuntimeError(f"[CHECK] sleeve {sname} 名单为空, 拒绝静默产出空单!")
        topk = scfg["topk"]
        sleeve_cash = capital * scfg["weight"]
        target_pos = sleeve_cash / topk          # 每只目标仓位(等权)
        remaining = sleeve_cash                  # 钱包余额: 逐笔扣减, 绝不透支
        filled = 0
        for rank, inst in enumerate(picks, 1):
            if filled >= topk:
                break
            px = prices.get(inst)
            if px is None or not np.isfinite(px) or px <= 0:
                skipped.append({"sleeve": sname, "instrument": inst, "reason": "无价格"})
                continue
            lot_amount = px * lot
            # 一手就超出目标仓位太多 → 买它会严重破坏等权, 顺延给下一名
            if lot_amount > target_pos * skip_ratio:
                skipped.append({"sleeve": sname, "instrument": inst,
                                "reason": f"一手{lot_amount:.0f}元 > 目标{target_pos:.0f}×{skip_ratio}"})
                continue
            # 向下取整(不超目标), 至少1手; 再按钱包余额收敛, 买不起就顺延
            lots = max(1, int(target_pos // lot_amount))
            if lots * lot_amount > remaining:
                lots = int(remaining // lot_amount)
            if lots < 1:
                skipped.append({"sleeve": sname, "instrument": inst,
                                "reason": f"腿内余额{remaining:.0f}元不足一手({lot_amount:.0f}元)"})
                continue
            amount = lots * lot_amount
            remaining -= amount
            rows.append({"sleeve": sname, "rank": rank, "instrument": inst,
                         "price": px, "lots": lots, "shares": lots * lot,
                         "amount": amount, "target_pos": target_pos})
            filled += 1
        if filled < topk:
            skipped.append({"sleeve": sname, "instrument": "-",
                            "reason": f"候选耗尽, 仅填 {filled}/{topk} 只 (名单需更长)"})
        # 余钱加仓: 向下取整会留下大量闲置现金(小资金下可达 10%+), 而回测是满仓口径,
        # 现金拖累会直接吃掉收益 → 用剩余资金给"当前离目标最远"的持仓补一手, 逼近等权
        mine = [r for r in rows if r["sleeve"] == sname]
        while mine:
            affordable = [r for r in mine if r["price"] * lot <= remaining]
            if not affordable:
                break
            r = min(affordable, key=lambda x: x["amount"] / x["target_pos"])
            step = r["price"] * lot
            # 补一手后若显著超出目标仓位则不补 (宁留现金, 不破坏等权)
            if (r["amount"] + step) > r["target_pos"] * skip_ratio:
                mine = [x for x in mine if x is not r]
                continue
            r["lots"] += 1
            r["shares"] += lot
            r["amount"] += step
            remaining -= step

    orders = pd.DataFrame(rows)
    if orders.empty:
        raise RuntimeError("[CHECK] 全部候选被过滤, 无可下单标的!")
    orders["weight_in_sleeve"] = orders["amount"] / orders.groupby("sleeve")["amount"].transform("sum")
    orders["weight_total"] = orders["amount"] / capital
    orders["small_ticket"] = orders["amount"] < MIN_TICKET

    invested = float(orders["amount"].sum())
    summary = {
        "capital": capital,
        "invested": invested,
        "cash_left": capital - invested,
        "cash_pct": (capital - invested) / capital,
        "n_positions": int(orders["instrument"].nunique()),
        "n_orders": len(orders),
        "sleeve_invested": orders.groupby("sleeve")["amount"].sum().to_dict(),
        "sleeve_target": {k: capital * v["weight"] for k, v in sleeves.items()},
        "skipped": skipped,
        # 等权偏差: 一手取整后各仓位与目标的最大偏离, 用于判断资金是否够用
        "max_weight_dev": float((orders["amount"] / orders["target_pos"] - 1).abs().max()),
    }
    return orders, summary


def merge_overlap(orders):
    """同一股票被两个 sleeve 同时选中 → 合并为一笔下单 (自然加仓)"""
    if orders.empty:
        return orders
    g = orders.groupby("instrument", as_index=False).agg(
        price=("price", "first"), shares=("shares", "sum"), amount=("amount", "sum"),
        sleeves=("sleeve", lambda s: "+".join(sorted(set(s)))))
    g["lots"] = g["shares"] // LOT
    return g.sort_values("amount", ascending=False).reset_index(drop=True)


def diff_positions(target_orders, held):
    """目标持仓 vs 当前持仓 → 调仓指令 (卖出/买入/加减仓) 与实际换手率。

    target_orders: merge_overlap 的输出 (instrument/shares/price)
    held: {instrument: shares} 当前实际持仓
    """
    tgt = dict(zip(target_orders["instrument"], target_orders["shares"]))
    px = dict(zip(target_orders["instrument"], target_orders["price"]))
    held = {k: v for k, v in (held or {}).items() if v}
    rows = []
    for inst in sorted(set(tgt) | set(held)):
        t, h = tgt.get(inst, 0), held.get(inst, 0)
        d = t - h
        if d == 0:
            act = "持有"
        elif h == 0:
            act = "买入"
        elif t == 0:
            act = "清仓"
        else:
            act = "加仓" if d > 0 else "减仓"
        rows.append({"instrument": inst, "held": h, "target": t, "delta": d,
                     "action": act, "price": px.get(inst, np.nan)})
    df = pd.DataFrame(rows)
    # 换手率: 双边成交额 / 2 / 目标市值 (与回测单边换手口径一致)
    tv = float(target_orders["amount"].sum()) if len(target_orders) else 0.0
    traded = float((df["delta"].abs() * df["price"].fillna(0)).sum())
    turnover = (traded / 2 / tv) if tv > 0 else 0.0
    cost = traded / 2 * config.FEE_ROUNDTRIP
    return df, {"turnover": turnover, "traded_amount": traded, "est_cost": cost}
