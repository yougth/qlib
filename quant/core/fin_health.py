#!/usr/bin/env python3
"""
fin_health —— 财务健康过滤 (ROIC / 利息保障 / 资产负债率), PIT 无穿越
================================================================================
动机: "十年双正"只保证利润和现金流为正, 不保证资本效率与偿债安全:
  陷阱1 低效繁荣: 赚1亿需投10亿 → ROIC 长期低于资本成本, 毁灭股东价值
  陷阱2 杠杆走钢丝: 有息负债攀升, 信贷收紧时现金牛也会休克
规则 (奥卡姆剃刀式减法, 不加新因子):
  1) 近3年平均 ROIC > 8%   (约覆盖股权资本成本, 剔除重资产堆砌的伪白马)
  2) 利息保障倍数 EBIT/利息 > 3 或 资产负债率 < 60% (只留内生造血的公司)
PIT 口径 (与底池一致): 回测年 Y 用 Y-2..Y-4 三个年报 (Y-1 年报在 Y 年 3-4 月
才公布, 按现有池 fcf_years=Y-11..Y-2 的保守口径, 不使用);
A股另有精确 notice_date 双保险校验, 港股无公告日字段, 保守规则本身已无穿越。
金融股豁免: 银行/保险/证券的商业模式就是高杠杆经营, 两规则均不适用。
缺数据非金融股: 保守剔除 (减法原则宁缺毋滥)。
"""
import os
import pandas as pd

ROIC_FLOOR = 0.08        # 近3年平均 ROIC 下限
INT_COVER_FLOOR = 3.0    # 利息保障倍数下限
DEBT_RATIO_CEIL = 0.60   # 资产负债率上限 (与利息保障二选一)

IND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "data_cache", "industry_map.csv")
FIN_KW = ("银行", "保险", "证券", "信托", "金融")


def load_fin_codes(path=IND_PATH):
    """行业口径识别金融股 (A股 EM2016 如'金融-银行-国有银行', 港股如'银行')"""
    if not os.path.exists(path):
        return set()
    ind = pd.read_csv(path, dtype={"code": str})
    m = ind["industry"].fillna("").astype(str)
    return set(ind.loc[m.str.contains("|".join(FIN_KW)), "code"])


def load_fin_health(path):
    """加载 fetch_fin_health.py 产出的缓存"""
    return pd.read_parquet(path)


def _yearly_roic(r):
    """单年 ROIC = NOPAT / 投入资本; NOPAT = EBIT×(1-有效税率)"""
    ebit, te, debt, tax = r["ebit"], r["te"], r["int_debt"], r["tax"]
    if not ebit or ebit <= 0 or te is None or te <= 0:
        return None
    ic = te + (debt or 0.0)
    if ic <= 0:
        return None
    # 有效税率 = 税项/除税前利润; 缺失或异常时按 25%
    etr = (tax / ebit) if (tax is not None and ebit > 0 and 0 <= tax / ebit <= 1) else 0.25
    return ebit * (1 - etr) / ic


def filter_pool(codes, backtest_year, fin_df, fin_codes):
    """对整个池应用财务健康过滤, 返回 (通过列表, 剔除统计)"""
    kept, dropped_debt, dropped_roic = [], [], []
    for c in codes:
        if c in fin_codes:                              # 金融股豁免
            kept.append(c)
            continue
        g = fin_df[(fin_df["code"] == c)
                   & (fin_df["year"].between(backtest_year - 4, backtest_year - 2))]
        if len(g) < 2:
            dropped_debt.append(c)                      # 数据不足, 保守剔除
            continue
        roics = [x for x in (_yearly_roic(r) for _, r in g.iterrows()) if x is not None]
        if len(roics) < 2 or sum(roics) / len(roics) <= ROIC_FLOOR:
            dropped_roic.append(c)
            continue
        ic_rows = g.dropna(subset=["ebit"])
        cover = [r["ebit"] / r["int_exp"] for _, r in ic_rows.iterrows()
                 if r.get("int_exp") and r["int_exp"] > 0]
        latest = g.sort_values("year").iloc[-1]
        dr_ok = (latest["ta"] and latest["tl"] is not None
                 and latest["tl"] / latest["ta"] < DEBT_RATIO_CEIL)
        cover_ok = bool(cover) and sum(cover) / len(cover) > INT_COVER_FLOOR
        if cover_ok or dr_ok:
            kept.append(c)
        else:
            dropped_debt.append(c)
    stats = {"kept": len(kept), "roic_fail": len(dropped_roic),
             "debt_fail": len(dropped_debt),
             "dropped": sorted(dropped_roic + dropped_debt)}
    return kept, stats
