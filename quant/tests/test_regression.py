#!/usr/bin/env python3
"""
tests/test_regression.py —— 数值回归测试 (防重构漂移的最后一道闸)
================================================================================
无穿越守卫单测 (test_no_lookahead.py) 保证"逻辑正确"; 本文件保证"数字不变"。
任何重构 / 依赖升级 / 口径改动导致下列冻结数值变化, 测试立刻失败 —— 这是唯一能
拦住"改代码把收益改高了自己却没发现"的机制。

冻结基准 (2026-07-31 定版, 剔2020 对齐口径年化):
    ENS(XGB+LGB集成) Top10 = 24.88%      ← 主结果, run_rolling
    DoubleEnsemble   Top10 = 26.48%      ← Phase B 上线腿, deC/deD 双跑一致
    value_comp       Top20 = 23.00%      ← 冻结对照基线
    十年双正池等权          =  5.19%      ← 池子本身收益(选股超额的分母)
    沪深300                = -2.35%

分两级:
  · 轻量 (默认跑): 纯函数数值锚定 + 组合层规则 + 产出文件里的冻结指标, 秒级, 无需 qlib
  · 重型 (QUANT_HEAVY=1): 从 checkpoint 持仓重算回测指标, 需 qlib 行情, 约 1 分钟
    —— 这一级才真正锁住 core.backtest 的算法本身

用法 (无需 pytest, 标准库 unittest):
    cd quant
    PYTHONPATH=. python3 -m unittest tests.test_regression -v
    QUANT_HEAVY=1 PYTHONPATH=. python3 -m unittest tests.test_regression -v
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core import config
from core import portfolio as pf
from core.backtest import calc_metrics, annualized_since
from core.pipeline import zscore_mean

HEAVY = os.environ.get("QUANT_HEAVY") == "1"

# ---- 冻结基准: 剔2020 年化 (来自 outputs/*.csv, 见文件头说明) ----
FROZEN = {
    "rolling10y_summary.csv": {
        "集成ENS Top10": 0.24880513910671476,
        "XGB Top10": 0.20544234233333625,
        "LGB Top10": 0.18551627999700226,
        "value_comp Top20": 0.23003873871012326,
        "十年双正池等权": 0.05194042325790815,
        "沪深300基准": -0.023507161617583483,
    },
    "benchmark_phaseB_deC.csv": {
        "DoubleEnsemble Top10": 0.2647595418724138,
        "value_comp Top20": 0.23113128099785674,
        "十年双正池等权": 0.05193990582428332,
    },
}
TOL = 1e-9      # 冻结数值按浮点精度比对, 不给"差一点点"留余地


def _read_summary(fname):
    p = f"{config.OUT_DIR}/{fname}"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, sep="\t")
    return dict(zip(df["策略"], df["年化(剔2020)"]))


class TestFrozenMetrics(unittest.TestCase):
    """产出文件里的冻结指标未被覆盖/漂移"""

    def test_frozen_annualized(self):
        checked = 0
        for fname, expect in FROZEN.items():
            got = _read_summary(fname)
            if got is None:
                self.skipTest(f"{fname} 不存在 (尚未跑出该结果)")
            for strat, val in expect.items():
                self.assertIn(strat, got, f"{fname} 缺少策略行 {strat}")
                self.assertAlmostEqual(
                    float(got[strat]), val, delta=TOL,
                    msg=f"\n{fname} [{strat}] 剔2020年化漂移: "
                        f"冻结 {val:.10f} → 现在 {float(got[strat]):.10f}\n"
                        f"    若为有意改动, 请在 FROZEN 里更新并在 README 记录原因。")
                checked += 1
        self.assertGreater(checked, 0, "一个冻结指标都没校验到")

    def test_de_beats_value_comp(self):
        """上线组合的前提: DE 腿必须显著强于纯估值对照 (否则模型没价值)"""
        got = _read_summary("benchmark_phaseB_deC.csv")
        if got is None:
            self.skipTest("deC 结果不存在")
        de, v20 = float(got["DoubleEnsemble Top10"]), float(got["value_comp Top20"])
        self.assertGreater(de - v20, 0.02, f"DE({de:.4f}) 对 V20({v20:.4f}) 优势 <2pt, 上线前提不成立")

    def test_stock_picking_alpha(self):
        """选股超额: 组合腿必须远高于池子等权 (否则收益只是池子筛出来的, 与模型无关)"""
        got = _read_summary("benchmark_phaseB_deC.csv")
        if got is None:
            self.skipTest("deC 结果不存在")
        self.assertGreater(float(got["DoubleEnsemble Top10"]) - float(got["十年双正池等权"]), 0.15,
                           "DE 相对池子等权的选股超额 <15pt, 与历史基准不符")


class TestMetricMath(unittest.TestCase):
    """指标函数本身的数值锚定 (纯函数, 无外部依赖)"""

    def setUp(self):
        # 固定序列: 244 个交易日, 日收益 0.1% → 年化应精确为 1.001**244 - 1
        idx = pd.bdate_range("2021-01-01", periods=244)
        self.r = pd.Series(0.001, index=idx)

    def test_annualized_exact(self):
        m = calc_metrics(self.r)
        self.assertAlmostEqual(m["ar"], 1.001 ** 244 - 1, places=12)
        self.assertAlmostEqual(m["total"], 1.001 ** 244 - 1, places=12)
        self.assertEqual(m["mdd"], 0.0, "单调上涨序列回撤必须为 0")

    def test_annualized_since_alignment(self):
        """剔2020 口径: 起点之后的子区间年化, 与全区间一致(常数收益序列下)"""
        idx = pd.bdate_range("2020-01-01", periods=488)
        r = pd.Series(0.001, index=idx)
        self.assertAlmostEqual(annualized_since(r, "2021-01-01"), 1.001 ** 244 - 1, places=6)

    def test_mdd_and_sharpe_sign(self):
        r = pd.Series([0.1, -0.5, 0.2], index=pd.bdate_range("2021-01-01", periods=3))
        m = calc_metrics(r)
        self.assertLess(m["mdd"], -0.4, "50% 单日跌幅必须体现在回撤里")
        self.assertLess(m["sharpe"], 0, "负收益 Sharpe 必须为负")

    def test_zscore_mean_is_denoising_not_averaging(self):
        """种子集成: 量纲不同的两个预测融合后, 不应被方差大的那个主导"""
        idx = pd.MultiIndex.from_product([pd.bdate_range("2021-01-01", periods=2),
                                          ["A", "B", "C", "D"]], names=["datetime", "instrument"])
        small = pd.Series([1, 2, 3, 4] * 2, index=idx, dtype=float)          # 小量纲
        big = pd.Series([400, 300, 200, 100] * 2, index=idx, dtype=float)    # 大量纲, 排序相反
        z = zscore_mean([small, big])
        # 两者排序完全相反且 z-score 后量纲相同 → 融合结果应接近 0 (互相抵消)
        self.assertLess(float(z.abs().max()), 1e-9,
                        "z-score 融合未消除量纲差异, 大数值预测仍在主导")
        # 直接平均则会被 big 主导 (反向), 用于证明 z-score 是必要的
        naive = (small + big) / 2
        self.assertGreater(naive.loc[(idx[0][0], "A")], naive.loc[(idx[0][0], "D")],
                           "对照: 朴素平均确实被大量纲主导")


class TestPortfolioRules(unittest.TestCase):
    """组合层硬规则: 这些是真金白银, 任何一条破了都会导致下单失败或超买"""

    def setUp(self):
        self.picks = {
            "DE": [f"SZ{i:06d}" for i in range(1, 21)],
            "V20": [f"SH6{i:05d}" for i in range(1, 41)],
        }
        de_px = [12.5, 80.0, 33.3, 5.2, 199.0, 45.0, 7.7, 22.1, 61.0, 15.5]
        v_px = [6.1, 18.9, 3.4, 55.0, 11.2, 9.8, 27.5, 4.6, 120.0, 14.3]
        self.prices = {}
        for i, c in enumerate(self.picks["DE"]):
            self.prices[c] = de_px[i % 10]
        for i, c in enumerate(self.picks["V20"]):
            self.prices[c] = v_px[i % 10]

    def test_never_overdraw(self):
        """绝不透支: 各资金档实投必须 ≤ 总资金"""
        for cap in [20000, 30000, 50000, 100000, 200000, 500000, 1000000]:
            _, s = pf.allocate(self.picks, self.prices, cap)
            self.assertLessEqual(s["invested"], cap + 1e-6,
                                 f"资金 {cap} 下实投 {s['invested']:.0f} 超过总资金!")

    def test_cash_drag_bounded(self):
        """现金拖累: 回测是满仓口径, 5万及以上闲置现金须 <5%"""
        for cap in [50000, 100000, 200000]:
            _, s = pf.allocate(self.picks, self.prices, cap)
            self.assertLess(s["cash_pct"], 0.05,
                            f"资金 {cap} 下闲置现金 {s['cash_pct']*100:.1f}% ≥5%, 现金拖累过大")

    def test_expensive_stock_is_skipped(self):
        """一手买不起必须顺延, 不能硬买破坏等权 (5万下 DE 腿目标仓位 4000 元)"""
        orders, s = pf.allocate(self.picks, self.prices, 50000)
        bought = set(orders["instrument"])
        self.assertNotIn("SZ000005", bought, "199 元/股(一手19900) 在4000元目标下必须被跳过")
        self.assertTrue(any("一手" in x["reason"] for x in s["skipped"]), "跳过原因未记录")

    def test_sleeve_topk_filled(self):
        """两条腿都必须填满 (DE 5 只 + V20 10 只 = 15 只)"""
        orders, _ = pf.allocate(self.picks, self.prices, 100000)
        n = orders.groupby("sleeve").size().to_dict()
        self.assertEqual(n["DE"], pf.SLEEVES["DE"]["topk"])
        self.assertEqual(n["V20"], pf.SLEEVES["V20"]["topk"])

    def test_sleeve_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(v["weight"] for v in pf.SLEEVES.values()), 1.0, places=9)

    def test_bad_weights_rejected(self):
        with self.assertRaises(ValueError):
            pf.allocate(self.picks, self.prices, 50000,
                        sleeves={"DE": {"weight": 0.5, "topk": 5}})   # 合计 0.5 != 1

    def test_overlap_merged_not_duplicated(self):
        """同股被两腿选中 → 合并成一笔, 股数相加 (自然加仓)"""
        picks = {k: list(v) for k, v in self.picks.items()}
        picks["V20"][0] = picks["DE"][0]
        orders, _ = pf.allocate(picks, self.prices, 100000)
        merged = pf.merge_overlap(orders)
        self.assertEqual(merged["instrument"].nunique(), len(merged), "合并后仍有重复标的")
        dup = picks["DE"][0]
        self.assertEqual(int(merged.loc[merged["instrument"] == dup, "shares"].iloc[0]),
                         int(orders.loc[orders["instrument"] == dup, "shares"].sum()),
                         "重合股股数未相加")

    def test_all_shares_are_whole_lots(self):
        orders, _ = pf.allocate(self.picks, self.prices, 50000)
        self.assertTrue((orders["shares"] % pf.LOT == 0).all(), "存在非整手股数, A股无法下单")

    def test_empty_sleeve_rejected(self):
        with self.assertRaises(RuntimeError):
            pf.allocate({"DE": [], "V20": self.picks["V20"]}, self.prices, 50000)

    def test_turnover_first_build_is_full(self):
        """首次建仓 (无持仓) 换手率应为 100%"""
        orders, _ = pf.allocate(self.picks, self.prices, 100000)
        merged = pf.merge_overlap(orders)
        _, t = pf.diff_positions(merged, {})
        self.assertAlmostEqual(t["turnover"], 0.5, places=6,
                               msg="无持仓建仓的单边换手口径应为 0.5(双边/2)")

    def test_turnover_no_change_is_zero(self):
        """持仓与目标完全一致 → 换手 0, 无摩擦成本"""
        orders, _ = pf.allocate(self.picks, self.prices, 100000)
        merged = pf.merge_overlap(orders)
        held = dict(zip(merged["instrument"], merged["shares"].astype(int)))
        d, t = pf.diff_positions(merged, held)
        self.assertEqual(t["turnover"], 0.0)
        self.assertEqual(t["est_cost"], 0.0)
        self.assertTrue((d["action"] == "持有").all())


@unittest.skipUnless(HEAVY, "重型回归 (需 qlib 行情): 设 QUANT_HEAVY=1 开启")
class TestBacktestReplay(unittest.TestCase):
    """从 checkpoint 的冻结持仓重算回测 → 锁住 core.backtest 算法本身。
    这是最强的一道闸: 持仓不变而指标变了, 说明回测引擎被改动过。"""

    def test_replay_deC_matches_frozen(self):
        import pickle
        from core import data as datalayer
        from core.backtest import portfolio_backtest

        ck = f"{config.OUT_DIR}/_phaseB_ckpt_deC.pkl"
        if not os.path.exists(ck):
            self.skipTest("deC checkpoint 不存在")
        with open(ck, "rb") as f:      # 本进程自产的 checkpoint, 非外部来源
            d = pickle.load(f)
        reb = d["rebalances"]["DoubleEnsemble"]
        self.assertGreater(len(reb), 70, "deC 调仓期数异常 (应为 79 期)")

        datalayer.init_qlib()
        insts = sorted({i for _, tops in reb for i in tops})
        pm = datalayer.load_price_matrix(insts)
        rets, _, _ = portfolio_backtest(reb, pm)
        got = annualized_since(rets, config.ALIGN_START)
        self.assertAlmostEqual(
            got, FROZEN["benchmark_phaseB_deC.csv"]["DoubleEnsemble Top10"], delta=1e-6,
            msg=f"\n从冻结持仓重算的剔2020年化 = {got:.10f}, "
                f"与冻结值 {FROZEN['benchmark_phaseB_deC.csv']['DoubleEnsemble Top10']:.10f} 不符\n"
                f"    → core.backtest 或 core.data 的行为已改变!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
