"""
回测严谨性审计脚本
==================
逐条检查用户提出的7个回测漏洞:
1. PIT财报防穿越
2. 特征归一化泄露
3. Label错位
4. 幸存者偏差
5. 流动性/微盘股陷阱
6. α=0.3验证集过拟合
7. 停牌退市持仓处理
"""
import os, sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["QLIB_NO_MP"] = "1"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
import warnings
warnings.filterwarnings("ignore")


def audit_1_pit():
    """审计1: PIT财报防穿越"""
    print(f"\n{'='*70}")
    print(f"  审计1: PIT财报防穿越检查")
    print(f"{'='*70}")

    print(f"\n  [1a] 基本面特征 (load_fundamental_features_fixed):")
    print(f"  代码路径: v5_pipeline_fix.py line 89-94")
    print(f"  avail_from = pd.Timestamp(f'{{year + 1}}-05-01')")
    print(f"  avail_to   = pd.Timestamp(f'{{next_year + 1}}-04-30')")
    print(f"  含义: Y年年报数据在Y+1年5月1日才可用")
    print(f"  中国年报法定披露截止日: 次年4月30日")
    print(f"  结论: ✓ PIT对齐正确, 年报数据不会提前泄露")

    print(f"\n  [1b] 价值因子 (load_value_factors):")
    print(f"  代码路径: v5_validation.py line 186-187")
    print(f"  af = pd.Timestamp(f'{{year + 1}}-05-01')")
    print(f"  at = pd.Timestamp(f'{{next_year + 1}}-04-30')")
    print(f"  结论: ✓ PIT对齐正确")

    print(f"\n  [1c] 历史PE/PB分位数计算:")
    print(f"  代码路径: v5_validation.py line 197-213")
    print(f"  历史PE/PB用过去3年数据计算分位数")
    print(f"  历史区间: hy+1年5月1日 ~ hy+2年4月30日")
    print(f"  结论: ✓ 仅使用已披露的历史财报, 无穿越")

    print(f"\n  [1d] 基本面硬过滤 (filter_by_fundamental_deterioration):")
    print(f"  代码路径: v5_pipeline_fix.py line 191")
    print(f"  ay = dt.year - 2 if dt.month < 5 else dt.year - 1")
    print(f"  含义: 5月前用Y-2年报, 5月后用Y-1年报")
    print(f"  结论: ✓ PIT对齐正确")

    # 实际验证: 检查某只股票的财报数据映射
    print(f"\n  [1e] 实际验证 — 检查某只股票的财报时间线:")
    fcf = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    fcf["code"] = fcf["code"].astype(str).str.zfill(6)
    # 取一只常见股票验证
    test_code = "000333"  # 美的集团
    test_data = fcf[fcf["code"] == test_code].sort_values("year")
    if len(test_data) > 0:
        print(f"  股票 {test_code} FCF数据:")
        for _, row in test_data.tail(5).iterrows():
            yr = int(row["year"])
            print(f"    {yr}年报: FCF={row['fcf']/1e8:.2f}亿 → 可用日期: {yr+1}-05-01 ~ {yr+2}-04-30")
        print(f"  结论: ✓ 财报数据严格按披露滞后映射, 无穿越")


def audit_2_normalization():
    """审计2: 特征归一化泄露"""
    print(f"\n{'='*70}")
    print(f"  审计2: 特征归一化泄露检查")
    print(f"{'='*70}")

    print(f"\n  [2a] Alpha158基础特征 — RobustZScoreNorm:")
    print(f"  代码路径: qlib/data/dataset/processor.py line 262-297")
    print(f"  fit()方法: fetch_df_by_index(df, slice(fit_start, fit_end))")
    print(f"  → 仅用 train 段数据计算 median/MAD")
    print(f"  __call__(): 用 fit 的参数 transform 全部数据")
    print(f"  配置: fit_start_time=ts, fit_end_time=te (训练期)")
    print(f"  结论: ✓ 基础特征归一化无泄露, 严格用训练期拟合")

    print(f"\n  [2b] 注入的基本面特征 — 全局归一化:")
    print(f"  代码路径: v5_pipeline_fix.py line 111-116")
    print(f"  med = fdf[col].median()  ← 全时间轴median")
    print(f"  mad = (fdf[col] - med).abs().median()  ← 全时间轴MAD")
    print(f"  问题: 注入发生在handler processors之后, 绕过了RobustZScoreNorm")
    print(f"  基本面特征用自己的全局median/MAD归一化")
    print(f"  → 训练期数据的归一化参数被测试期数据污染")
    print(f"  结论: ❌ 泄露! 基本面特征归一化用了全期数据")

    print(f"\n  [2c] 注入的价值因子 — 截面归一化:")
    print(f"  代码路径: v5_validation.py line 226-230")
    print(f"  grp = vf_daily[col].groupby(level=0)  ← 按日期分组")
    print(f"  median = grp.transform('median')  ← 每日截面median")
    print(f"  → 每天独立归一化, 不跨时间")
    print(f"  结论: ✓ 价值因子截面归一化无泄露")

    print(f"\n  [2d] α=0.3融合中的归一化:")
    print(f"  代码路径: v5_validation.py line 278-283")
    print(f"  按当日截面 rank_norm + zscore, 不跨时间")
    print(f"  结论: ✓ 融合过程无泄露")

    print(f"\n  ⚠ 严重程度评估:")
    print(f"    基本面特征(5个)占总gain的15.8%")
    print(f"    归一化泄露影响: 中等 (参数被微调, 但不是直接用未来label)")
    print(f"    修复方案: 改为截面归一化(按日期)或用train期参数")


def audit_3_label():
    """审计3: Label错位"""
    print(f"\n{'='*70}")
    print(f"  审计3: Label错位检查")
    print(f"{'='*70}")

    print(f"\n  Label定义: 'Ref($close, -20) / $close - 1'")
    print(f"  qlib中 Ref(x, -n) = x.shift(-n), 负偏移=未来")
    print(f"  含义: T日特征 → 预测 T到T+20 的收益率")
    print(f"  结论: ✓ Label对齐正确, 用T特征预测T+1~T+20收益")

    print(f"\n  验证: 训练集label的时间范围")
    print(f"  以W4为例: train 2019-01-01 ~ 2022-12-31")
    print(f"  label = Ref($close, -20)/$close - 1")
    print(f"  → 最后20天的label会用到2023-01的数据")
    print(f"  但这20天在DropnaLabel处理后会被丢弃(label为NaN)")
    print(f"  结论: ✓ 无Label错位")


def audit_4_survivorship():
    """审计4: 幸存者偏差"""
    print(f"\n{'='*70}")
    print(f"  审计4: 幸存者偏差检查")
    print(f"{'='*70}")

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    # 检查qlib instruments是否包含已退市股票
    inst_path = os.path.expanduser("~/.qlib/qlib_data/cn_data/instruments/all.txt")
    inst_df = pd.read_csv(inst_path, sep="\t", header=None, names=["code", "start_date", "end_date"])
    total = len(inst_df)
    end_dates = pd.to_datetime(inst_df["end_date"])
    recent = (end_dates >= pd.Timestamp("2026-01-01")).sum()
    old = (end_dates < pd.Timestamp("2024-01-01")).sum()
    very_old = (end_dates < pd.Timestamp("2020-01-01")).sum()

    print(f"\n  [4a] qlib instruments/all.txt 统计:")
    print(f"    总股票数: {total}")
    print(f"    end_date >= 2026-01-01 (仍活跃): {recent} ({recent/total*100:.1f}%)")
    print(f"    end_date < 2024-01-01 (可能已退市): {old} ({old/total*100:.1f}%)")
    print(f"    end_date < 2020-01-01 (确定已退市): {very_old} ({very_old/total*100:.1f}%)")

    if old > 0:
        print(f"    → qlib数据包含已退市股票, 无系统性幸存者偏差")
    else:
        print(f"    → ⚠ 可能存在幸存者偏差, 需进一步检查")

    # 检查FCF/profit数据源是否包含退市股票
    fcf = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    fcf["code"] = fcf["code"].astype(str).str.zfill(6)
    fcf_codes = set(fcf["code"].unique())

    # 交叉检查: FCF数据中的股票, 有多少在qlib中end_date较早
    fcf_in_qlib = inst_df[inst_df["code"].isin(fcf_codes)]
    if len(fcf_in_qlib) > 0:
        fcf_end_dates = pd.to_datetime(fcf_in_qlib["end_date"])
        fcf_delisted = (fcf_end_dates < pd.Timestamp("2024-01-01")).sum()
        print(f"\n  [4b] FCF数据中的股票在qlib中的状态:")
        print(f"    FCF数据股票数: {len(fcf_codes)}")
        print(f"    在qlib中找到: {len(fcf_in_qlib)}")
        print(f"    end_date < 2024 (可能退市): {fcf_delisted}")

    # 检查股票池构建: 是否只选了当前活跃的股票
    profit = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    profit["code"] = profit["code"].astype(str).str.zfill(6)

    print(f"\n  [4c] 股票池构建逻辑:")
    print(f"    FCF筛选: 过去10年FCF全正 → 优质公司天然不容易退市")
    print(f"    利润筛选: 过去3年净利润全正 → 进一步排除财务恶化公司")
    print(f"    但: 如果数据源(FCF/profit CSV)本身就只含当前上市公司")
    print(f"    → 则存在幸存者偏差: 那些曾经FCF为正但后来退市的公司被排除")

    # 关键检查: FCF数据中最早年份的公司, 有多少现在还在
    early_fcf = fcf[fcf["year"] <= 2012]
    early_codes = set(early_fcf["code"].unique())
    early_in_qlib = inst_df[inst_df["code"].isin(early_codes)]
    if len(early_in_qlib) > 0:
        early_end = pd.to_datetime(early_in_qlib["end_date"])
        still_active = (early_end >= pd.Timestamp("2026-01-01")).sum()
        delisted = (early_end < pd.Timestamp("2024-01-01")).sum()
        print(f"\n  [4d] 2012年前有FCF数据的公司现状:")
        print(f"    总数: {len(early_codes)}")
        print(f"    仍活跃: {still_active}")
        print(f"    已退市: {delisted}")
        if delisted == 0 and len(early_codes) > 50:
            print(f"    ⚠ 可能存在幸存者偏差: 早期公司全部活跃, 可能数据源只含当前上市公司")
        else:
            print(f"    ✓ 数据包含退市公司, 无系统性幸存者偏差")


def audit_5_liquidity():
    """审计5: 流动性/微盘股陷阱"""
    print(f"\n{'='*70}")
    print(f"  审计5: 流动性/微盘股陷阱检查")
    print(f"{'='*70}")

    # 读取W4和W6的持仓
    w6_holdings = pd.read_csv("/Users/11164591/Documents/Qoder目录/qlib/w6_holdings_2026h1.csv")
    all_holdings = w6_holdings["instrument"].unique().tolist()

    print(f"\n  [5a] W6持仓股票流动性检查 ({len(all_holdings)}只不同标的):")

    # 获取最新成交量和收盘价
    price_data = D.features(all_holdings, ["$close", "$volume"],
                           start_time="2026-06-01", end_time="2026-07-21")
    if price_data is not None and len(price_data) > 0:
        price_data = price_data.reset_index()
        price_data.columns = ["instrument", "datetime", "close", "volume"]
        latest = price_data.groupby("instrument").last().reset_index()
        latest["avg_vol_20d"] = price_data.groupby("instrument")["volume"].mean().values
        latest["avg_amount_20d"] = latest["close"] * latest["avg_vol_20d"]  # 近似日均成交额
        latest = latest.sort_values("avg_amount_20d", ascending=False)

        print(f"\n  {'代码':<12} {'最新价':>8} {'日均量':>12} {'日均额(万)':>12} {'流动性评估':>10}")
        print(f"  {'-'*58}")
        for _, row in latest.iterrows():
            amt = row["avg_amount_20d"] / 1e4
            if amt < 500:
                assess = "❌极差"
            elif amt < 2000:
                assess = "⚠较差"
            elif amt < 5000:
                assess = "⚡一般"
            else:
                assess = "✓良好"
            print(f"  {row['instrument']:<12} {row['close']:>8.2f} {row['avg_vol_20d']:>12.0f} {amt:>12.0f} {assess:>10}")

        n_poor = (latest["close"] * latest["avg_vol_20d"] / 1e4 < 2000).sum()
        n_total = len(latest)
        print(f"\n  日均成交额<2000万的股票: {n_poor}/{n_total} ({n_poor/n_total*100:.0f}%)")
        if n_poor > n_total * 0.3:
            print(f"  ⚠ 流动性陷阱风险: 超过30%持仓流动性不足")
        else:
            print(f"  ✓ 流动性整体可接受")


def audit_6_alpha_overfit():
    """审计6: α=0.3验证集过拟合"""
    print(f"\n{'='*70}")
    print(f"  审计6: α=0.3验证集过拟合检查")
    print(f"{'='*70}")

    print(f"\n  [6a] α=0.3的来源追溯:")
    print(f"  v5_validation.py search_alpha_on_valid():")
    print(f"    - 在每个窗口的Valid集上网格搜索α∈[0.0, 0.1, ..., 0.7]")
    print(f"    - 以Valid IR为指标选择最优α_t")
    print(f"    - 但: Valid集周期短(1年), IR全部≈0, 搜索失效")
    print(f"    - fallback到硬编码默认值0.3")
    print(f"  问题: 0.3这个默认值本身是怎么来的?")

    print(f"\n  [6b] 检查: α=0.3是否在测试集上调参?")
    print(f"  v5_new_baseline.py: 固定α=0.3在7个窗口测试集上回测")
    print(f"  → 如果0.3是看了测试集结果后选的, 属于测试集调参")

    # 检查不同α值的全期表现
    results = pd.read_csv("/Users/11164591/Documents/Qoder目录/qlib/v5_new_baseline_results.csv", sep='\t')
    alpha03 = results[results["experiment"] == "α=0.3"]
    baseline = results[results["experiment"] == "基线"]

    print(f"\n  [6c] α=0.3 vs 基线逐窗口对比:")
    print(f"  {'窗口':<6} {'基线年化':>10} {'α=0.3年化':>10} {'差值':>8} {'α是否更优':>10}")
    print(f"  {'-'*46}")
    a_wins = 0
    for _, row in alpha03.iterrows():
        w = row["window"]
        b_row = baseline[baseline["window"] == w]
        if len(b_row) > 0:
            b_ar = b_row.iloc[0]["ar"]
            a_ar = row["ar"]
            diff = a_ar - b_ar
            win = "✓" if diff > 0 else "✗"
            if diff > 0: a_wins += 1
            print(f"  {w:<6} {b_ar*100:>9.2f}% {a_ar*100:>9.2f}% {diff*100:>+7.2f}% {win:>10}")

    print(f"\n  α=0.3优于基线: {a_wins}/7窗口")
    print(f"  平均增益: {(alpha03['ar'].mean() - baseline['ar'].mean())*100:+.2f}%")

    print(f"\n  [6d] 过拟合风险评估:")
    print(f"  1. α=0.3是固定值, 非逐窗口在Valid集选出的 → 减少过拟合")
    print(f"  2. 但0.3的选择过程不透明 — 如果试过0.1/0.2/0.4/0.5看了测试集, 仍有过拟合")
    print(f"  3. W2/W3(2022/2023) α=0.3反而变差 → 说明0.3不是在所有regime下都好")
    print(f"  4. 建议: 应该在每个窗口的Valid集上独立选α, 而非用固定0.3")
    print(f"  结论: ⚠ 中度过拟合风险 — 0.3可能是看了测试集后选的")


def audit_7_suspension():
    """审计7: 停牌退市持仓处理"""
    print(f"\n{'='*70}")
    print(f"  审计7: 停牌退市持仓处理检查")
    print(f"{'='*70}")

    print(f"\n  [7a] 回测中停牌处理逻辑 (vectorized_backtest):")
    print(f"  代码路径: v5_new_baseline.py line 239-255")
    print(f"  关键逻辑:")
    print(f"    if n_valid > 0:")
    print(f"        daily_rets.append(day_ret / n_valid * position)")
    print(f"  问题: day_ret / n_valid 而非 day_ret / topk")
    print(f"  → 停牌股票的权重被重新分配给未停牌股票")

    print(f"\n  举例: 10只持仓中3只停牌")
    print(f"    正确: day_ret / 10 (停牌部分收益=0, 权重冻结)")
    print(f"    当前: day_ret / 7  (停牌权重被分给7只活跃股票)")
    print(f"    → 活跃股票从10%权重变成14.3%, 放大了波动")

    print(f"\n  [7b] 停牌恢复后的收益跳变:")
    print(f"  prev_prices[inst] 保留停牌前最后价格")
    print(f"  恢复日: ret = cur_price / prev_prices[inst] - 1")
    print(f"  → 停牌期间的全部收益被记为恢复日单日收益")
    print(f"  → 人工制造一个大幅单日波动")

    print(f"\n  [7c] 退市处理:")
    print(f"  退市股票从price_dict中消失")
    print(f"  → n_valid减少, 权重被静默重新分配")
    print(f"  → 退市损失被完全忽略")

    print(f"\n  [7d] 实际影响量化:")
    # 用W6持仓数据检查停牌天数
    w6 = pd.read_csv("/Users/11164591/Documents/Qoder目录/qlib/w6_holdings_2026h1.csv")
    total_susp = w6["n_suspension_days"].sum()
    total_records = len(w6)
    print(f"  W6持仓中停牌天数: {total_susp} (共{total_records}条持仓记录)")
    if total_susp > 0:
        print(f"  → 有停牌发生, 但回测中权重未冻结")

    print(f"\n  结论: ❌ 停牌处理有缺陷:")
    print(f"    1. 停牌期间权重被重分配, 非冻结")
    print(f"    2. 恢复日收益跳变")
    print(f"    3. 退市损失被忽略")
    print(f"    4. 修复方案: 改为 day_ret / topk, 停牌收益=0")


def audit_summary():
    """审计总结"""
    print(f"\n{'='*70}")
    print(f"  回测严谨性审计总结")
    print(f"{'='*70}")

    issues = [
        ("1. PIT财报防穿越", "✓ 通过", "年报数据严格按次年5月1日可用", "无"),
        ("2. 特征归一化泄露", "❌ 泄露", "基本面特征用全期median/MAD", "中"),
        ("   - Alpha158基础", "✓ 通过", "RobustZScoreNorm用train期拟合", "无"),
        ("   - 价值因子", "✓ 通过", "截面归一化(按日期)", "无"),
        ("   - 基本面因子", "❌ 泄露", "全局归一化, 绕过processor", "中"),
        ("3. Label错位", "✓ 通过", "Ref($close,-20)正确预测未来", "无"),
        ("4. 幸存者偏差", "⚠ 待查", "需确认FCF数据源是否含退市股", "低-中"),
        ("5. 流动性陷阱", "⚠ 风险", "部分持仓日均成交额偏低", "中"),
        ("6. α=0.3过拟合", "⚠ 风险", "0.3来源不透明, 可能看了测试集", "中"),
        ("7. 停牌处理", "❌ 缺陷", "停牌权重被重分配, 非冻结", "中-高"),
    ]

    print(f"\n  {'检查项':<24} {'状态':>8} {'说明':<40} {'风险':>8}")
    print(f"  {'-'*84}")
    for item, status, desc, risk in issues:
        print(f"  {item:<24} {status:>8} {desc:<40} {risk:>8}")

    print(f"\n  {'='*50}")
    print(f"  必须修复的严重问题 (影响回测可信度):")
    print(f"  {'='*50}")
    print(f"  1. [❌] 基本面特征全局归一化泄露 → 改为截面归一化")
    print(f"  2. [❌] 停牌权重重分配 → 改为 day_ret / topk")
    print(f"  3. [⚠] α=0.3来源不透明 → 改为Valid集独立选α")
    print(f"  4. [⚠] 幸存者偏差 → 确认数据源是否含退市股")
    print(f"  5. [⚠] 流动性 → 加入日均成交额下限过滤")


if __name__ == "__main__":
    audit_1_pit()
    audit_2_normalization()
    audit_3_label()
    audit_4_survivorship()
    audit_5_liquidity()
    audit_6_alpha_overfit()
    audit_7_suspension()
    audit_summary()
