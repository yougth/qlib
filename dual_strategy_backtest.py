"""
双策略回测 — 趋势跟随 & 历史新高
=====================================
股票池: 10年净利润为正的270+只股票
策略1: 趋势跟随 (MA50/100突破 + 3xATR追踪止损 + 风险均衡仓位)
策略2: 历史新高 (ATH买入 + 10xATR(40d)追踪止损 + 等风险仓位 + 无再平衡)

回测周期: 2021-01-01 ~ 2026-07-21 (连续回测, 按年统计)
"""
import os, sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
import warnings
warnings.filterwarnings("ignore")


def format_qlib_code(code):
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"


def load_prices(universe, start, end):
    print(f"  加载价格数据: {len(universe)} 只股票, {start}~{end}")
    df = D.features(list(universe), ["$open", "$high", "$low", "$close", "$volume"],
                     start_time=start, end_time=end)
    if df is None or len(df) == 0:
        return None
    df = df.reset_index()
    df.columns = ["instrument", "datetime", "open", "high", "low", "close", "volume"]
    return df


def compute_indicators(df):
    results = []
    for inst, g in df.groupby("instrument"):
        g = g.sort_values("datetime").copy()
        c = g["close"]; h = g["high"]; l = g["low"]
        g["ma50"] = c.rolling(50, min_periods=50).mean()
        g["ma100"] = c.rolling(100, min_periods=100).mean()
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        g["atr14"] = tr.rolling(14, min_periods=14).mean()
        g["atr40"] = tr.rolling(40, min_periods=40).mean()
        g["ath"] = c.expanding(min_periods=1).max()
        g["is_new_high"] = c >= g["ath"]
        g["trend_up"] = g["ma50"] > g["ma100"]
        g["above_ma50"] = c > g["ma50"]
        g["prev_above_ma50"] = g["above_ma50"].shift(1, fill_value=False)
        g["breakout"] = g["above_ma50"] & ~g["prev_above_ma50"]
        results.append(g)
    return pd.concat(results, ignore_index=True)


class BacktestEngine:
    def __init__(self, initial_capital=1e8, max_positions=10, cost_buy=0.0015, cost_sell=0.0025, min_cost=5):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.cost_buy = cost_buy
        self.cost_sell = cost_sell
        self.min_cost = min_cost

    def _calc_equity(self, cash, positions, day_data):
        eq = cash
        for inst, pos in positions.items():
            if inst in day_data.index:
                p = day_data.loc[inst, "close"]
                if pd.isna(p) or p <= 0:
                    p = pos["entry"]
                eq += pos["shares"] * p
            else:
                eq += pos["shares"] * pos["entry"]
        return max(eq, 0)

    def _try_buy(self, inst, price, atr, stop_multiplier, cash, total_equity, positions):
        """尝试买入, 返回 (shares, cost) 或 None"""
        if price <= 0 or atr <= 0 or np.isnan(atr) or np.isnan(price) or np.isnan(total_equity) or total_equity <= 0:
            return None
        stop_distance = stop_multiplier * atr
        if stop_distance <= 0 or np.isnan(stop_distance):
            return None
        # 风险预算: 每仓位最大亏损 = 总资产 / max_positions
        risk_budget = total_equity / self.max_positions
        # 按风险计算手数
        shares_by_risk = int(risk_budget / stop_distance / 100) * 100
        # 按资金计算手数 (最多用 1/max_positions 的资金)
        cash_budget = total_equity / self.max_positions
        shares_by_cash = int(cash_budget / price / 100) * 100
        # 取较小值
        shares = min(shares_by_risk, shares_by_cash)
        if shares < 100:
            shares = 100
        cost_amount = shares * price
        trade_cost = max(cost_amount * self.cost_buy, self.min_cost)
        if cash < cost_amount + trade_cost:
            shares = int((cash - trade_cost) / price / 100) * 100
            if shares < 100:
                return None
            cost_amount = shares * price
            trade_cost = max(cost_amount * self.cost_buy, self.min_cost)
            if cash < cost_amount + trade_cost:
                return None
        return shares, cost_amount + trade_cost

    def run_trend_following(self, data, start_date, end_date):
        """策略1: 趋势跟随"""
        dates = sorted(data["datetime"].unique())
        dates = [d for d in dates if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]

        cash = self.initial_capital
        positions = {}
        daily_records = []
        n_buys = n_sells = 0

        for dt in dates:
            day_data = data[data["datetime"] == dt].set_index("instrument")

            # 1. 止损检查 (用最高价追踪, 不是收盘价)
            to_sell = []
            for inst, pos in positions.items():
                if inst not in day_data.index:
                    continue
                row = day_data.loc[inst]
                # 更新追踪止损: 用当日最高价 - 3*ATR
                if not pd.isna(row["atr14"]) and row["atr14"] > 0:
                    new_stop = row["high"] - 3 * row["atr14"]
                    pos["stop"] = max(pos["stop"], new_stop)
                if row["low"] <= pos["stop"]:
                    to_sell.append(("stop", inst))
                elif not pd.isna(row["ma50"]) and not pd.isna(row["ma100"]) and row["ma50"] < row["ma100"]:
                    to_sell.append(("trend_reverse", inst))

            for reason, inst in to_sell:
                if inst not in positions:
                    continue
                row = day_data.loc[inst]
                pos = positions[inst]
                sell_price = min(pos["stop"], row["high"])
                if sell_price <= 0:
                    sell_price = row["close"]
                shares = pos["shares"]
                proceeds = shares * sell_price
                cost = max(proceeds * self.cost_sell, self.min_cost)
                cash += proceeds - cost
                del positions[inst]
                n_sells += 1

            # 2. 买入信号
            if len(positions) < self.max_positions:
                total_equity = self._calc_equity(cash, positions, day_data)
                slots = self.max_positions - len(positions)
                buy_candidates = []
                for inst in day_data.index:
                    if inst in positions:
                        continue
                    row = day_data.loc[inst]
                    if pd.isna(row["ma50"]) or pd.isna(row["ma100"]) or pd.isna(row["atr14"]):
                        continue
                    if row["atr14"] <= 0:
                        continue
                    # 条件: 趋势向上(MA50>MA100) + 当日突破MA50
                    if row["trend_up"] and row.get("breakout", False):
                        buy_candidates.append((inst, row["close"], row["atr14"], True))
                # 按ATR/price排序 (低波动优先)
                for inst, price, atr, is_breakout in buy_candidates[:slots]:
                    result = self._try_buy(inst, price, atr, 3, cash, total_equity, positions)
                    if result is None:
                        continue
                    shares, total_cost = result
                    cash -= total_cost
                    positions[inst] = {"shares": shares, "entry": price, "stop": price - 3 * atr, "atr": atr}
                    n_buys += 1

            # 3. 记录
            equity = self._calc_equity(cash, positions, day_data)
            if daily_records:
                prev_eq = daily_records[-1]["equity"]
                daily_ret = (equity - prev_eq) / prev_eq if prev_eq > 0 else 0
            else:
                daily_ret = (equity - self.initial_capital) / self.initial_capital
            daily_records.append({"datetime": dt, "return": daily_ret, "equity": equity})

        print(f"    交易: 买入{n_buys}次, 卖出{n_sells}次")
        return pd.DataFrame(daily_records)

    def run_ath_strategy(self, data, start_date, end_date):
        """策略2: 历史新高"""
        dates = sorted(data["datetime"].unique())
        dates = [d for d in dates if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]

        cash = self.initial_capital
        positions = {}
        daily_records = []
        n_buys = n_sells = 0

        for dt in dates:
            day_data = data[data["datetime"] == dt].set_index("instrument")

            # 1. 止损检查
            to_sell = []
            for inst, pos in positions.items():
                if inst not in day_data.index:
                    continue
                row = day_data.loc[inst]
                if not pd.isna(row["atr40"]) and row["atr40"] > 0:
                    new_stop = row["high"] - 10 * row["atr40"]
                    pos["stop"] = max(pos["stop"], new_stop)
                if row["low"] <= pos["stop"]:
                    to_sell.append(inst)

            for inst in to_sell:
                row = day_data.loc[inst]
                pos = positions[inst]
                sell_price = min(pos["stop"], row["high"])
                shares = pos["shares"]
                proceeds = shares * sell_price
                cost = max(proceeds * self.cost_sell, self.min_cost)
                cash += proceeds - cost
                del positions[inst]
                n_sells += 1

            # 2. 买入: 创历史新高
            if len(positions) < self.max_positions:
                total_equity = self._calc_equity(cash, positions, day_data)
                slots = self.max_positions - len(positions)
                buy_candidates = []
                for inst in day_data.index:
                    if inst in positions:
                        continue
                    row = day_data.loc[inst]
                    if pd.isna(row["atr40"]) or row["atr40"] <= 0:
                        continue
                    if row.get("is_new_high", False):
                        buy_candidates.append((inst, row["close"], row["atr40"]))
                buy_candidates.sort(key=lambda x: x[2] / x[1])
                for inst, price, atr in buy_candidates[:slots]:
                    result = self._try_buy(inst, price, atr, 10, cash, total_equity, positions)
                    if result is None:
                        continue
                    shares, total_cost = result
                    cash -= total_cost
                    positions[inst] = {"shares": shares, "entry": price, "stop": price - 10 * atr, "atr": atr}
                    n_buys += 1

            # 3. 记录
            equity = self._calc_equity(cash, positions, day_data)
            if daily_records:
                prev_eq = daily_records[-1]["equity"]
                daily_ret = (equity - prev_eq) / prev_eq if prev_eq > 0 else 0
            else:
                daily_ret = (equity - self.initial_capital) / self.initial_capital
            daily_records.append({"datetime": dt, "return": daily_ret, "equity": equity})

        print(f"    交易: 买入{n_buys}次, 卖出{n_sells}次")
        return pd.DataFrame(daily_records)

    def analyze(self, returns_df):
        if returns_df is None or len(returns_df) == 0:
            return {"ar": 0, "ir": 0, "mdd": 0, "total_return": 0}
        rets = returns_df.set_index("datetime")["return"]
        total_return = (1 + rets).prod() - 1
        n_days = len(rets)
        ar = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 and total_return > -1 else 0
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        cum = (1 + rets).cumprod()
        running_max = cum.expanding().max()
        dd = (cum - running_max) / running_max
        mdd = dd.min()
        return {
            "ar": ar * 100, "ir": sharpe, "sharpe": sharpe,
            "mdd": mdd * 100, "total_return": total_return * 100,
            "n_days": n_days, "final_equity": returns_df["equity"].iloc[-1]
        }

    def analyze_by_year(self, returns_df):
        """按年统计"""
        if returns_df is None or len(returns_df) == 0:
            return []
        df = returns_df.copy()
        df["year"] = df["datetime"].dt.year
        results = []
        for year, g in df.groupby("year"):
            rets = g.set_index("datetime")["return"]
            total_ret = (1 + rets).prod() - 1
            n_days = len(rets)
            ar = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 and total_ret > -1 else 0
            sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
            cum = (1 + rets).cumprod()
            dd = (cum - cum.expanding().max()) / cum.expanding().max()
            results.append({
                "year": str(year) if year < 2026 else "2026H1",
                "ar": ar * 100, "ir": sharpe, "mdd": dd.min() * 100,
                "total_return": total_ret * 100, "n_days": n_days
            })
        return results


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv")
    universe = df["code"].apply(format_qlib_code).tolist()
    print(f"股票池: {len(universe)} 只")

    data_start = "2020-06-01"
    data_end = "2026-07-21"
    prices = load_prices(universe, data_start, data_end)
    if prices is None:
        print("加载数据失败!")
        return

    print("  计算技术指标...")
    data = compute_indicators(prices)
    print(f"  数据: {len(data)} 行, {data['instrument'].nunique()} 只股票")

    bt_start = "2021-01-01"
    bt_end = "2026-07-21"
    engine = BacktestEngine(initial_capital=1e8, max_positions=10)

    # ====== 策略1: 趋势跟随 ======
    print(f"\n{'='*60}")
    print(f"  策略1: 趋势跟随 (MA50/100 + 3xATR止损)")
    print(f"{'='*60}")
    tf_rets = engine.run_trend_following(data, bt_start, bt_end)
    tf_full = engine.analyze(tf_rets)
    tf_yearly = engine.analyze_by_year(tf_rets)
    print(f"\n  全周期: 年化 {tf_full['ar']:.2f}%, IR {tf_full['ir']:.4f}, "
          f"回撤 {tf_full['mdd']:.2f}%, 终值 {tf_full['final_equity']/1e8:.4f}亿")
    for y in tf_yearly:
        print(f"    {y['year']}: 年化 {y['ar']:.2f}%, IR {y['ir']:.4f}, 回撤 {y['mdd']:.2f}%")

    # ====== 策略2: 历史新高 ======
    print(f"\n{'='*60}")
    print(f"  策略2: 历史新高 (ATH + 10xATR(40d)止损)")
    print(f"{'='*60}")
    ath_rets = engine.run_ath_strategy(data, bt_start, bt_end)
    ath_full = engine.analyze(ath_rets)
    ath_yearly = engine.analyze_by_year(ath_rets)
    print(f"\n  全周期: 年化 {ath_full['ar']:.2f}%, IR {ath_full['ir']:.4f}, "
          f"回撤 {ath_full['mdd']:.2f}%, 终值 {ath_full['final_equity']/1e8:.4f}亿")
    for y in ath_yearly:
        print(f"    {y['year']}: 年化 {y['ar']:.2f}%, IR {y['ir']:.4f}, 回撤 {y['mdd']:.2f}%")

    # ====== 对比汇总 ======
    print(f"\n\n{'='*85}")
    print(f"  双策略回测结果汇总 (2021-01 ~ 2026-07)")
    print(f"{'='*85}")
    print(f"\n  {'策略':<20} {'年化':>8} {'IR':>8} {'最大回撤':>8} {'总收益':>10} {'终值(亿)':>10}")
    print(f"  {'-'*64}")
    for name, m in [("趋势跟随", tf_full), ("历史新高", ath_full)]:
        print(f"  {name:<20} {m['ar']:>7.2f}% {m['ir']:>8.4f} {m['mdd']:>7.2f}% "
              f"{m['total_return']:>9.2f}% {m['final_equity']/1e8:>9.4f}")

    print(f"\n  逐年对比:")
    print(f"  {'年份':<8} {'趋势跟随年化':>12} {'趋势IR':>8} {'趋势回撤':>10} "
          f"{'新高年化':>12} {'新高IR':>8} {'新高回撤':>10}")
    print(f"  {'-'*68}")
    for i in range(max(len(tf_yearly), len(ath_yearly))):
        tf = tf_yearly[i] if i < len(tf_yearly) else {"year":"?","ar":0,"ir":0,"mdd":0}
        ah = ath_yearly[i] if i < len(ath_yearly) else {"year":"?","ar":0,"ir":0,"mdd":0}
        print(f"  {tf['year']:<8} {tf['ar']:>11.2f}% {tf['ir']:>8.4f} {tf['mdd']:>9.2f}% "
              f"{ah['ar']:>11.2f}% {ah['ir']:>8.4f} {ah['mdd']:>9.2f}%")

    # 对比V5
    print(f"\n  对比 V5 多因子模型 (年化 18.13%, IR 0.86, 回撤 -15.60%):")
    print(f"    趋势跟随: {'优于' if tf_full['ar'] > 18.13 else '不及'} V5 "
          f"(年化 {tf_full['ar']:.2f}% vs 18.13%, 回撤 {tf_full['mdd']:.2f}% vs -15.60%)")
    print(f"    历史新高: {'优于' if ath_full['ar'] > 18.13 else '不及'} V5 "
          f"(年化 {ath_full['ar']:.2f}% vs 18.13%, 回撤 {ath_full['mdd']:.2f}% vs -15.60%)")

    pd.DataFrame(tf_yearly).to_csv("/Users/11164591/Documents/Qoder目录/qlib/trend_following_results.csv", index=False)
    pd.DataFrame(ath_yearly).to_csv("/Users/11164591/Documents/Qoder目录/qlib/ath_strategy_results.csv", index=False)
    print(f"\n  结果已保存")


if __name__ == "__main__":
    run()
