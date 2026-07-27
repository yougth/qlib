"""
动量轮动策略回测
==================
股票池: 10年净利润为正的270+只股票
交易频率: 每周三调仓 (周四执行)
核心规则:
  1. momentum_score = 年化对数回归斜率 × R²
  2. 仓位: shares = 账户总值 × 0.001 / ATR(20)
  3. 大盘过滤: 上证指数在200日均线上方才买入
  4. 卖出: 不在前20%排名 或 低于100日均线
  5. 双周三: 头寸再平衡 (按当前ATR调整仓位)
"""
import os, sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
from scipy import stats
import qlib
from qlib.constant import REG_CN
from qlib.data import D
import warnings
warnings.filterwarnings("ignore")


def format_qlib_code(code):
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"


def momentum_score(ts):
    """
    input: 价格序列, 按交易日期降序 (最新在前)
    output: 收益率指数回归的斜率 × R²
    """
    ts = np.array(ts, dtype=float)
    if len(ts) < 20 or np.any(ts <= 0):
        return 0.0
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
    annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
    return annualized_slope * (r_value ** 2)


def get_atr(high, low, close, period=20):
    """计算ATR(20)"""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


class MomentumBacktest:
    def __init__(self, initial_capital=1e8, cost_buy=0.0015, cost_sell=0.0025, min_cost=5):
        self.initial_capital = initial_capital
        self.cost_buy = cost_buy
        self.cost_sell = cost_sell
        self.min_cost = min_cost
        self.risk_factor = 0.001  # 账户总值 × 0.001 / ATR

    def run(self, price_data, index_data, start_date, end_date,
            lookback=90, ma100_period=100, ma200_period=200, top_pct=0.2):
        """
        price_data: DataFrame[instrument, datetime, open, high, low, close, volume]
        index_data: Series[datetime -> close] 上证指数收盘
        """
        # 准备每只股票的技术指标
        print("  计算技术指标...")
        all_data = []
        for inst, g in price_data.groupby("instrument"):
            g = g.sort_values("datetime").copy()
            c = g["close"]
            g["ma100"] = c.rolling(ma100_period, min_periods=ma100_period).mean()
            g["atr20"] = get_atr(g["high"], g["low"], c, 20)
            # 动量得分: 用过去90天收盘价 (降序传入momentum_score)
            g["mom_score"] = c.rolling(lookback, min_periods=lookback).apply(
                lambda x: momentum_score(x[::-1]), raw=True
            )
            all_data.append(g)
        data = pd.concat(all_data, ignore_index=True)

        # 大盘趋势: 上证200日均线
        idx = index_data.sort_index()
        idx_ma200 = idx.rolling(ma200_period, min_periods=ma200_period).mean()
        idx_above_ma = (idx > idx_ma200).astype(int)

        # 交易日历
        dates = sorted(data["datetime"].unique())
        dates = [d for d in dates if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]
        print(f"  回测区间: {dates[0]} ~ {dates[-1]}, 共 {len(dates)} 个交易日")

        # 找周三 ( weekday()==2 )
        wednesdays = [d for d in dates if d.weekday() == 2]
        # 双周三: 每隔一周的周三
        bi_weekly = set()
        prev_wed = None
        for w in wednesdays:
            if prev_wed is None or (w - prev_wed).days >= 14:
                bi_weekly.add(w)
                prev_wed = w
        print(f"  周三数量: {len(wednesdays)}, 双周三: {len(bi_weekly)}")

        # 状态
        cash = self.initial_capital
        positions = {}  # {inst: {"shares": n, "entry": price}}
        daily_records = []
        n_buys = n_sells = n_rebal = 0
        last_rank = None

        for dt in dates:
            day_data = data[data["datetime"] == dt].set_index("instrument")

            # 计算当日总资产
            equity = cash
            for inst, pos in positions.items():
                if inst in day_data.index:
                    p = day_data.loc[inst, "close"]
                    if pd.isna(p) or p <= 0:
                        p = pos["entry"]
                    equity += pos["shares"] * p
                else:
                    equity += pos["shares"] * pos["entry"]
            equity = max(equity, 0)

            # 是否周三
            is_wed = dt in wednesdays
            is_bi_wed = dt in bi_weekly

            if is_wed:
                # 1. 更新排名
                valid = day_data[day_data["mom_score"].notna() & (day_data["mom_score"] > 0)]
                if len(valid) > 0:
                    valid = valid.sort_values("mom_score", ascending=False)
                    top_n = max(int(len(valid) * top_pct), 1)
                    top_stocks = set(valid.index[:top_n])
                    last_rank = valid[["mom_score", "close", "atr20"]].copy()
                else:
                    top_stocks = set()

                # 2. 检查卖出条件
                to_sell = []
                for inst in list(positions.keys()):
                    if inst not in day_data.index:
                        continue
                    row = day_data.loc[inst]
                    # 条件1: 不在前20%排名
                    if inst not in top_stocks:
                        to_sell.append(inst)
                    # 条件2: 低于100日均线
                    elif not pd.isna(row["ma100"]) and row["close"] < row["ma100"]:
                        to_sell.append(inst)

                for inst in to_sell:
                    row = day_data.loc[inst]
                    sell_price = row["close"]
                    if pd.isna(sell_price) or sell_price <= 0:
                        continue
                    pos = positions[inst]
                    proceeds = pos["shares"] * sell_price
                    cost = max(proceeds * self.cost_sell, self.min_cost)
                    cash += proceeds - cost
                    del positions[inst]
                    n_sells += 1

                # 3. 双周三: 头寸再平衡
                if is_bi_wed:
                    for inst in list(positions.keys()):
                        if inst not in day_data.index:
                            continue
                        row = day_data.loc[inst]
                        if pd.isna(row["atr20"]) or row["atr20"] <= 0 or pd.isna(row["close"]):
                            continue
                        # 期望手数 = equity * 0.001 / ATR
                        target_shares = int(equity * self.risk_factor / row["atr20"] / 100) * 100
                        if target_shares < 100:
                            target_shares = 100
                        current_shares = positions[inst]["shares"]
                        diff = target_shares - current_shares
                        if abs(diff) < 100:
                            continue
                        price = row["close"]
                        if diff > 0:
                            # 买入差额
                            cost_amount = diff * price
                            trade_cost = max(cost_amount * self.cost_buy, self.min_cost)
                            if cash >= cost_amount + trade_cost:
                                cash -= cost_amount + trade_cost
                                positions[inst]["shares"] = target_shares
                                n_rebal += 1
                        elif diff < 0:
                            # 卖出差额
                            sell_shares = -diff
                            proceeds = sell_shares * price
                            trade_cost = max(proceeds * self.cost_sell, self.min_cost)
                            cash += proceeds - trade_cost
                            positions[inst]["shares"] = target_shares
                            n_rebal += 1

                # 4. 买入 (有现金 + 大盘趋势向上)
                idx_trend_up = idx_above_ma.get(dt, 0) == 1 if dt in idx_above_ma.index else False

                if cash > 0 and idx_trend_up and last_rank is not None and len(top_stocks) > 0:
                    # 按动量排名顺序买入
                    for inst in last_rank.index:
                        if inst in positions:
                            continue
                        if inst not in top_stocks:
                            continue
                        if inst not in day_data.index:
                            continue
                        row = day_data.loc[inst]
                        if pd.isna(row["atr20"]) or row["atr20"] <= 0 or pd.isna(row["close"]):
                            continue
                        price = row["close"]
                        target_shares = int(equity * self.risk_factor / row["atr20"] / 100) * 100
                        if target_shares < 100:
                            target_shares = 100
                        cost_amount = target_shares * price
                        trade_cost = max(cost_amount * self.cost_buy, self.min_cost)
                        if cash < cost_amount + trade_cost:
                            # 买不起完整仓位, 尽量买
                            target_shares = int((cash - trade_cost) / price / 100) * 100
                            if target_shares < 100:
                                continue
                            cost_amount = target_shares * price
                            trade_cost = max(cost_amount * self.cost_buy, self.min_cost)
                            if cash < cost_amount + trade_cost:
                                continue
                        cash -= cost_amount + trade_cost
                        positions[inst] = {"shares": target_shares, "entry": price}
                        n_buys += 1

            # 记录日收益率
            if daily_records:
                prev_eq = daily_records[-1]["equity"]
                daily_ret = (equity - prev_eq) / prev_eq if prev_eq > 0 else 0
            else:
                daily_ret = (equity - self.initial_capital) / self.initial_capital
            daily_records.append({"datetime": dt, "return": daily_ret, "equity": equity})

        print(f"  交易统计: 买入{n_buys}次, 卖出{n_sells}次, 再平衡{n_rebal}次")
        return pd.DataFrame(daily_records)

    def analyze(self, returns_df):
        if returns_df is None or len(returns_df) == 0:
            return {"ar": 0, "ir": 0, "mdd": 0, "total_return": 0}
        rets = returns_df.set_index("datetime")["return"]
        total_return = (1 + rets).prod() - 1
        n_days = len(rets)
        if total_return <= -1:
            ar = -100.0
        else:
            ar = ((1 + total_return) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        cum = (1 + rets).cumprod()
        running_max = cum.expanding().max()
        dd = (cum - running_max) / running_max
        mdd = dd.min() * 100
        return {
            "ar": ar, "ir": sharpe, "sharpe": sharpe,
            "mdd": mdd, "total_return": total_return * 100,
            "n_days": n_days, "final_equity": returns_df["equity"].iloc[-1]
        }

    def analyze_by_year(self, returns_df):
        if returns_df is None or len(returns_df) == 0:
            return []
        df = returns_df.copy()
        df["year"] = df["datetime"].dt.year
        results = []
        for year, g in df.groupby("year"):
            rets = g.set_index("datetime")["return"]
            total_ret = (1 + rets).prod() - 1
            n_days = len(rets)
            if total_ret <= -1:
                ar = -100.0
            else:
                ar = ((1 + total_ret) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0
            sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
            cum = (1 + rets).cumprod()
            dd = (cum - cum.expanding().max()) / cum.expanding().max()
            label = str(year) if year < 2026 else "2026H1"
            results.append({"year": label, "ar": ar, "ir": sharpe, "mdd": dd.min() * 100})
        return results


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv")
    universe = df["code"].apply(format_qlib_code).tolist()
    print(f"股票池: {len(universe)} 只")

    # 加载数据 (提前足够天数用于计算MA200和动量)
    data_start = "2019-06-01"
    data_end = "2026-07-21"
    print(f"  加载价格数据: {data_start}~{data_end}")
    prices = D.features(list(universe), ["$open", "$high", "$low", "$close", "$volume"],
                         start_time=data_start, end_time=data_end)
    if prices is None or len(prices) == 0:
        print("加载数据失败!")
        return
    prices = prices.reset_index()
    prices.columns = ["instrument", "datetime", "open", "high", "low", "close", "volume"]

    # 上证指数 (qlib无指数数据, 用股票池等权均价代替)
    print("  计算大盘趋势代理...")
    bench_raw = D.features(list(universe[:50]), ["$close"], start_time=data_start, end_time=data_end)
    bench_raw = bench_raw.reset_index()
    bench_raw.columns = ["instrument", "datetime", "close"]
    index_data = bench_raw.groupby("datetime")["close"].mean()

    bt = MomentumBacktest(initial_capital=1e8)

    print(f"\n{'='*60}")
    print(f"  动量轮动策略回测")
    print(f"  周三调仓 | 双周三再平衡 | 大盘200MA过滤 | 动量前20%")
    print(f"{'='*60}")

    rets = bt.run(prices, index_data, "2021-01-01", "2026-07-21")
    metrics = bt.analyze(rets)
    yearly = bt.analyze_by_year(rets)

    print(f"\n  全周期: 年化 {metrics['ar']:.2f}%, IR {metrics['ir']:.4f}, "
          f"回撤 {metrics['mdd']:.2f}%, 终值 {metrics['final_equity']/1e8:.4f}亿")
    print(f"\n  逐年:")
    print(f"  {'年份':<8} {'年化':>10} {'IR':>8} {'回撤':>10}")
    print(f"  {'-'*36}")
    for y in yearly:
        print(f"  {y['year']:<8} {y['ar']:>9.2f}% {y['ir']:>8.4f} {y['mdd']:>9.2f}%")

    # 对比
    print(f"\n{'='*60}")
    print(f"  策略对比")
    print(f"{'='*60}")
    print(f"  {'策略':<24} {'年化':>8} {'IR':>8} {'回撤':>8}")
    print(f"  {'-'*48}")
    print(f"  {'动量轮动(本周)':<24} {metrics['ar']:>7.2f}% {metrics['ir']:>8.4f} {metrics['mdd']:>7.2f}%")
    print(f"  {'V5 多因子模型':<24} {'18.13':>7}% {'0.86':>8} {'-15.60':>7}%")
    print(f"  {'趋势跟随(前策略)':<24} {'-8.69':>7}% {'-0.25':>8} {'-71.42':>7}%")
    print(f"  {'历史新高(前策略)':<24} {'-9.25':>7}% {'-0.28':>8} {'-53.49':>7}%")

    pd.DataFrame(yearly).to_csv("/Users/11164591/Documents/Qoder目录/qlib/momentum_results.csv", index=False)
    print(f"\n  结果已保存")


if __name__ == "__main__":
    run()
