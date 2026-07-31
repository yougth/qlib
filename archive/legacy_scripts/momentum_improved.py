"""
动量轮动策略改进实验
=====================
基线: 90天回看 + 周频 + 纯动量 (上次结果: -1.65%)
改进方向:
  E1: 120天回看 + 周频
  E2: 150天回看 + 周频
  E3: 120天回看 + 月频
  E4: 120天回看 + 周频 + 价值因子融合(α=0.3)
  E5: 120天回看 + 月频 + 价值因子融合(α=0.3)  ← 全部改进叠加
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
    ts = np.array(ts, dtype=float)
    if len(ts) < 20 or np.any(ts <= 0):
        return 0.0
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
    annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
    return annualized_slope * (r_value ** 2)


def get_atr(high, low, close, period=20):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def load_value_factors(universe, start, end):
    """加载价值因子 (复用V6逻辑)"""
    vf = pd.read_csv("/Users/11164591/Documents/Qoder目录/value_factors_cache.csv")
    vf["date"] = pd.to_datetime(vf["date"])
    ann = vf[vf["quarter"] == 12].sort_values(["code", "year"]).reset_index(drop=True)
    fin = {}
    for _, r in ann.iterrows():
        c = str(r["code"]).zfill(6); y = int(r["year"])
        if c not in fin: fin[c] = {}
        fin[c][y] = {"roe": r.get("roe", np.nan), "eps": r.get("eps", np.nan),
                     "bps": r.get("bps", np.nan), "div_payout": r.get("div_payout", np.nan)}

    price = D.features(list(universe), ["$close"], start_time=start, end_time=end)
    if price is None or len(price) == 0: return None
    price = price.reset_index()
    price.columns = ["instrument", "datetime", "close"]

    cal_set = set(price["datetime"].unique())
    rows = []
    for qc in universe:
        cs = qc[2:]
        if cs not in fin: continue
        f = fin[cs]
        sp = price[price["instrument"] == qc].sort_values("datetime").set_index("datetime")
        if len(sp) == 0: continue
        years = sorted(f.keys())
        for i, y in enumerate(years):
            eps, bps, roe, dp = f[y]["eps"], f[y]["bps"], f[y]["roe"], f[y]["div_payout"]
            if pd.isna(eps) or eps == 0 or pd.isna(bps) or bps == 0: continue
            af = pd.Timestamp(f"{y+1}-05-01")
            at = pd.Timestamp(f"{years[i+1]+1}-04-30") if i+1 < len(years) else pd.Timestamp(f"{y+2}-04-30")
            w = sp[(sp.index >= af) & (sp.index <= at)].copy()
            if len(w) == 0: continue
            w["pe"] = w["close"] / eps; w["pb"] = w["close"] / bps; w["roe_val"] = roe
            w["div_yield"] = np.nan
            if not pd.isna(dp) and eps != 0:
                w["div_yield"] = (dp / 100.0) * eps / w["close"]
            hpe, hpb = [], []
            for hy in range(y - 3, y):
                if hy in fin:
                    he, hb = fin[hy]["eps"], fin[hy]["bps"]
                    if pd.isna(he) or he == 0 or pd.isna(hb) or hb == 0: continue
                    hf, ht = pd.Timestamp(f"{hy+1}-05-01"), pd.Timestamp(f"{hy+2}-04-30")
                    hw = sp[(sp.index >= hf) & (sp.index <= ht)]
                    if len(hw) > 0:
                        hpe.extend((hw["close"] / he).tolist())
                        hpb.extend((hw["close"] / hb).tolist())
            if len(hpe) >= 50:
                hpe_a = np.array([x for x in hpe if 0 < x < 500])
                hpb_a = np.array([x for x in hpb if 0 < x < 50])
                if len(hpe_a) >= 30 and len(hpb_a) >= 30:
                    hs_p, hs_b = np.sort(hpe_a), np.sort(hpb_a)
                    w["pb_pct_3y"] = w["pb"].apply(lambda x: np.searchsorted(hs_b, x) / len(hpb_a) if 0 < x < 50 else np.nan)
                else:
                    w["pb_pct_3y"] = np.nan
            else:
                w["pb_pct_3y"] = np.nan
            for dt, r in w.iterrows():
                if dt not in cal_set: continue
                rows.append({"instrument": qc, "datetime": dt,
                             "roe_annual": r["roe_val"], "pb_pct_3y": r["pb_pct_3y"],
                             "div_yield_est": r["div_yield"]})
    if not rows: return None
    vfd = pd.DataFrame(rows).set_index(["datetime", "instrument"])
    vfd = vfd[~vfd.index.duplicated(keep="last")]
    for c in vfd.columns:
        g = vfd[c].groupby(level=0)
        med = g.transform("median")
        mad = g.transform(lambda x: (x - x.median()).abs().median()).replace(0, np.nan)
        vfd[c] = ((vfd[c] - med) / (1.4826 * mad)).clip(-3, 3).fillna(0)
    return vfd


def fuse_momentum_value(mom_scores, vf_data, alpha=0.3):
    """
    将动量得分与价值因子融合
    mom_scores: Series[instrument -> mom_score] for a given date
    vf_data: DataFrame indexed by (datetime, instrument) with value columns
    alpha: value weight (0=pure momentum, 1=pure value)
    """
    if vf_data is None or alpha <= 0:
        return mom_scores

    # 获取当日的价值因子
    dt = mom_scores.name if hasattr(mom_scores, 'name') else None
    # mom_scores is a Series with instrument as index
    insts = mom_scores.index

    # Try to get value factors for these instruments
    # vf_data is indexed by (datetime, instrument)
    # We need to find the most recent value factor data for each instrument
    vf_subset = None
    if dt is not None:
        try:
            vf_subset = vf_data.xs(dt, level=0)
        except KeyError:
            pass

    if vf_subset is None or len(vf_subset) == 0:
        return mom_scores

    # Align instruments
    vf_aligned = vf_subset.reindex(insts)

    # Calculate value score: ROE rank + PB rank (lower PB better) + div_yield rank
    def pct_rank(s):
        valid = s.dropna()
        if len(valid) == 0:
            return pd.Series(0.5, index=s.index)
        return s.rank(pct=True).fillna(0.5)

    roe_rk = pct_rank(vf_aligned["roe_annual"])
    pb_rk = pct_rank(-vf_aligned["pb_pct_3y"])  # lower PB percentile = cheaper = better
    div_rk = pct_rank(vf_aligned["div_yield_est"])
    value_score = roe_rk + pb_rk + div_rk

    # Normalize both
    def zscore(s):
        valid = s.dropna()
        if len(valid) < 2:
            return pd.Series(0.0, index=s.index)
        return (s - valid.mean()) / (valid.std() + 1e-8)

    mom_norm = zscore(mom_scores)
    val_norm = zscore(value_score)

    fused = (1 - alpha) * mom_norm + alpha * val_norm
    # Only apply fusion where value data exists
    has_vf = vf_aligned["roe_annual"].notna()
    result = mom_scores.copy()
    result[has_vf] = fused[has_vf]
    return result


class MomentumBacktestV2:
    def __init__(self, initial_capital=1e8, cost_buy=0.0015, cost_sell=0.0025, min_cost=5):
        self.initial_capital = initial_capital
        self.cost_buy = cost_buy
        self.cost_sell = cost_sell
        self.min_cost = min_cost
        self.risk_factor = 0.001

    def run(self, price_data, index_data, vf_data, start_date, end_date,
            lookback=90, ma100_period=100, ma200_period=200, top_pct=0.2,
            rebal_freq="weekly", value_alpha=0.0):
        """
        rebal_freq: "weekly" (每周三) or "monthly" (每月最后一个周三)
        value_alpha: 0=纯动量, 0.3=30%价值权重
        """
        print(f"  计算技术指标 (lookback={lookback})...")
        all_data = []
        for inst, g in price_data.groupby("instrument"):
            g = g.sort_values("datetime").copy()
            c = g["close"]
            g["ma100"] = c.rolling(ma100_period, min_periods=ma100_period).mean()
            g["atr20"] = get_atr(g["high"], g["low"], c, 20)
            g["mom_score"] = c.rolling(lookback, min_periods=lookback).apply(
                lambda x: momentum_score(x[::-1]), raw=True
            )
            all_data.append(g)
        data = pd.concat(all_data, ignore_index=True)

        idx = index_data.sort_index()
        idx_ma200 = idx.rolling(ma200_period, min_periods=ma200_period).mean()
        idx_above_ma = (idx > idx_ma200).astype(int)

        dates = sorted(data["datetime"].unique())
        dates = [d for d in dates if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]

        # 确定调仓日
        wednesdays = [d for d in dates if d.weekday() == 2]
        if rebal_freq == "monthly":
            # 每月最后一个周三
            rebal_days = set()
            monthly_weds = {}
            for w in wednesdays:
                m = w.to_period("M")
                monthly_weds[m] = w  # 后出现的覆盖前面的 = 月末周三
            rebal_days = set(monthly_weds.values())
            # 月频: 头寸再平衡也在调仓日 (不再有双周概念)
            bi_rebal_days = rebal_days
            freq_label = "月频"
        else:
            rebal_days = set(wednesdays)
            # 双周三
            bi_rebal_days = set()
            prev_wed = None
            for w in wednesdays:
                if prev_wed is None or (w - prev_wed).days >= 14:
                    bi_rebal_days.add(w)
                    prev_wed = w
            freq_label = "周频"

        print(f"  回测区间: {dates[0]} ~ {dates[-1]}, {len(dates)}个交易日")
        print(f"  调仓频率: {freq_label}, 调仓日{len(rebal_days)}个, 再平衡日{len(bi_rebal_days)}个")
        if value_alpha > 0:
            print(f"  价值因子融合: α={value_alpha}")

        cash = self.initial_capital
        positions = {}
        daily_records = []
        n_buys = n_sells = n_rebal = 0

        for dt in dates:
            day_data = data[data["datetime"] == dt].set_index("instrument")

            equity = cash
            for inst, pos in positions.items():
                if inst in day_data.index:
                    p = day_data.loc[inst, "close"]
                    if pd.isna(p) or p <= 0: p = pos["entry"]
                    equity += pos["shares"] * p
                else:
                    equity += pos["shares"] * pos["entry"]
            equity = max(equity, 0)

            is_rebal = dt in rebal_days
            is_bi = dt in bi_rebal_days

            if is_rebal:
                # 1. 更新排名
                valid = day_data[day_data["mom_score"].notna() & (day_data["mom_score"] > 0)]
                if len(valid) > 0:
                    mom_series = valid["mom_score"].copy()

                    # 价值因子融合
                    if value_alpha > 0 and vf_data is not None:
                        mom_series.name = dt
                        fused = fuse_momentum_value(mom_series, vf_data, alpha=value_alpha)
                        valid = valid.assign(fused_score=fused)
                        valid = valid.sort_values("fused_score", ascending=False)
                    else:
                        valid = valid.sort_values("mom_score", ascending=False)

                    top_n = max(int(len(valid) * top_pct), 1)
                    top_stocks = set(valid.index[:top_n])
                    last_rank = valid[["mom_score", "close", "atr20"]].copy()
                else:
                    top_stocks = set()
                    last_rank = None

                # 2. 检查卖出条件
                to_sell = []
                for inst in list(positions.keys()):
                    if inst not in day_data.index:
                        continue
                    row = day_data.loc[inst]
                    if inst not in top_stocks:
                        to_sell.append(inst)
                    elif not pd.isna(row["ma100"]) and row["close"] < row["ma100"]:
                        to_sell.append(inst)

                for inst in to_sell:
                    row = day_data.loc[inst]
                    sell_price = row["close"]
                    if pd.isna(sell_price) or sell_price <= 0: continue
                    pos = positions[inst]
                    proceeds = pos["shares"] * sell_price
                    cost = max(proceeds * self.cost_sell, self.min_cost)
                    cash += proceeds - cost
                    del positions[inst]
                    n_sells += 1

                # 3. 头寸再平衡
                if is_bi:
                    for inst in list(positions.keys()):
                        if inst not in day_data.index: continue
                        row = day_data.loc[inst]
                        if pd.isna(row["atr20"]) or row["atr20"] <= 0 or pd.isna(row["close"]): continue
                        target_shares = int(equity * self.risk_factor / row["atr20"] / 100) * 100
                        if target_shares < 100: target_shares = 100
                        current_shares = positions[inst]["shares"]
                        diff = target_shares - current_shares
                        if abs(diff) < 100: continue
                        price = row["close"]
                        if diff > 0:
                            cost_amount = diff * price
                            trade_cost = max(cost_amount * self.cost_buy, self.min_cost)
                            if cash >= cost_amount + trade_cost:
                                cash -= cost_amount + trade_cost
                                positions[inst]["shares"] = target_shares
                                n_rebal += 1
                        elif diff < 0:
                            sell_shares = -diff
                            proceeds = sell_shares * price
                            trade_cost = max(proceeds * self.cost_sell, self.min_cost)
                            cash += proceeds - trade_cost
                            positions[inst]["shares"] = target_shares
                            n_rebal += 1

                # 4. 买入
                idx_trend_up = idx_above_ma.get(dt, 0) == 1 if dt in idx_above_ma.index else False
                if cash > 0 and idx_trend_up and last_rank is not None and len(top_stocks) > 0:
                    for inst in last_rank.index:
                        if inst in positions: continue
                        if inst not in top_stocks: continue
                        if inst not in day_data.index: continue
                        row = day_data.loc[inst]
                        if pd.isna(row["atr20"]) or row["atr20"] <= 0 or pd.isna(row["close"]): continue
                        price = row["close"]
                        target_shares = int(equity * self.risk_factor / row["atr20"] / 100) * 100
                        if target_shares < 100: target_shares = 100
                        cost_amount = target_shares * price
                        trade_cost = max(cost_amount * self.cost_buy, self.min_cost)
                        if cash < cost_amount + trade_cost:
                            target_shares = int((cash - trade_cost) / price / 100) * 100
                            if target_shares < 100: continue
                            cost_amount = target_shares * price
                            trade_cost = max(cost_amount * self.cost_buy, self.min_cost)
                            if cash < cost_amount + trade_cost: continue
                        cash -= cost_amount + trade_cost
                        positions[inst] = {"shares": target_shares, "entry": price}
                        n_buys += 1

            if daily_records:
                prev_eq = daily_records[-1]["equity"]
                daily_ret = (equity - prev_eq) / prev_eq if prev_eq > 0 else 0
            else:
                daily_ret = (equity - self.initial_capital) / self.initial_capital
            daily_records.append({"datetime": dt, "return": daily_ret, "equity": equity})

        print(f"  交易统计: 买入{n_buys}, 卖出{n_sells}, 再平衡{n_rebal}")
        return pd.DataFrame(daily_records)

    def analyze(self, returns_df):
        if returns_df is None or len(returns_df) == 0:
            return {"ar": 0, "ir": 0, "mdd": 0, "total_return": 0}
        rets = returns_df.set_index("datetime")["return"]
        total_return = (1 + rets).prod() - 1
        n_days = len(rets)
        ar = ((1 + total_return) ** (252 / n_days) - 1) * 100 if total_return > -1 and n_days > 0 else -100.0
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        cum = (1 + rets).cumprod()
        dd = (cum - cum.expanding().max()) / cum.expanding().max()
        return {"ar": ar, "ir": sharpe, "mdd": dd.min() * 100,
                "total_return": total_return * 100,
                "final_equity": returns_df["equity"].iloc[-1]}

    def analyze_by_year(self, returns_df):
        if returns_df is None or len(returns_df) == 0: return []
        df = returns_df.copy()
        df["year"] = df["datetime"].dt.year
        results = []
        for year, g in df.groupby("year"):
            rets = g.set_index("datetime")["return"]
            total_ret = (1 + rets).prod() - 1
            n_days = len(rets)
            ar = ((1 + total_ret) ** (252 / n_days) - 1) * 100 if total_ret > -1 and n_days > 0 else -100.0
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

    data_start = "2019-06-01"
    data_end = "2026-07-21"
    print("  加载价格数据...")
    prices = D.features(list(universe), ["$open", "$high", "$low", "$close", "$volume"],
                         start_time=data_start, end_time=data_end)
    prices = prices.reset_index()
    prices.columns = ["instrument", "datetime", "open", "high", "low", "close", "volume"]

    print("  计算大盘趋势代理...")
    bench_raw = D.features(list(universe[:50]), ["$close"], start_time=data_start, end_time=data_end)
    bench_raw = bench_raw.reset_index()
    bench_raw.columns = ["instrument", "datetime", "close"]
    index_data = bench_raw.groupby("datetime")["close"].mean()

    print("  加载价值因子...")
    vf_data = load_value_factors(list(universe), data_start, data_end)
    if vf_data is not None:
        print(f"  价值因子: {len(vf_data)} 条记录")
    else:
        print("  价值因子加载失败, 融合实验将跳过")

    bt = MomentumBacktestV2()

    experiments = [
        {"name": "E1: 120天+周频",       "lookback": 120, "rebal_freq": "weekly",  "value_alpha": 0.0},
        {"name": "E2: 150天+周频",       "lookback": 150, "rebal_freq": "weekly",  "value_alpha": 0.0},
        {"name": "E3: 120天+月频",       "lookback": 120, "rebal_freq": "monthly", "value_alpha": 0.0},
        {"name": "E4: 120天+周频+价值",  "lookback": 120, "rebal_freq": "weekly",  "value_alpha": 0.3},
        {"name": "E5: 120天+月频+价值",  "lookback": 120, "rebal_freq": "monthly", "value_alpha": 0.3},
    ]

    all_results = []
    all_yearly = {}

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"  {exp['name']}")
        print(f"{'='*60}")
        rets = bt.run(prices, index_data, vf_data, "2021-01-01", "2026-07-21",
                      lookback=exp["lookback"], rebal_freq=exp["rebal_freq"],
                      value_alpha=exp["value_alpha"])
        metrics = bt.analyze(rets)
        yearly = bt.analyze_by_year(rets)
        all_yearly[exp["name"]] = yearly

        result = {"策略": exp["name"], "年化": metrics["ar"], "IR": metrics["ir"],
                  "回撤": metrics["mdd"], "终值(亿)": metrics["final_equity"] / 1e8}
        all_results.append(result)

        print(f"\n  全周期: 年化 {metrics['ar']:.2f}%, IR {metrics['ir']:.4f}, "
              f"回撤 {metrics['mdd']:.2f}%, 终值 {metrics['final_equity']/1e8:.4f}亿")
        print(f"  逐年: ", end="")
        for y in yearly:
            print(f"{y['year']} {y['ar']:+.1f}% ", end="")
        print()

    # 汇总
    print(f"\n{'='*70}")
    print(f"  汇总对比")
    print(f"{'='*70}")
    print(f"  {'策略':<24} {'年化':>8} {'IR':>8} {'回撤':>8} {'终值':>8}")
    print(f"  {'-'*56}")
    # 基线
    print(f"  {'基线: 90天+周频':<24} {'-1.65':>7}% {'-0.16':>8} {'-21.47':>7}% {'0.92':>7}亿")
    for r in all_results:
        print(f"  {r['策略']:<24} {r['年化']:>7.2f}% {r['IR']:>8.4f} {r['回撤']:>7.2f}% {r['终值(亿)']:>7.4f}亿")
    print(f"  {'V5 多因子模型':<24} {'18.13':>7}% {'0.86':>8} {'-15.60':>7}% {'~2.3':>7}亿")

    # 逐年对比表
    print(f"\n  逐年年化对比:")
    years_set = set()
    for yl in all_yearly.values():
        for y in yl:
            years_set.add(y["year"])
    years_sorted = sorted(years_set)
    header = f"  {'策略':<24}" + "".join(f" {y:>10}" for y in years_sorted)
    print(header)
    print(f"  {'-'*(24 + 11*len(years_sorted))}")
    print(f"  {'基线: 90天+周频':<24}" + "".join(f" {'--':>10}" for _ in years_sorted))
    for name, yl in all_yearly.items():
        row = f"  {name:<24}"
        yd = {y["year"]: y["ar"] for y in yl}
        for y in years_sorted:
            v = yd.get(y, 0)
            row += f" {v:>+9.1f}%"
        print(row)

    pd.DataFrame(all_results).to_csv("/Users/11164591/Documents/Qoder目录/qlib/momentum_improved_results.csv", index=False)
    print(f"\n  结果已保存")


if __name__ == "__main__":
    run()
