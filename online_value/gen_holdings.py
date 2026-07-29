"""
出持仓与买入清单 (生产脚本)
==================================================================
策略口径全部来自 core/ (自包含, 不依赖 qlib/ 根目录的实验脚本):
  股票池 : 最近十年净利润与FCF双正 (PIT增广缓存, 用 year-2 财务避前视)
  信号日 : 每月首个交易日的前一交易日, 用当日估值缓存 (ep/bp/cfp/sp)
  打分   : value_comp = 4个估值倒数的池内截面rank百分位均值 (越大越便宜)
  过滤   : 20日均成交额>=500万真实值; 涨停/停牌剔除
  持仓   : Top20 等权 (每只5%), 月初首个交易日尾盘成交
  回测锚 : golden/backtest_baseline.csv (run_backtest.py 复现)

用法:
  /usr/bin/python3 gen_holdings.py                # 历史各月持仓 + 最新买入清单
  /usr/bin/python3 gen_holdings.py --years 2025 2026
输出:
  holdings_YYYY.csv          # 该年各月 Top20 (与回测同一路径算出)
  latest_buy_YYYYMMDD.csv    # 最新买入清单 (Top20 + 5只备选, 含名称/收盘价)

执行说明:
  - 每月第一个交易日尾盘(14:45后)按清单调仓; 某只当日涨停/停牌则用备选顺延
  - 月度单边换手约15%: 每月实际只需买卖各2-4只
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from qlib.data import D

from core import config
from core.universe import load_fin_caches, get_universe, name_map
from core.valuation import load_raw_valuation, compute_value_comp
from core.calendar_rules import get_calendar, next_rebalance
from core.selection import liquidity_at, tradable_universe, pick_topk
from core.market import build_liquidity_table, build_limit_up_set
from core.signal import year_context, year_rebalances, holdings_frame

OUT_DIR = config.ONLINE_DIR


def hist_years(argv):
    """--years 2025 2026; 默认 去年 + 今年"""
    if "--years" in argv:
        i = argv.index("--years")
        ys = [int(a) for a in argv[i + 1:] if a.isdigit()]
        if ys:
            return ys
    y = pd.Timestamp.now().year
    return [y - 1, y]


def dump_history(years, cal_list, val_piv, fcf_df, profit_df, nmap):
    """历史各月持仓: 与回测完全同一条路径 (core.signal.year_rebalances)"""
    for y in years:
        if y not in config.YEARS:
            print(f"  [跳过] {y} 不在回测年份集合 {config.YEARS} 内", flush=True)
            continue
        ctx = year_context(y, cal_list, fcf_df, profit_df, need_close=False, verbose=False)
        df = holdings_frame(year_rebalances(ctx, val_piv), nmap)
        df.to_csv(f"{OUT_DIR}/holdings_{y}.csv", sep='\t', index=False)
        print(f"[+] holdings_{y}.csv  {df['exec_date'].nunique()}期 x Top{config.TOPK}", flush=True)


def prev_list():
    """最近一份 latest_buy_*.csv, 用于自动比对换入换出"""
    fs = sorted(glob.glob(f"{OUT_DIR}/latest_buy_*.csv"))
    if not fs:
        return None, None
    df = pd.read_csv(fs[-1], sep='\t')
    return os.path.basename(fs[-1]), df[df["role"] == "持仓"]["instrument"].tolist()


def dump_latest(cal_list, val_piv, fcf_df, profit_df, nmap):
    """最新买入清单: 信号日 = min(行情末日, 估值末日)"""
    y = cal_list[-1].year
    universe = get_universe(y, fcf_df, profit_df)
    last_px_day = cal_list[-1]
    sig = min(val_piv["ep"].index[-1], last_px_day)

    sig_r, exec_r = next_rebalance(cal_list)
    print(f"\n  信号日 = {sig.date()}  (行情至 {last_px_day.date()}, 估值至 "
          f"{val_piv['ep'].index[-1].date()})", flush=True)
    if sig_r is not None:
        flag = "✓ 正是月频换仓口径" if sig == sig_r else "!! 不是月频换仓日"
        print(f"  下一换仓点: 信号日 {sig_r.date()} → 执行日 {exec_r.date()}   [{flag}]", flush=True)
        if sig != sig_r:
            print(f"  [!] 本清单信号日({sig.date()})不等于换仓信号日({sig_r.date()}), "
                  f"仅供预演。正式下单请在 {sig_r.date()} 收盘后重跑。", flush=True)

    px = D.features(universe, ["$close"],
                    start_time=str((last_px_day - pd.Timedelta(days=45)).date()),
                    end_time=str(last_px_day.date()))
    px = px.reset_index(); px.columns = ["instrument", "datetime", "close"]
    close_piv = px.pivot(index="datetime", columns="instrument", values="close").sort_index().ffill()

    vc = compute_value_comp(universe, sig, val_piv)
    liq_table = build_liquidity_table(universe, sig - pd.Timedelta(days=10), str(last_px_day.date()))
    liq = liquidity_at(liq_table, sig, vc.index, fallback_prev=True)
    # 执行日在未来不可知, 用信号日的涨停/停牌状态代理; 实际下单时以备选顺延兜底
    lu, sus = build_limit_up_set(universe, cal_list)
    trd = tradable_universe(vc.index, liq, sig, lu, sus)
    n_drop = int(vc.notna().sum()) - len(trd.intersection(vc.dropna().index))
    top = pick_topk(vc, trd, config.TOPK + config.N_BACKUP)

    rows = []
    for rk, (inst, score) in enumerate(top.items(), 1):
        rows.append(dict(rank=rk, instrument=inst, name=nmap.get(inst[2:], ""),
                         value_comp=round(float(score), 4),
                         close=round(float(close_piv[inst].iloc[-1]), 2) if inst in close_piv else np.nan,
                         weight=round(1.0 / config.TOPK, 4) if rk <= config.TOPK else 0.0,
                         role="持仓" if rk <= config.TOPK else "备选(涨停/停牌顺延)"))
    out = pd.DataFrame(rows)
    pf, phold = prev_list()
    fn = f"{OUT_DIR}/latest_buy_{sig.date().strftime('%Y%m%d')}.csv"
    out.to_csv(fn, sep='\t', index=False)

    hold = out[out["role"] == "持仓"]
    print(f"[+] {os.path.basename(fn)}  Top{config.TOPK}+备选{config.N_BACKUP}  "
          f"(池{len(universe)}只, 流动性/涨停/停牌剔除{n_drop}只)", flush=True)
    fin = hold[hold["instrument"].isin(config.FIN_SET)]
    print(f"  金融股占位: {len(fin)}/{config.TOPK} 只 (回测期月均4.8只, 3~8只属正常)", flush=True)
    bshare = out[out["instrument"].str.contains("SZ2|SH9")]
    if len(bshare):
        print(f"  [!] 清单含B股 {bshare['instrument'].tolist()} — 需单独B股账户, 可用备选替换", flush=True)
    if phold:
        cur = set(hold["instrument"]); old = set(phold)
        print(f"  与上一份 {pf} 对比: 重叠{len(cur & old)}只  "
              f"新进{sorted(cur - old)}  移出{sorted(old - cur)}", flush=True)
    print(out.to_string(index=False), flush=True)


def main():
    config.init_qlib()
    cal_list = get_calendar()          # 生产: 用数据自然末端
    val_piv = load_raw_valuation()
    fcf_df, profit_df = load_fin_caches()
    nmap = name_map()
    dump_history(hist_years(sys.argv), cal_list, val_piv, fcf_df, profit_df, nmap)
    dump_latest(cal_list, val_piv, fcf_df, profit_df, nmap)


if __name__ == "__main__":
    main()
