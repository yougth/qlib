#!/usr/bin/env python3
"""
live/monthly_signal.py —— 每月一条命令: 输出"买哪几只、各买多少股" (生产入口)
================================================================================
上线纪律: 模型年度冻结, 本脚本每月只做推理, 不训练 → 同一信号日重复运行结果完全一致。

流程 (每一步都复用 core 里回测用的同一函数, 保证实盘与回测同口径):
  1. 冻结模型 (load_frozen: SHA256 + config_hash 双校验)
  2. 信号日 = 最新可用交易日 (默认) 或 --date 指定; 非月末会显式告警
  3. PIT 股票池 (build_dynamic_universe, year-2 规则)
  4. 可交易候选 (build_candidates: 一字涨停/停牌/流动性 ≥2000万)
  5. DE 腿 = 冻结模型打分 Top5; V20 腿 = value_comp Top10  (排序名单留冗余供顺延)
  6. 仓位 (portfolio.allocate: sleeve 40/60 + 一手取整 + 买不起顺延)
  7. 与当前持仓比对 → 调仓指令 + 实际换手率; 全量存档到 live/signals/

用法:
    cd quant
    PYTHONPATH=. python3 live/monthly_signal.py --capital 50000
    PYTHONPATH=. python3 live/monthly_signal.py --capital 50000 --date 2026-07-31
    PYTHONPATH=. python3 live/monthly_signal.py --capital 50000 --held live/positions.json
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import json
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from qlib.data import D

warnings.filterwarnings("ignore")

from core import config
from core import data as datalayer
from core import pipeline
from core import portfolio
from core import strategy
from core.universe import build_dynamic_universe, format_qlib_code, load_pit_caches
from core.valuation import load_valuation, value_comp_score
from core.tradability import build_tradability

SIG_DIR = f"{config.QUANT_DIR}/live/signals"
NAME_CSV = f"{config.DATA_DIR}/fcf_result.csv"
CAND_MULT = 4          # 候选名单取 topk × 此倍数, 为"买不起顺延"留冗余


def load_names():
    """code → 中文名 (缺失不报错, 新股可能不在缓存里, 用代码占位)"""
    if not os.path.exists(NAME_CSV):
        return {}
    df = pd.read_csv(NAME_CSV, dtype={"code": str})
    return {format_qlib_code(c): n for c, n in zip(df["code"], df["name"])}


def resolve_signal_date(cal, want=None):
    """信号日: 指定则取 ≤ 该日的最后交易日; 否则取日历最后一个交易日。

    两类生产风险必须显式告警, 否则会拿着错误的信号下单:
      1. 数据陈旧: qlib 行情落后于今天 → 信号是"过去某天"的, 必须先更新数据
      2. 非月末: 策略口径是月末调仓, 中途执行会偏离回测
    """
    if want:
        ds = [d for d in cal if d <= pd.Timestamp(want)]
        if not ds:
            raise SystemExit(f"[CHECK] {want} 之前无交易日")
        sig = ds[-1]
    else:
        sig = cal[-1]

    today = pd.Timestamp.today().normalize()
    stale = (today - cal[-1]).days
    if stale > 7:
        print(f"[WARN] 行情数据止于 {cal[-1].date()}, 距今 {stale} 天 —— 数据陈旧!\n"
              f"       请先重跑 tools/fetch_ohlcv.py 更新行情并重建 qlib bin, "
              f"否则信号基于过期价格。", flush=True)
    # 真正的月末 = 该月最后一个交易日; 数据末尾那天不算(它只是"数据到这儿了")
    month_last_bd = sig + pd.offsets.BMonthEnd(0)
    later_in_month = [d for d in cal if d > sig and d.month == sig.month and d.year == sig.year]
    if later_in_month or sig < month_last_bd - pd.Timedelta(days=3):
        print(f"[WARN] 信号日 {sig.date()} 不是 {sig.year}-{sig.month:02d} 的最后交易日 "
              f"(该月末≈{month_last_bd.date()}) —— 策略口径是月末调仓, 非月末执行会偏离回测。\n"
              f"       确认这是补跑/试算再继续。", flush=True)
    return sig


def latest_prices(insts, sig_dt):
    """信号日**真实**收盘价 (下单金额基准; 实际按 T+1 收盘成交, 价格会有偏差)

    必须除以 $factor: qlib 里的 $close 是复权价, 与券商下单界面的价格不是一回事。
    实测原 cn_data 中 358 只里有 54 只两者相差 10% 以上 (最大 1.89 倍) —— 直接用
    $close 会把"一手多少钱"算错到 ±90%, 一手取整/买不起顺延的判断全部失效。
    """
    px = D.features(list(insts), ["$close", "$factor"],
                    start_time=sig_dt - pd.Timedelta(days=10), end_time=sig_dt)
    if px is None or len(px) == 0:
        raise RuntimeError("[CHECK] 信号日价格拉取失败!")
    real = px["$close"] / px["$factor"]
    s = real.unstack(level=0).sort_index().ffill().iloc[-1]
    out = {k: float(v) for k, v in s.items() if np.isfinite(v) and v > 0}
    if not out:
        raise RuntimeError("[CHECK] 复权因子还原后无有效价格 ($factor 缺失?)!")
    return out


def _fmt(orders, names, held_map=None):
    o = orders.copy()
    o["名称"] = o["instrument"].map(lambda x: names.get(x, x))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, required=True, help="总投入资金(元)")
    ap.add_argument("--model", default="DoubleEnsemble")
    ap.add_argument("--year", type=int, default=0, help="冻结模型年份 (默认按信号日年份)")
    ap.add_argument("--date", default="", help="信号日 YYYY-MM-DD (默认最新交易日)")
    ap.add_argument("--held", default="", help="当前持仓 json {instrument: shares}")
    ap.add_argument("--no-archive", action="store_true", help="只打印不存档")
    args = ap.parse_args()

    datalayer.init_qlib()
    cal = datalayer.get_calendar()
    sig = resolve_signal_date(cal, args.date or None)
    year = args.year or sig.year
    names = load_names()

    # ---- 股票池 (与回测同一函数, year-2 PIT 规则) ----
    fcf_df, profit_df = load_pit_caches()
    universe = [format_qlib_code(c) for c in build_dynamic_universe(year, fcf_df, profit_df)]
    print(f"\n{'='*74}\n[monthly_signal] 信号日 {sig.date()} | 池 {len(universe)} 只 | "
          f"资金 {args.capital:,.0f} 元\n{'='*74}", flush=True)

    # ---- 冻结模型推理 (不训练) ----
    models, manifest = pipeline.load_frozen(args.model, year)
    print(f"[model] {args.model} W{year} 冻结于 {manifest['frozen_at']} | "
          f"{manifest['n_seeds']} 种子 | config_hash={manifest['config_hash']} | "
          f"git={manifest['git_commit']}", flush=True)
    if sorted(universe) != manifest["universe"]:
        print(f"[WARN] 当前池({len(universe)}只) 与冻结时({manifest['universe_size']}只) 不一致 —— "
              f"PIT 缓存已更新。推理仍按当前池, 但年度模型建议重训。", flush=True)
    # 推理段: 从信号日往前留足 60 自然日, 保证 handler 能算出滚动特征
    test_seg = ((sig - pd.Timedelta(days=90)).strftime("%Y-%m-%d"), sig.strftime("%Y-%m-%d"))
    pred = pipeline.predict_frozen(models, manifest, universe, test_seg)
    if sig not in pred.index.get_level_values(0):
        raise RuntimeError(f"[CHECK] 模型未对信号日 {sig.date()} 出分 (行情或特征缺失)!")
    cross = pred.xs(sig, level=0)

    # ---- 可交易候选 (与回测同一函数) ----
    limit_up, susp, liq = build_tradability(universe, sig.strftime("%Y-%m-%d"),
                                           sig.strftime("%Y-%m-%d"))
    cand = strategy.eligible_candidates(cross.index, sig, limit_up, susp, liq,
                                        config.MIN_CAND, strict=True, tag="live")
    print(f"[cand] 可交易候选 {len(cand)} 只 (已过滤一字涨停/停牌/流动性<2000万)", flush=True)

    # ---- 两条腿打分 → 排序名单 (留冗余供顺延) ----
    val_piv = load_valuation()
    de_rank = cross.reindex(cand).dropna().sort_values(ascending=False)
    v_rank = value_comp_score(cand, sig, val_piv).dropna().sort_values(ascending=False)
    picks = {
        "DE": list(de_rank.index[:portfolio.SLEEVES["DE"]["topk"] * CAND_MULT]),
        "V20": list(v_rank.index[:portfolio.SLEEVES["V20"]["topk"] * CAND_MULT]),
    }

    # ---- 仓位分配 ----
    prices = latest_prices(set(picks["DE"]) | set(picks["V20"]), sig)
    orders, summ = portfolio.allocate(picks, prices, args.capital)
    merged = portfolio.merge_overlap(orders)

    # ---- 输出 ----
    print(f"\n【一、下单清单】信号日 {sig.date()} 收盘价计价, 次一交易日收盘执行")
    m = merged.copy()
    m["名称"] = m["instrument"].map(lambda x: names.get(x, x))
    m["占比"] = (m["amount"] / args.capital * 100).round(1).astype(str) + "%"
    print(m[["instrument", "名称", "sleeves", "price", "lots", "shares", "amount", "占比"]]
          .rename(columns={"instrument": "代码", "sleeves": "腿", "price": "价格",
                           "lots": "手数", "shares": "股数", "amount": "金额"})
          .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print(f"\n【二、资金使用】")
    print(f"  总资金 {summ['capital']:,.0f} | 实投 {summ['invested']:,.0f} "
          f"({summ['invested']/summ['capital']*100:.1f}%) | 现金余 {summ['cash_left']:,.0f} "
          f"({summ['cash_pct']*100:.1f}%)")
    for s, amt in summ["sleeve_invested"].items():
        print(f"  {s} 腿: 实投 {amt:,.0f} / 目标 {summ['sleeve_target'][s]:,.0f} "
              f"({amt/summ['sleeve_target'][s]*100:.1f}%)")
    print(f"  持仓 {summ['n_positions']} 只 (下单 {summ['n_orders']} 笔, 重合已合并) | "
          f"等权最大偏离 {summ['max_weight_dev']*100:.1f}%")
    if summ["skipped"]:
        print(f"  跳过 {len(summ['skipped'])} 项:")
        for sk in summ["skipped"][:8]:
            print(f"    - [{sk['sleeve']}] {names.get(sk['instrument'], sk['instrument'])}: {sk['reason']}")
    small = m[merged["amount"] < portfolio.MIN_TICKET] if len(merged) else merged
    if len(small):
        print(f"  [WARN] {len(small)} 笔金额 <{portfolio.MIN_TICKET} 元, 佣金占比偏高 "
              f"(务必用免5佣金券商)")

    # ---- 调仓指令 ----
    held = {}
    if args.held and os.path.exists(args.held):
        with open(args.held) as f:
            held = json.load(f)
    diff, tstat = portfolio.diff_positions(merged, held)
    if held:
        print(f"\n【三、调仓指令】(vs 当前持仓)")
        d = diff[diff["action"] != "持有"].copy()
        d["名称"] = d["instrument"].map(lambda x: names.get(x, x))
        print(d[["instrument", "名称", "action", "held", "target", "delta"]]
              .rename(columns={"instrument": "代码", "action": "操作", "held": "现持",
                               "target": "目标", "delta": "增减"}).to_string(index=False))
        print(f"  换手率 {tstat['turnover']*100:.1f}% | 成交额 {tstat['traded_amount']:,.0f} | "
              f"预估摩擦 {tstat['est_cost']:,.0f} 元")
    else:
        print(f"\n【三、调仓指令】未提供当前持仓 (--held), 视为首次建仓: "
              f"全部 {len(merged)} 只买入")

    # ---- 存档 (可复查/可对账) ----
    if not args.no_archive:
        tag = sig.strftime("%Y-%m-%d")
        d = f"{SIG_DIR}/{tag}"
        os.makedirs(d, exist_ok=True)
        merged.to_csv(f"{d}/orders.csv", index=False)
        orders.to_csv(f"{d}/orders_by_sleeve.csv", index=False)
        diff.to_csv(f"{d}/rebalance.csv", index=False)
        de_rank.to_csv(f"{d}/score_DE.csv", header=["score"])
        v_rank.to_csv(f"{d}/score_V20.csv", header=["score"])
        meta = {"signal_date": tag, "capital": args.capital, "model": args.model,
                "year": year, "manifest": manifest["files"],
                "config_hash": manifest["config_hash"], "git_commit": manifest["git_commit"],
                "sleeves": portfolio.SLEEVES,
                "summary": {k: v for k, v in summ.items() if k != "skipped"},
                "skipped": summ["skipped"], "turnover": tstat}
        with open(f"{d}/meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        # 下月对账用: 本次目标持仓即下次的 held
        with open(f"{d}/positions_after.json", "w") as f:
            json.dump(dict(zip(merged["instrument"], merged["shares"].astype(int))), f, indent=2)
        print(f"\n[archive] 已存档 → live/signals/{tag}/ "
              f"(orders.csv / rebalance.csv / meta.json / positions_after.json)")
        print(f"[next] 下月运行时加: --held live/signals/{tag}/positions_after.json")


if __name__ == "__main__":
    main()
