#!/usr/bin/env python3
"""
live/monthly_signal.py —— 信号生成 (生产入口): 支持 ICW_SW / VG / VGH / DUAL_LEG 四策略
================================================================================
上线纪律: 模型年度冻结, 本脚本只做推理不训练 → 同一信号日重复运行结果完全一致。

支持策略:
  ICW_SW   : ICW双周+熊市切VGH (消融版)  — 牛市10只ICW, 熊市3只ICW+7只VGH
  VG       : VG Top10 (value+盈利)       — 纯规则, 10只等权
  VGH      : VGH Top10 (结构化剥离)      — 纯规则, 10只等权
  DUAL_LEG : DE模型+V20规则 (旧版双腿)   — 40%DE + 60%V20 (向后兼容)

用法:
    cd quant
    PYTHONPATH=. python3 live/monthly_signal.py --capital 50000 --strategy ICW_SW
    PYTHONPATH=. python3 live/monthly_signal.py --capital 50000 --strategy VG
    PYTHONPATH=. python3 live/monthly_signal.py --capital 50000 --strategy VGH
    PYTHONPATH=. python3 live/monthly_signal.py --capital 50000 --strategy DUAL_LEG
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
from core.valuation import (load_valuation, value_comp_score,
                            value_growth_score, value_hk_fcf_score)
from core.tradability import build_tradability

SIG_DIR = f"{config.QUANT_DIR}/live/signals"
NAME_CSV = f"{config.DATA_DIR}/fcf_result.csv"
CAND_MULT = 4

STRATEGY_LABELS = {
    "ICW_SW": "ICW双周+熊市切VGH (消融版)",
    "VG": "VG Top10 (value+盈利)",
    "VGH": "VGH Top10 (结构化剥离)",
    "DUAL_LEG": "DE模型+V20双腿 (旧版)",
}


def load_names():
    if not os.path.exists(NAME_CSV):
        return {}
    df = pd.read_csv(NAME_CSV, dtype={"code": str})
    return {format_qlib_code(c): n for c, n in zip(df["code"], df["name"])}


def resolve_signal_date(cal, want=None, strategy=None):
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

    # 调仓日程: ICW_SW = 每 10 个交易日 (双周), VG/VGH = 月末
    is_biweekly = strategy == "ICW_SW"
    if is_biweekly:
        print(f"[schedule] ICW_SW 双周调仓 (每 10 交易日), 当前信号日 {sig.date()} 无需对齐月末。",
              flush=True)
    else:
        month_last_bd = sig + pd.offsets.BMonthEnd(0)
        later_in_month = [d for d in cal if d > sig and d.month == sig.month and d.year == sig.year]
        if later_in_month or sig < month_last_bd - pd.Timedelta(days=3):
            print(f"[WARN] 信号日 {sig.date()} 不是 {sig.year}-{sig.month:02d} 的最后交易日 "
                  f"(该月末≈{month_last_bd.date()}) —— 策略口径是月末调仓, 非月末执行会偏离回测。\n"
                  f"       确认这是补跑/试算再继续。", flush=True)
    return sig


def fetch_hk_raw_close(hk_insts, sig_dt):
    """腾讯接口抓港股不复权真实收盘价。

    背景: qlib bin 里港股 $factor 字段被腾讯接口污染 (fetch_hk_data 把接口第8列
    当复权因子, 实际是每日随机值 0.002~0.73), close/factor 无法还原真实价。
    回测用复权 close 算收益不受影响; 生产下单价格必须真实, 故单独抓 raw 序列。
    """
    import time
    import requests
    out = {}
    for inst in hk_insts:
        code5 = inst[2:]
        url = ("https://web.ifzq.gtimg.cn/appstock/app/kline/kline?"
               f"param=hk{code5},day,,{sig_dt.strftime('%Y-%m-%d')},5,")
        try:
            r = requests.get(url, timeout=10,
                             headers={"Referer": "https://gu.qq.com",
                                      "User-Agent": "Mozilla/5.0"}).json()
            rows = (r.get("data") or {}).get(f"hk{code5}", {}).get("day") or []
            for row in reversed(rows):
                if pd.Timestamp(row[0]) <= sig_dt:
                    p = float(row[2])
                    if p > 0:
                        out[inst] = p
                    break
        except Exception:
            pass
        time.sleep(0.15)
    miss = [i for i in hk_insts if i not in out]
    if miss:
        print(f"[WARN] 港股真实价抓取失败 {len(miss)} 只: {miss[:5]}", flush=True)
    return out


def latest_prices(insts, sig_dt):
    """信号日真实收盘价: A股 = $close/$factor (bin 内还原); 港股 = 腾讯接口 raw 价"""
    insts = list(insts)
    a_insts = [i for i in insts if not i.startswith("hk")]
    hk_insts = [i for i in insts if i.startswith("hk")]
    out = {}
    if a_insts:
        px = D.features(a_insts, ["$close", "$factor"],
                        start_time=sig_dt - pd.Timedelta(days=10), end_time=sig_dt)
        if px is None or len(px) == 0:
            raise RuntimeError("[CHECK] 信号日价格拉取失败!")
        real = px["$close"] / px["$factor"]
        s = real.unstack(level=0).sort_index().ffill().iloc[-1]
        out.update({k: float(v) for k, v in s.items() if np.isfinite(v) and v > 0})
    if hk_insts:
        out.update(fetch_hk_raw_close(hk_insts, sig_dt))
    if not out:
        raise RuntimeError("[CHECK] 信号日无有效真实价格!")
    return out


def fix_hk_suspension(susp, universe, sig, close_px):
    """修复港股误判停牌: 数据源末日港股 $volume 常缺失但 $close 有值 (真停牌不会有当日价),
    这类属于数据缺陷而非停牌, 移出停牌集合。回测口径不动, 仅生产信号侧容错。"""
    n_fix = 0
    for inst in universe:
        if inst.startswith("hk") and (sig, inst) in susp:
            p = close_px.get((sig, inst), np.nan)
            if not pd.isna(p):
                susp.discard((sig, inst))
                n_fix += 1
    if n_fix:
        print(f"[fix] 港股 volume 缺失误判停牌, 解除 {n_fix} 只 (close 有值 = 非停牌)", flush=True)
    return susp


def bear_signal(sig_dt, bench_close, ma_period=60, bandwidth=0.03, hyst_days=3):
    """带宽+滞回熊市判断 (与 run_rolling.py band_bear 同逻辑, PIT: 只用 ≤ sig_dt 的数据)

    返回 True=牛市, False=熊市
    """
    hist = bench_close[bench_close.index <= sig_dt]
    if len(hist) < ma_period + hyst_days:
        return True
    ma = hist.rolling(ma_period, min_periods=20).mean()
    below = hist < ma
    deep = hist < ma * (1 - bandwidth)
    state = False
    run = 0
    for i in range(len(hist)):
        if not state:
            run = run + 1 if deep.iloc[i] else 0
            if run >= hyst_days:
                state = True
        elif not below.iloc[i]:
            state = False
            run = 0
    return not state


def _valid_rank_ic(pred, universe, valid_seg):
    """对 valid 段预测与真实 label(Ref($close,-20)/$close-1) 计算逐日 RankIC 均值。

    与 run_rolling.py 训练时的 valid IC 同口径 (RankICEval: 逐日截面 spearman,
    截面样本>5 才计入); 模型已冻结 → 结果确定可复现。
    """
    from scipy.stats import spearmanr
    vs, ve = valid_seg
    label = D.features(list(universe), ["Ref($close, -20) / $close - 1"],
                       start_time=vs, end_time=ve)
    if label is None or label.empty:
        return float("nan")
    lab = label.iloc[:, 0]
    # D.features 返回 (instrument, datetime), 统一切换为 (datetime, instrument)
    if lab.index.nlevels == 2 and lab.index.names[0] != "datetime":
        lab = lab.swaplevel().sort_index()
    ics = []
    for dt, cross in pred.groupby(level=0):
        # groupby(level=0) 的组内仍保留 (datetime, instrument) 双层索引, 降为 instrument 单层
        cross = cross.droplevel(0)
        y = lab.xs(dt, level=0).reindex(cross.index) if dt in lab.index.get_level_values(0) else None
        if y is None:
            continue
        m = y.notna() & cross.notna()
        if int(m.sum()) > 5:
            ics.append(spearmanr(cross[m], y[m])[0])
    return float(np.mean(ics)) if ics else float("nan")


def predict_icw(universe, year, sig):
    """加载 XGB/LGB/DE 三个冻结模型, 按 valid RankIC 加权 rank 融合 (ICW 口径)。

    步骤:
      1. 各模型多种子 zscore_mean 融合 (predict_frozen 内置)
      2. 对 manifest 的 valid 段推理 → 计算各模型 valid RankIC (IC 权重)
      3. 对信号日推理 → 逐日截面 rank(pct) × IC权重 求和 (与 run_rolling ICW 同式)
    """
    test_seg = ((sig - pd.Timedelta(days=90)).strftime("%Y-%m-%d"), sig.strftime("%Y-%m-%d"))
    model_preds, manifs, valid_ics = [], {}, {}
    for mname in ("XGBoost", "LightGBM", "DoubleEnsemble"):
        models, man = pipeline.load_frozen(mname, year)
        manifs[mname] = man
        pred = pipeline.predict_frozen(models, man, universe, test_seg)
        model_preds.append(pred)
        vic = _valid_rank_ic(
            pipeline.predict_frozen(models, man, universe, tuple(man["segments"]["valid"])),
            universe, tuple(man["segments"]["valid"]))
        valid_ics[mname] = vic
        print(f"[model] {mname} W{year} 冻结于 {man['frozen_at']} | "
              f"{man['n_seeds']}种子 | config_hash={man['config_hash'][:12]} | "
              f"valid RankIC={vic:.4f}", flush=True)

    # IC 加权: 负 IC 也参与 (与回测一致: max(ic, 0.01) 防止权重为负/为零)
    ics = [max(valid_ics[m], 0.01) for m in ("XGBoost", "LightGBM", "DoubleEnsemble")]
    ic_sum = sum(ics)
    ranks = [p.groupby(level=0).rank(pct=True) for p in model_preds]
    icw = sum(r * (w / ic_sum) for r, w in zip(ranks, ics))
    if sig not in icw.index.get_level_values(0):
        raise RuntimeError(f"[CHECK] ICW 融合后无信号日 {sig.date()} 截面!")
    return icw, manifs, valid_ics


def gen_icw_sw(sig, year, universe, val_piv, fcf_df, profit_df, bench_close,
               limit_up, susp, liq, close_px, capital):
    """ICW_SW: 牛市 10只ICW等权, 熊市 3只ICW(30%) + 7只VGH(70%) sleeve分仓"""
    is_bull = bear_signal(sig, bench_close,
                          ma_period=config.TIMING_MA,
                          bandwidth=config.TIMING_BANDWIDTH,
                          hyst_days=config.TIMING_HYSTERESIS)
    regime = "牛市" if is_bull else "熊市"
    print(f"[regime] {sig.date()} 趋势判断: {regime} "
          f"(MA{config.TIMING_MA} 带宽{config.TIMING_BANDWIDTH} 滞回{config.TIMING_HYSTERESIS}日)",
          flush=True)

    icw_pred, manifs, valid_ics = predict_icw(universe, year, sig)
    cross_icw = icw_pred.xs(sig, level=0)

    cand = strategy.eligible_candidates(cross_icw.index, sig, limit_up, susp, liq,
                                        config.MIN_CAND, strict=True, tag="ICW", px=close_px)
    print(f"[cand] 可交易候选 {len(cand)} 只 (已过滤一字涨停/停牌/流动性<2000万)", flush=True)

    icw_rank = cross_icw.reindex(cand).dropna().sort_values(ascending=False)

    if is_bull:
        sleeves = {"ICW": {"weight": 1.0, "topk": 10}}
        picks = {"ICW": list(icw_rank.index[:10 * CAND_MULT])}
    else:
        w_bw = config.SWITCH_W_BEAR
        n_bw = max(1, round(10 * w_bw))
        n_vgh = 10 - n_bw
        sleeves = {"ICW": {"weight": w_bw, "topk": n_bw},
                   "VGH": {"weight": 1 - w_bw, "topk": n_vgh}}
        vgh_score = value_hk_fcf_score(cand, sig, val_piv, fcf_df, profit_df,
                                       hk_factor=config.VHF_FACTOR,
                                       grow_w=config.VALUE_GROWTH_W).dropna().sort_values(ascending=False)
        picks = {
            "ICW": list(icw_rank.index[:n_bw * CAND_MULT]),
            "VGH": list(vgh_score.index[:n_vgh * CAND_MULT]),
        }
        print(f"[VGH] 熊市防御腿 {n_vgh} 只", flush=True)

    prices = latest_prices(set().union(*[set(v) for v in picks.values()]), sig)
    orders, summ = portfolio.allocate(picks, prices, capital, sleeves=sleeves)
    merged = portfolio.merge_overlap(orders)
    return merged, orders, summ, {"regime": regime, "is_bull": is_bull, "sleeves": sleeves,
                                  "valid_ics": valid_ics,
                                  "manifests": {k: {"frozen_at": v["frozen_at"],
                                                     "config_hash": v["config_hash"],
                                                     "git_commit": v["git_commit"]}
                                                for k, v in manifs.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, required=True, help="总投入资金(元)")
    ap.add_argument("--strategy", default="DUAL_LEG",
                    choices=["ICW_SW", "VG", "VGH", "DUAL_LEG"],
                    help="策略名 (ICW_SW/VG/VGH/DUAL_LEG)")
    ap.add_argument("--model", default="DoubleEnsemble", help="DUAL_LEG 用的模型 (默认DoubleEnsemble)")
    ap.add_argument("--year", type=int, default=0, help="冻结模型年份 (默认最新滚动窗口)")
    ap.add_argument("--date", default="", help="信号日 YYYY-MM-DD (默认最新交易日)")
    ap.add_argument("--held", default="", help="当前持仓 json {instrument: shares}")
    ap.add_argument("--no-archive", action="store_true", help="只打印不存档")
    args = ap.parse_args()

    datalayer.init_qlib()
    cal = datalayer.get_calendar()
    sig = resolve_signal_date(cal, args.date or None, strategy=args.strategy)
    # 默认年份 = 最新滚动窗口 (数据末日 2026-07 属于 W2027 的 test 段, 池/模型都用 2027)
    from core.universe import build_windows
    year = args.year or max(w["year"] for w in build_windows())
    names = load_names()

    fcf_df, profit_df = load_pit_caches()
    universe = [format_qlib_code(c) for c in build_dynamic_universe(year, fcf_df, profit_df)]
    strat_label = STRATEGY_LABELS[args.strategy]
    print(f"\n{'='*74}\n[signal] {strat_label} | 信号日 {sig.date()} | "
          f"池 {len(universe)} 只 | 资金 {args.capital:,.0f} 元\n{'='*74}", flush=True)

    val_piv = load_valuation()
    limit_up, susp, liq, close_px = build_tradability(universe, sig.strftime("%Y-%m-%d"),
                                                       sig.strftime("%Y-%m-%d"))
    susp = fix_hk_suspension(susp, universe, sig, close_px)
    bench = datalayer.load_benchmark()
    bench_close = (1 + bench).cumprod()

    if args.strategy == "ICW_SW":
        merged, orders, summ, extra = gen_icw_sw(
            sig, year, universe, val_piv, fcf_df, profit_df, bench_close,
            limit_up, susp, liq, close_px, args.capital)
        score_dfs = {}
    elif args.strategy == "VG":
        cand = strategy.eligible_candidates(universe, sig, limit_up, susp, liq,
                                            config.MIN_CAND, strict=True, tag="VG", px=close_px)
        print(f"[cand] 可交易候选 {len(cand)} 只", flush=True)
        vg_s = value_growth_score(cand, sig, val_piv, fcf_df, profit_df,
                                  hk_factor=config.VHF_FACTOR, grow_w=config.VALUE_GROWTH_W,
                                  q_w=config.VALUE_QUALITY_W).dropna().sort_values(ascending=False)
        prices = latest_prices(set(list(vg_s.index[:40])), sig)
        picks = {"VG": list(vg_s.index[:10 * CAND_MULT])}
        sleeves_cfg = {"VG": {"weight": 1.0, "topk": 10}}
        orders, summ = portfolio.allocate(picks, prices, args.capital, sleeves=sleeves_cfg)
        merged = portfolio.merge_overlap(orders)
        extra = {"sleeves": sleeves_cfg}
        score_dfs = {"score_VG": vg_s}
    elif args.strategy == "VGH":
        cand = strategy.eligible_candidates(universe, sig, limit_up, susp, liq,
                                            config.MIN_CAND, strict=True, tag="VGH", px=close_px)
        print(f"[cand] 可交易候选 {len(cand)} 只", flush=True)
        vgh_s = value_hk_fcf_score(cand, sig, val_piv, fcf_df, profit_df,
                                   hk_factor=config.VHF_FACTOR,
                                   grow_w=config.VALUE_GROWTH_W).dropna().sort_values(ascending=False)
        prices = latest_prices(set(list(vgh_s.index[:40])), sig)
        picks = {"VGH": list(vgh_s.index[:10 * CAND_MULT])}
        sleeves_cfg = {"VGH": {"weight": 1.0, "topk": 10}}
        orders, summ = portfolio.allocate(picks, prices, args.capital, sleeves=sleeves_cfg)
        merged = portfolio.merge_overlap(orders)
        extra = {"sleeves": sleeves_cfg}
        score_dfs = {"score_VGH": vgh_s}
    else:
        models, manifest = pipeline.load_frozen(args.model, year)
        print(f"[model] {args.model} W{year} 冻结于 {manifest['frozen_at']} | "
              f"{manifest['n_seeds']}种子 | config_hash={manifest['config_hash']} | "
              f"git={manifest['git_commit']}", flush=True)
        if sorted(universe) != manifest["universe"]:
            print(f"[WARN] 当前池({len(universe)}只) 与冻结时({manifest['universe_size']}只) 不一致",
                  flush=True)
        test_seg = ((sig - pd.Timedelta(days=90)).strftime("%Y-%m-%d"), sig.strftime("%Y-%m-%d"))
        pred = pipeline.predict_frozen(models, manifest, universe, test_seg)
        if sig not in pred.index.get_level_values(0):
            raise RuntimeError(f"[CHECK] 模型未对信号日 {sig.date()} 出分!")
        cross = pred.xs(sig, level=0)
        cand = strategy.eligible_candidates(cross.index, sig, limit_up, susp, liq,
                                            config.MIN_CAND, strict=True, tag="live", px=close_px)
        print(f"[cand] 可交易候选 {len(cand)} 只", flush=True)
        de_rank = cross.reindex(cand).dropna().sort_values(ascending=False)
        v_rank = value_comp_score(cand, sig, val_piv).dropna().sort_values(ascending=False)
        picks = {
            "DE": list(de_rank.index[:portfolio.SLEEVES["DE"]["topk"] * CAND_MULT]),
            "V20": list(v_rank.index[:portfolio.SLEEVES["V20"]["topk"] * CAND_MULT]),
        }
        prices = latest_prices(set(picks["DE"]) | set(picks["V20"]), sig)
        orders, summ = portfolio.allocate(picks, prices, args.capital)
        merged = portfolio.merge_overlap(orders)
        extra = {"sleeves": portfolio.SLEEVES, "model": args.model,
                 "manifest": {"frozen_at": manifest["frozen_at"],
                              "config_hash": manifest["config_hash"],
                              "git_commit": manifest["git_commit"]}}
        score_dfs = {"score_DE": de_rank, "score_V20": v_rank}

    print(f"\n【一、下单清单】信号日 {sig.date()} 收盘价计价, 次一交易日执行")
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
        print(f"  [WARN] {len(small)} 笔金额 <{portfolio.MIN_TICKET} 元, 佣金占比偏高")

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

    if not args.no_archive:
        tag = sig.strftime("%Y-%m-%d")
        d = f"{SIG_DIR}/{tag}/{args.strategy}"
        os.makedirs(d, exist_ok=True)
        merged.to_csv(f"{d}/orders.csv", index=False)
        orders.to_csv(f"{d}/orders_by_sleeve.csv", index=False)
        diff.to_csv(f"{d}/rebalance.csv", index=False)
        for sname, sdf in score_dfs.items():
            sdf.to_csv(f"{d}/{sname}.csv", header=["score"])
        meta = {"signal_date": tag, "strategy": args.strategy, "strategy_label": strat_label,
                "capital": args.capital, "year": year,
                "config_hash_ablate_sumd": config.ABLATE_SUMD,
                "summary": {k: v for k, v in summ.items() if k != "skipped"},
                "skipped": summ["skipped"], "turnover": tstat, **extra}
        with open(f"{d}/meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        with open(f"{d}/positions_after.json", "w") as f:
            json.dump(dict(zip(merged["instrument"], merged["shares"].astype(int))), f, indent=2)
        print(f"\n[archive] 已存档 → live/signals/{tag}/{args.strategy}/")
        print(f"[next] 下次运行: --held live/signals/{tag}/{args.strategy}/positions_after.json")


if __name__ == "__main__":
    main()
