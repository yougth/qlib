#!/usr/bin/env python3
"""tools/validate_pt_ledger.py —— 三组合与六腿 Paper Trading 台账数据体检
================================================================================
纯只读检查 (不修改任何台账), 每次盯市后可复跑:
  1) 六腿台账勾稽: cash + Σ(shares × last_px) ≈ nav_history 末值
  2) 净值快照日期序列 (无重复 / 升序 / 落到最新台账日)
  3) 三组合加权重算 vs 台账 (与 combo_track.compute 同口径)
  4) 新三腿持仓行情新鲜度 (vs 收盘日历; 台账可 intraday 领先一日) + 单日异动扫描
  5) 建仓价核对 (avg_cost vs 最新真实价) + 费用勾稽 (nav_init - nav ≈ 半费)
  6) 新三腿信号文件 vs 台账持仓 (集合一致性)
  7) 持仓市值占比 vs 信号目标权重 (整手偏差)

用法: cd qlib/new_quant && /usr/bin/python3 tools/validate_pt_ledger.py
"""
import json
import os

import pandas as pd

NEW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # new_quant
QLIB = os.path.dirname(NEW_DIR)  # qlib
LEDGER = os.path.join(QLIB, "quant", "live", "ledger")
TENCENT = os.path.join(QLIB, "data_cache", "tencent")
ETF_DIR = os.path.join(NEW_DIR, "data", "etf")
LOF_DIR = os.path.join(NEW_DIR, "data", "lof")
SIG_DIR = os.path.join(NEW_DIR, "outputs", "pt_signals")
LEGS = ["ICW_SW", "VG", "VGH", "M4", "LOF", "TREND"]

n_ok, n_bad = 0, 0


def check(name, cond, detail=""):
    global n_ok, n_bad
    if cond:
        n_ok += 1
    else:
        n_bad += 1
    print(("[OK] " if cond else "[!!] ") + name + (f" | {detail}" if detail else ""),
          flush=True)


def load_state(leg):
    with open(os.path.join(LEDGER, leg, "state.json")) as f:
        return json.load(f)


def px_series(leg, sym):
    """真实价序列 (index=date)"""
    if leg == "LOF":
        d = pd.read_csv(os.path.join(LOF_DIR, f"{sym[2:]}_px.csv"))
        s = d["close"].astype(float)
        s.index = pd.to_datetime(d["date"])
    else:
        d = pd.read_parquet(os.path.join(
            ETF_DIR if leg == "TREND" else TENCENT, f"{sym.lower()}.parquet"))
        s = (d["close"] / d["factor"]).astype(float)
        s.index = pd.to_datetime(d["date"])
    return s.sort_index()


DT = max(h["date"] for leg in LEGS for h in load_state(leg)["nav_history"])
DT_TS = pd.Timestamp(DT)
# 收盘日历 (sh000001, 不存在则退任一持仓 parquet)
_cal_p = os.path.join(TENCENT, "sh000001.parquet")
if not os.path.exists(_cal_p):
    _st0 = load_state("M4")
    _sym0 = sorted(_st0["positions"])[0]
    _cal_p = os.path.join(TENCENT_DIR, f"{_sym0.lower()}.parquet")
CAL_LAST = pd.to_datetime(pd.read_parquet(_cal_p, columns=["date"])["date"]).max()
CAL_LAST = pd.Timestamp(CAL_LAST).strftime("%Y-%m-%d")
print(f"体检基准日 (六腿台账最新): {DT} | 收盘日历最新: {CAL_LAST}"
      + (" (台账盘中领先)" if DT > CAL_LAST else "") + "\n", flush=True)

print("=" * 86)
print("一、六腿台账勾稽: cash + Σ(shares × last_px) ≈ nav_history 末值")
print("=" * 86)
for leg in LEGS:
    st = load_state(leg)
    mv = sum(v["shares"] * st["last_px"][k] for k, v in st["positions"].items())
    nav_calc, nav_rec = st["cash"] + mv, st["nav_history"][-1]["nav"]
    check(f"{leg:<7} 勾稽", abs(nav_calc - nav_rec) < 0.05,
          f"算 {nav_calc:>12,.2f} vs 记 {nav_rec:>12,.2f} | {len(st['positions'])}只 "
          f"仓位 {mv / nav_rec:>6.1%} | last_px_date {st.get('last_px_date', '?')}")

print()
print("=" * 86)
print("二、净值快照日期序列 (无重复 / 升序 / 落到基准日)")
print("=" * 86)
for leg in LEGS:
    ds = [h["date"] for h in load_state(leg)["nav_history"]]
    check(f"{leg:<7} 序列", len(ds) == len(set(ds)) and ds == sorted(ds) and ds[-1] == DT,
          f"{len(ds)}快照 {ds[0]} ~ {ds[-1]}")

print()
print("=" * 86)
print("三、三组合加权重算 vs 台账 (与 combo_track.compute 同口径)")
print("=" * 86)
COMBOS = {
    "COMBO_A": {"ICW_SW": 0.60, "M4": 0.40},
    "COMBO_B": {"ICW_SW": 0.54, "M4": 0.36, "LOF": 0.10},
    "COMBO_C": {"ICW_SW": 0.1454, "VG": 0.1971, "VGH": 0.2300,
                "M4": 0.1141, "TREND": 0.5040, "LOF": 0.1395},
}
for combo, w in COMBOS.items():
    st = load_state(combo)
    leg_s = {}
    for leg in w:
        h = load_state(leg)["nav_history"]
        leg_s[leg] = pd.Series({pd.Timestamp(x["date"]): x["nav"] for x in h})
    idx = [d for d in sorted(set().union(*[set(s.index) for s in leg_s.values()]))
           if d >= pd.Timestamp(st["start_date"])]
    M = pd.DataFrame({leg: leg_s[leg].reindex(idx).ffill() for leg in w}).dropna()
    rc = sum(w[leg] * M[leg].pct_change().fillna(0.0) for leg in w)
    nav = st["nav_init"] * (1 + rc).cumprod()
    d = abs(nav.iloc[-1] - st["nav_history"][-1]["nav"])
    check(f"{combo} 重算", d < 0.05,
          f"{nav.iloc[-1]:>12,.2f} vs 台账 {st['nav_history'][-1]['nav']:>12,.2f} | "
          f"权重和 {sum(w.values()):.4f} | 起点 {st['start_date']} "
          f"基点 {st['nav_init']:,.0f}")

print()
print("=" * 86)
print(f"四、数据源新鲜度 (新三腿持仓收盘日 vs 日历 {CAL_LAST}) + 最新日异动 + 盘中对照")
print("=" * 86)
for leg in ["M4", "LOF", "TREND"]:
    st = load_state(leg)
    lag, jumps = {}, []
    for sym in st["positions"]:
        s = px_series(leg, sym)
        last_d = s.index[-1].strftime("%Y-%m-%d")
        if last_d < CAL_LAST:            # 早于收盘日历才算真滞后
            lag[sym] = last_d
        if len(s) > 1:                   # 各标的最新两根K线异动扫描
            ret = s.iloc[-1] / s.iloc[-2] - 1
            if abs(ret) > 0.11:
                jumps.append(f"{sym} {ret:+.1%}")
    check(f"{leg:<7} 行情日", not lag,
          f"{len(st['positions'])} 只收盘到 {CAL_LAST}" + (f" | 滞后 {lag}" if lag else ""))
    check(f"{leg:<7} 异动", not jumps,
          "最新收盘日无 |日涨跌|>11% 标的" + (f" | {jumps}" if jumps else ""))
check("日历对照", DT >= CAL_LAST,
      f"台账基准日 {DT} vs 收盘日历最新 {CAL_LAST}"
      + (" (盘中领先, intraday 口径)" if DT > CAL_LAST else " (同步)"))

print()
print("=" * 86)
print("五、建仓价核对 (avg_cost vs 最新真实价) + 费用勾稽 (nav_init - nav ≈ 半费)")
print("=" * 86)
HALF = {"M4": 0.002, "LOF": 0.0005, "TREND": 0.0005}  # FEE_RT/2
for leg in ["M4", "LOF", "TREND"]:
    st = load_state(leg)
    dev, cost = [], 0.0
    for sym, v in st["positions"].items():
        s = px_series(leg, sym)
        p = float(s.loc[pd.Timestamp(CAL_LAST)]) if pd.Timestamp(CAL_LAST) in s.index \
            else float(s.iloc[-1])
        cost += v["shares"] * v["avg_cost"]
        if len(st["nav_history"]) == 1:  # 仅建仓日核对 (后续建仓价应自然偏离)
            dv = abs(v["avg_cost"] / p - 1)
            if dv > 0.005:
                dev.append(f"{sym} 成本{v['avg_cost']:.4f} vs 价{p:.4f} ({dv:.2%})")
    fee_real = st["nav_init"] - st["nav_history"][0]["nav"]
    fee_est = cost * HALF[leg]
    check(f"{leg:<7} 建仓价", not dev, f"{len(st['positions'])} 只对齐建仓日价"
          + (f" | 偏差>0.5%: {dev}" if dev else ""))
    check(f"{leg:<7} 费用", abs(fee_real - fee_est) < max(3.0, fee_est * 0.05),
          f"实扣 {fee_real:,.2f} vs 估 {fee_est:,.2f} (半费{HALF[leg]:.3%})")

print()
print("=" * 86)
print("六、新三腿信号文件 vs 台账持仓 (集合一致性)")
print("=" * 86)
for leg in ["M4", "LOF", "TREND"]:
    with open(os.path.join(SIG_DIR, f"{leg}_latest.json")) as f:
        sig = json.load(f)
    st = load_state(leg)
    same = set(sig["weights"]) == set(st["positions"])
    wsum = sum(sig["weights"].values())
    check(f"{leg:<7} 信号=持仓", same,
          f"信号 {len(sig['weights'])} 只 | asof {sig['signal_asof']} | 权重和 {wsum:.4f}")

print()
print("=" * 86)
print("七、持仓市值占比 vs 信号目标权重 (整手偏差核对)")
print("=" * 86)
for leg in ["M4", "LOF", "TREND"]:
    with open(os.path.join(SIG_DIR, f"{leg}_latest.json")) as f:
        sig = json.load(f)
    st = load_state(leg)
    nav = st["nav_history"][-1]["nav"]
    worst = []
    for sym, v in st["positions"].items():
        tgt = sig["weights"].get(sym, 0.0)
        act = v["shares"] * st["last_px"][sym] / nav
        if tgt > 0.02 and abs(act / tgt - 1) > 0.25:
            worst.append(f"{sym} 目标{tgt:.2%} 实际{act:.2%}")
    check(f"{leg:<7} 权重贴合", not worst,
          f"{len(st['positions'])} 只整手偏差 ≤25%" + (f" | 超限: {worst}" if worst else ""))

print()
print("=" * 86)
print(f"体检结论: 通过 {n_ok} 项 / 异常 {n_bad} 项")
print("=" * 86)
