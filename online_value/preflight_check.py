"""
上线前 / 每月交易前 自动体检 (preflight)
==========================================================
目的: 防止"代码或数据被动改动 → 持仓静默漂移 → 实盘偏离已验证策略"。
任一项 FAIL 则禁止当月交易, 必须先排查。

六道检查:
  A. 口径常量锁 : 流动性阈值/成本/TopK/月度换仓日规则 是否仍是回测时的值
  B. 指纹锁     : core/ 全部模块 + 入口脚本 + 判据文件(黄金持仓/回测基线) 的文件级 md5
                  与 golden/code_lock.json 比对 (任何人改了任何一行, 立刻报警到具体文件)
  C. 上游隔离   : core/ 与入口脚本不允许 import qlib 根目录的 vXX 实验脚本
                  (上线目录必须自包含, 否则上游改动会穿透进实盘)
  D. 数据鲜度   : 行情/估值截止日、两者落差、池子覆盖率、截面有效数
  E. 黄金回归   : 重算锁定年份的月度 Top20, 已锁定的期数必须逐行完全一致
                  (新增期数允许, 历史期数一个字都不能变)
  F. 历史不可变 : 锁定日之前的估值数据、2019年抽样复权行情的指纹必须不变
                  (E 只覆盖当年; 这一项覆盖全部历史, 抓上游回填/重抓)

用法:
  /usr/bin/python3 preflight_check.py                # 体检
  /usr/bin/python3 preflight_check.py --relock-code  # 只重建 B 代码指纹锁 (代码改了但持仓口径没变,
                                                     #   例如本次重构: 必须先确认 E 项 PASS)
  /usr/bin/python3 preflight_check.py --relock       # 重建全部锁, 含黄金持仓
                                                     #   (仅在你"有意"改策略并重跑回测后使用)
"""
import os, sys, json, glob, hashlib, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from qlib.data import D

from core import config
from core.universe import load_fin_caches, get_universe
from core.valuation import load_raw_valuation, compute_value_comp
from core.calendar_rules import get_calendar, monthly_signal_exec_dates, next_rebalance
from core.signal import year_context, year_rebalances, holdings_frame
from core.backtest import portfolio_backtest        # noqa: F401  (锁定引用, 确保可导入)

OUT_DIR = config.ONLINE_DIR
GOLD_DIR = config.GOLD_DIR
LOCK_JSON = f"{GOLD_DIR}/code_lock.json"
DATA_LOCK = f"{GOLD_DIR}/data_lock.json"
GOLD_HOLD = f"{GOLD_DIR}/holdings_2026_locked.csv"
GOLD_BASE = f"{GOLD_DIR}/backtest_baseline.csv"
GOLD_YEAR = 2026            # 黄金回归锁定的年份

# 回测时的口径 (改这里等于改策略, 必须同步重跑 run_backtest.py)
EXPECT = dict(LIQ_THRESHOLD=50000.0, FEE_ROUNDTRIP=0.004, TOPK=20,
              BT_END="2026-07-31", MONTHS_NORMAL=12)
# 被指纹锁覆盖的文件 (preflight 自身不含策略逻辑, 故不自锁)
# 判据文件(黄金持仓/回测基线)也必须入锁: 它们是所有"没变"结论的根据, 一旦被人改掉或被
# 误跑的 --relock 重建, E 项和回测对比就变成"跟改过的东西比", 会永远 PASS —— 那才是最坏的情况。
LOCK_FILES = sorted(glob.glob(f"{OUT_DIR}/core/*.py")) + [
    f"{OUT_DIR}/gen_holdings.py", f"{OUT_DIR}/run_backtest.py", f"{OUT_DIR}/monthly_update.py",
    GOLD_HOLD, GOLD_BASE]
# 历史行情抽样: 抓"复权基准被回填/全量重抓"这类静默改写 (只追加数据时该指纹恒定)
HIST_PX_YEAR, HIST_PX_N, HIST_PX_WIN = 2019, 20, ("2019-01-01", "2019-12-31")
# 上游实验脚本的 import 特征: 一旦出现就说明上线目录又被接回上游了
UPSTREAM_PAT = re.compile(r"^\s*(from|import)\s+v\d+\w*", re.M)

results = []


def chk(name, ok, detail):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def file_md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:12]


def part_a():
    print("\n--- A. 口径常量锁 ---", flush=True)
    chk("流动性阈值", config.LIQ_THRESHOLD == EXPECT["LIQ_THRESHOLD"],
        f"{config.LIQ_THRESHOLD:.0f} (期望{EXPECT['LIQ_THRESHOLD']:.0f}=真实500万/100, qlib volume单位为手)")
    chk("往返成本", config.FEE_ROUNDTRIP == EXPECT["FEE_ROUNDTRIP"],
        f"{config.FEE_ROUNDTRIP} (期望{EXPECT['FEE_ROUNDTRIP']})")
    chk("持仓只数", config.TOPK == EXPECT["TOPK"], f"Top{config.TOPK} 等权 (期望Top{EXPECT['TOPK']})")
    chk("回测末端", config.BT_END == EXPECT["BT_END"],
        f"{config.BT_END} (期望{EXPECT['BT_END']}; 改它=换基线, 须重建 golden/)")
    cal_list = get_calendar()
    n_gold = len(monthly_signal_exec_dates(GOLD_YEAR, cal_list))
    n_prev = len(monthly_signal_exec_dates(GOLD_YEAR - 1, cal_list))
    ok_sig = all(s < e for s, e in monthly_signal_exec_dates(GOLD_YEAR - 1, cal_list))
    chk("月度换仓日规则", n_prev == EXPECT["MONTHS_NORMAL"] and ok_sig and n_gold >= 1,
        f"{GOLD_YEAR-1}={n_prev}期(期望12) {GOLD_YEAR}={n_gold}期(随数据增长) "
        f"信号日严格早于执行日={ok_sig}")
    sig_r, exec_r = next_rebalance(cal_list)
    chk("下一换仓点可定位", sig_r is not None,
        f"信号日 {sig_r.date()} 收盘后出清单 → 执行日 {exec_r.date()} 尾盘换仓 (未来日期按工作日近似, 长假会偏早)"
        if sig_r is not None else "无法定位, 检查日历")
    return cal_list


def part_b(relock):
    print("\n--- B. 指纹锁 (core/ + 入口脚本 + 判据文件, 文件级) ---", flush=True)
    cur = {}
    for p in LOCK_FILES:
        if os.path.exists(p):
            cur[os.path.relpath(p, OUT_DIR)] = file_md5(p)
    if relock or not os.path.exists(LOCK_JSON):
        os.makedirs(GOLD_DIR, exist_ok=True)
        json.dump(cur, open(LOCK_JSON, "w"), indent=2, ensure_ascii=False)
        chk("指纹锁", True, f"{'重建' if relock else '首次生成'} {len(cur)}个文件指纹 → golden/code_lock.json")
        return
    old = json.load(open(LOCK_JSON))
    changed = sorted(k for k in cur if k in old and old[k] != cur[k])
    added = sorted(k for k in cur if k not in old)
    missing = sorted(k for k in old if k not in cur)
    ok = not (changed or added or missing)
    detail = f"{len(cur)}个文件未变动 (含2个判据文件)" if ok else \
        (f"改动={changed} 新增={added} 缺失={missing} → 若为有意改动, 必须先跑 run_backtest.py 再 --relock")
    chk("指纹锁", ok, detail)


def part_c():
    print("\n--- C. 上游隔离 (上线目录必须自包含) ---", flush=True)
    bad = []
    for p in [f for f in LOCK_FILES if f.endswith(".py")] + [os.path.abspath(__file__)]:
        if not os.path.exists(p):
            continue
        hits = UPSTREAM_PAT.findall(open(p, encoding="utf-8").read())
        if hits:
            bad.append(os.path.relpath(p, OUT_DIR))
    chk("无上游 vXX 依赖", not bad,
        "core/ 与入口脚本均不 import 任何 vXX 实验脚本" if not bad
        else f"以下文件仍在 import 上游实验脚本: {bad} → 上游改动会穿透进实盘, 必须切断")
    frozen = f"{OUT_DIR}/frozen_src"
    n = len(glob.glob(f"{frozen}/*.py")) if os.path.isdir(frozen) else 0
    chk("上游快照留档", n > 0, f"frozen_src/ 保留{n}个上游原始文件 (只作对照, 代码不引用)")


def part_d(cal_list):
    print("\n--- D. 数据鲜度 ---", flush=True)
    px_last = cal_list[-1]
    v = pd.read_csv(config.VAL_CACHE, sep='\t', dtype={"code": str}, usecols=["date", "code"])
    val_last = pd.to_datetime(v["date"]).max()
    today = pd.Timestamp.now().normalize()
    chk("行情截止日", (today - px_last).days <= 10, f"{px_last.date()} (距今{(today-px_last).days}天, 需<=10)")
    chk("估值截止日", (today - val_last).days <= 10, f"{val_last.date()} (距今{(today-val_last).days}天, 需<=10)")
    sig = min(px_last, val_last)
    chk("信号日可用", (today - sig).days <= 10, f"信号日={sig.date()} = min(行情,估值)")

    fcf_df, profit_df = load_fin_caches()
    universe = get_universe(px_last.year, fcf_df, profit_df)
    chk("池子规模", 150 <= len(universe) <= 500, f"{len(universe)}只 (正常250~350; <150须暂停)")
    val_piv = load_raw_valuation()
    vc = compute_value_comp(universe, sig, val_piv)
    cov = vc.notna().mean()
    chk("估值截面覆盖率", cov >= 0.95,
        f"{cov*100:.1f}% ({vc.notna().sum()}/{len(universe)}只有value_comp, 需>=95%)")
    px = D.features(universe, ["$close"], start_time=str((px_last - pd.Timedelta(days=15)).date()),
                    end_time=str(px_last.date()))
    n_px = px.reset_index()["instrument"].nunique() if px is not None and len(px) else 0
    chk("行情覆盖率", n_px / len(universe) >= 0.95, f"{n_px}/{len(universe)}只近15日有行情 (需>=95%)")


def part_e(cal_list, relock):
    print(f"\n--- E. 黄金回归 (重算{GOLD_YEAR}年月度Top20) ---", flush=True)
    fcf_df, profit_df = load_fin_caches()
    val_piv = load_raw_valuation(verbose=False)
    ctx = year_context(GOLD_YEAR, cal_list, fcf_df, profit_df, need_close=False, verbose=False)
    cur = holdings_frame(year_rebalances(ctx, val_piv))[["exec_date", "rank", "instrument", "value_comp"]]
    if relock or not os.path.exists(GOLD_HOLD):
        os.makedirs(GOLD_DIR, exist_ok=True)
        cur.to_csv(GOLD_HOLD, sep='\t', index=False)
        chk("黄金回归", True,
            f"{'重建' if relock else '首次生成'}锁定文件 {len(cur)}行 → {os.path.basename(GOLD_HOLD)}")
        return
    gold = pd.read_csv(GOLD_HOLD, sep='\t')
    gold["instrument"] = gold["instrument"].astype(str)
    locked_dates = set(gold["exec_date"])
    new_dates = sorted(set(cur["exec_date"]) - locked_dates)
    lost = sorted(locked_dates - set(cur["exec_date"]))
    sub = cur[cur["exec_date"].isin(locked_dates)]
    if lost:
        chk("黄金回归", False, f"锁定文件里的{len(lost)}期算不出来了: {lost[:3]} → 历史数据缺失, 禁止交易")
        return
    m = sub.merge(gold, on=["exec_date", "rank"], suffixes=("_now", "_gold"))
    bad = m[(m["instrument_now"] != m["instrument_gold"])
            | ((m["value_comp_now"] - m["value_comp_gold"]).abs() > 1e-4)]
    extra = f"; 另有新增{len(new_dates)}期({new_dates}) 属正常增长" if new_dates else ""
    chk("黄金回归", len(bad) == 0 and len(m) == len(gold),
        f"锁定的{sub['exec_date'].nunique()}期×Top{config.TOPK} 与锁定文件完全一致{extra}"
        if len(bad) == 0 and len(m) == len(gold)
        else f"{len(bad)}处不一致(比对{len(m)}/{len(gold)}行)! 例: "
             f"{bad.head(3)[['exec_date','rank','instrument_now','instrument_gold']].to_dict('records')}")


def val_hist_fp(cutoff=None):
    """cutoff(含)之前的估值数据指纹。先排序再按定宽格式化, 因此对行序调整、浮点写法变化免疫,
    只对真实数值改写敏感 —— 免疫这两件事很重要, 否则上游一次无害的重排就会造成假警报。"""
    df = pd.read_csv(config.VAL_CACHE, sep='\t', dtype={"code": str})
    cutoff = cutoff or str(df["date"].max())
    d = df[df["date"] <= cutoff].sort_values(["date", "code"])
    fp = hashlib.md5(d.to_csv(index=False, float_format="%.10g").encode()).hexdigest()[:12]
    return cutoff, fp, len(d)


def px_hist_fp(codes, win=HIST_PX_WIN):
    """固定样本股在固定历史窗口的行情指纹 (只追加新交易日时恒定不变)"""
    px = D.features(list(codes), ["$close", "$volume"], start_time=win[0], end_time=win[1])
    if px is None or not len(px):
        return None, 0
    px = px.sort_index().round(6)
    return hashlib.md5(px.to_csv(float_format="%.6f").encode()).hexdigest()[:12], len(px)


def part_f(relock):
    print("\n--- F. 历史不可变指纹 (界内一个数都不许变, 新数据只能追加) ---", flush=True)
    # 为什么需要这一项: E 项只重算当年(2026)。若 2019~2025 的估值或复权行情被上游改写,
    # E 项抓不到, 要等到下次跑 run_backtest.py 才暴露 —— 而那不是每月动作。
    if relock or not os.path.exists(DATA_LOCK):
        fcf_df, profit_df = load_fin_caches()
        codes = sorted(get_universe(HIST_PX_YEAR, fcf_df, profit_df))[:HIST_PX_N]
        cutoff, vfp, nrow = val_hist_fp()
        pfp, npx = px_hist_fp(codes)
        json.dump(dict(cutoff=cutoff, val_fp=vfp, val_rows=nrow, px_fp=pfp, px_rows=npx,
                       px_win=list(HIST_PX_WIN), px_codes=codes),
                  open(DATA_LOCK, "w"), indent=2, ensure_ascii=False)
        tag = "重建" if relock else "首次生成"
        chk("历史估值指纹", True, f"{tag}: {cutoff}及之前 {nrow}行 → {vfp}")
        chk("历史行情指纹", True, f"{tag}: 样本{len(codes)}只 × {HIST_PX_WIN[0][:4]}年 {npx}行 → {pfp}")
        return
    lk = json.load(open(DATA_LOCK))
    cutoff, vfp, nrow = val_hist_fp(lk["cutoff"])
    ok_v = vfp == lk["val_fp"] and nrow == lk["val_rows"]
    chk("历史估值指纹", ok_v,
        f"{cutoff}及之前 {nrow}行 指纹{vfp} 与锁定值一致" if ok_v else
        f"{cutoff}及之前的估值被改写! 锁定 {lk['val_rows']}行/{lk['val_fp']} → 现 {nrow}行/{vfp}; "
        f"历史一变, 回测结论即失效, 禁止交易")
    pfp, npx = px_hist_fp(lk["px_codes"], tuple(lk["px_win"]))
    ok_p = pfp == lk["px_fp"] and npx == lk["px_rows"]
    chk("历史行情指纹", ok_p,
        f"样本{len(lk['px_codes'])}只 × {lk['px_win'][0][:4]}年 {npx}行 指纹{pfp} 与锁定值一致" if ok_p else
        f"{lk['px_win'][0][:4]}年行情被改写! 锁定 {lk['px_rows']}行/{lk['px_fp']} → 现 {npx}行/{pfp}; "
        f"多半是复权基准被回填或全量重抓, 禁止交易")


def main():
    relock = "--relock" in sys.argv
    # 只重建代码指纹: 代码变了但持仓口径没变的情形(重构/改注释)。黄金持仓锁保持不动,
    # 这样"新代码重算后与旧锁一致"这个证据才不会被自己抹掉。
    relock_code = relock or "--relock-code" in sys.argv
    if relock:
        print("!! --relock 模式: 将用当前代码/数据重建**全部**锁(含黄金持仓)。仅当你有意修改策略并已重跑 "
              "run_backtest.py 确认 OOS 不下降时才可使用。", flush=True)
    elif relock_code:
        print("!! --relock-code 模式: 只重建代码指纹锁, 黄金持仓锁不动。"
              "前提是 E 黄金回归必须 PASS —— 否则说明口径真的变了, 不该用这个开关。", flush=True)
    config.init_qlib()
    print("=" * 92 + f"\n  上线体检 preflight  {pd.Timestamp.now():%Y-%m-%d %H:%M}\n" + "=" * 92, flush=True)
    cal_list = part_a()
    part_b(relock_code)
    part_c()
    part_d(cal_list)
    part_e(cal_list, relock)
    part_f(relock)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 92, flush=True)
    if n_fail == 0:
        print(f"  体检结论: 全部 {len(results)} 项 PASS → 可以按 latest_buy 清单交易", flush=True)
    else:
        print(f"  体检结论: {n_fail}/{len(results)} 项 FAIL → 【禁止交易】, 先排查以下项:", flush=True)
        for n, ok, d in results:
            if not ok:
                print(f"    - {n}: {d}", flush=True)
    print("=" * 92, flush=True)
    pd.DataFrame(results, columns=["check", "pass", "detail"]).to_csv(
        f"{OUT_DIR}/preflight_last.csv", sep='\t', index=False)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
