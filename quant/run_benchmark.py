#!/usr/bin/env python3
"""
run_benchmark —— qlib 全模型无穿越基准 (Phase A 单切分初筛 + Phase B top-K 完整滚动)
================================================================================
目标: 用同一份无穿越数据层 (core.dataset / core.universe) 驱动 qlib 全部内置模型,
      头对头比较, 回答"哪个模型效果最好", 并与 24.88% 的 ENS / value_comp Top20 对照。

无穿越单点保证: 所有模型共用 core.dataset.build_datasetH (Alpha158Enhanced + 同 processors
+ embargo 分段 + 20 日后向 label + train 段 fit 标准化) 与 core.universe 的 PIT 池。

方法 (本机无 CUDA, DL 走 CPU 很慢, 故分两阶段):
  Phase A  全模型在单一固定切分上初筛 (train2016-2021 / valid2022 embargo / test2023-2026H1),
           每个模型独立子进程 + wall-clock 预算隔离; 超时/报错显式记 timeout/failed 不静默跳过,
           按 OOS RankIC 排名。
  Phase B  取 Phase A 前 N 名进入完整 7 窗口滚动, 与 value_comp Top20 / POOL 等权严格头对头。

用法:
  python3 run_benchmark.py                      # Phase A 全模型初筛
  python3 run_benchmark.py --phase A --models XGBoost,LightGBM   # 只筛指定模型
  python3 run_benchmark.py --phase B --models XGBoost,LightGBM   # 指定模型完整滚动
  python3 run_benchmark.py --worker XGBoost --out outputs/_bench_XGBoost.json  # 内部单模型 worker
"""
import os

# --- macOS 原生崩溃修复: PyTorch/qlib 与 numpy 各自携带 libomp, 重复加载会段错误(无 Python traceback).
#     必须在任何 numpy/torch/qlib 导入之前设置, 否则全部 pytorch DL 模型 (GRU/LSTM/... ) worker 秒崩. ---
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import gc
import json
import time
import random
import argparse
import warnings
import logging
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed=42):
    random.seed(seed)
    import numpy as _np
    _np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from core import config
from core import data as datalayer
from core.universe import (build_windows, build_dynamic_universe, format_qlib_code,
                           load_pit_caches)
from core.valuation import load_valuation, value_comp_score
from core.dataset import build_datasetH
from core.models import train_qlib_model
from core.tradability import build_tradability, get_month_end_dates, build_candidates
from core.backtest import (portfolio_backtest, calc_metrics, annualized_since,
                           print_yearly_table, print_metric_table)

logging.getLogger("qlib").setLevel(logging.ERROR)

OUT_DIR = config.OUT_DIR
TOPK = config.TOPK
SPECIAL = ("VALUE20", "POOL_EW")          # 非训练对照系 (纯估值 / 全池等权)


# ==================================================================
#  信号 → 回测: 把一段预测 (或 value 打分) 转成月频 T+1 组合与图1指标
# ==================================================================
def _signals_for_window(name, pred, universe, cal, val_piv, xs, xe,
                        limit_up, susp, liq, fwd_mat, topk, ic_records=None,
                        rebalances=None, win_name=""):
    """单窗口: 遍历月末信号日, 产出 (exec_dt, 持仓) 追加到 rebalances[name]。
    pred=None 表示 value/pool 对照系。同时累计该模型 OOS 截面 IC/RankIC。"""
    cal_idx = pd.DatetimeIndex(cal)
    sig_dates = get_month_end_dates(cal, xs, xe)
    n_ok = 0
    for sig_dt in sig_dates:
        if pred is not None:
            if sig_dt not in pred.index.get_level_values(0):
                continue
            cross = pred.xs(sig_dt, level=0)
            base_idx = cross.index
        else:
            base_idx = pd.Index(universe)
        cand = build_candidates(base_idx, sig_dt, limit_up, susp, liq)
        if len(cand) < topk:
            continue
        if pred is not None:
            score = cross.reindex(cand)
        else:
            score = value_comp_score(cand, sig_dt, val_piv).reindex(cand)

        if fwd_mat is not None and sig_dt in fwd_mat.index and ic_records is not None:
            fwd = fwd_mat.loc[sig_dt].reindex(cand)
            df = pd.concat([score, fwd], axis=1).dropna()
            if len(df) >= 10:
                ic_records.append({"window": win_name, "model": name, "sig_date": sig_dt,
                                   "ic": df.iloc[:, 0].corr(df.iloc[:, 1]),
                                   "rank_ic": df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")})

        pos = int(cal_idx.searchsorted(sig_dt)) + 1
        if pos >= len(cal_idx):
            continue
        exec_dt = cal_idx[pos]
        if name == "POOL_EW":
            top = cand.tolist()
        else:
            top = score.dropna().nlargest(topk).index.tolist()
        if not top:
            continue
        rebalances[name].append((exec_dt, top))
        n_ok += 1
    return n_ok


# ==================================================================
#  Phase A worker: 单模型 单切分, 输出 JSON 指标
# ==================================================================
def run_worker(name, out_path):
    t0 = time.time()
    result = {"model": name, "status": "ok"}
    try:
        datalayer.init_qlib()
        fcf_df, profit_df = load_pit_caches()
        cal = datalayer.get_calendar()
        val_piv = load_valuation()
        codes = build_dynamic_universe(config.BENCH_UNIVERSE_YEAR, fcf_df, profit_df)
        universe = [format_qlib_code(c) for c in codes]
        seg = config.BENCH_SPLIT
        xs, xe = seg["test"]
        print(f"[worker {name}] 冻结股票池 {len(universe)} 只 | "
              f"train{seg['train']} valid{seg['valid']}(embargo) 信号{seg['test']}", flush=True)

        if name in SPECIAL:
            pred = None
        else:
            mcfg = config.MODEL_CONFIGS[name]
            dataset = build_datasetH(seg, universe, ds_class=mcfg["ds"])
            _, pred = train_qlib_model(name, dataset)
            del dataset
            gc.collect()

        limit_up, susp, liq = build_tradability(universe, xs, xe)
        fwd_mat = datalayer.forward_return_matrix(universe, xs, xe)
        price_mat = datalayer.load_price_matrix(universe, start="2022-11-01")
        bench = datalayer.load_benchmark()

        topk = 20 if name == "VALUE20" else TOPK
        ic_records, rebalances = [], {name: []}
        n_ok = _signals_for_window(name, pred, universe, cal, val_piv, xs, xe,
                                   limit_up, susp, liq, fwd_mat, topk,
                                   ic_records=ic_records, rebalances=rebalances, win_name="A")
        if not rebalances[name]:
            raise RuntimeError("无有效月度调仓信号 (预测截面缺失或候选不足)")

        rets, avg_to, n_buys = portfolio_backtest(rebalances[name], price_mat)
        rets = rets[rets.index >= pd.Timestamp(config.BENCH_BT_START)]
        m = calc_metrics(rets, bench)
        icv = np.nanmean([r["ic"] for r in ic_records]) if ic_records else np.nan
        ricv = np.nanmean([r["rank_ic"] for r in ic_records]) if ic_records else np.nan
        yearly = {int(yr): round((1 + g).prod() - 1, 4)
                  for yr, g in rets.groupby(rets.index.year)}
        result.update({
            "oos_rankic": _r(ricv), "oos_ic": _r(icv),
            "ar": _r(m["ar"]), "vol": _r(m["vol"]), "sharpe": _r(m["sharpe"], 3),
            "mdd": _r(m["mdd"]), "calmar": _r(m["calmar"], 3),
            "excess_ar": _r(m.get("excess_ar", np.nan)),
            "turnover": _r(avg_to), "n_buys": int(n_buys), "n_signals": n_ok,
            "yearly": yearly})
    except Exception as e:
        import traceback
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    result["seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"[worker {name}] status={result['status']} 用时{result['seconds']}s", flush=True)


def _r(x, d=4):
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else round(float(x), d)


# ==================================================================
#  Phase A 编排: 逐模型子进程 + wall-clock 预算, 汇总排名
# ==================================================================
def run_phase_a(models, budget):
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = models or (list(config.MODEL_CONFIGS) + list(SPECIAL))
    rows = []
    for name in targets:
        out_json = f"{OUT_DIR}/_bench_{name}.json"
        if os.path.exists(out_json):
            os.remove(out_json)
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", name, "--out", out_json]
        print(f"\n{'='*72}\n[Phase A] {name}  (wall-clock 预算 {budget}s)\n{'='*72}", flush=True)
        # pytorch DL 模型在 macOS CPU 上多线程 OpenMP 会与 numpy/torch 的 libomp 冲突而秒崩(无traceback);
        # 仅对 pytorch worker 强制单线程 (KMP_DUPLICATE_LIB_OK 单独不足). 树模型保持多线程全速.
        env = os.environ.copy()
        if "pytorch" in config.MODEL_CONFIGS.get(name, {}).get("module_path", ""):
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", KMP_DUPLICATE_LIB_OK="TRUE")
        t0 = time.time()
        try:
            subprocess.run(cmd, timeout=budget, check=False, env=env)
        except subprocess.TimeoutExpired:
            print(f"[Phase A] {name} TIMEOUT >{budget}s → 记 timeout, 继续下一个", flush=True)
            rows.append({"model": name, "status": "timeout", "seconds": budget})
            continue
        if os.path.exists(out_json):
            with open(out_json) as f:
                rows.append(json.load(f))
        else:
            rows.append({"model": name, "status": "failed",
                         "error": "worker 无输出 (可能崩溃/OOM)",
                         "seconds": round(time.time() - t0, 1)})
    df = pd.DataFrame(rows)
    df["_rk"] = df["status"].map(lambda s: 0 if s == "ok" else 1)
    df["_sort"] = df.get("oos_rankic", pd.Series([None] * len(df))).fillna(-99)
    df = df.sort_values(["_rk", "_sort"], ascending=[True, False]).drop(columns=["_rk", "_sort"])
    csv_path = f"{OUT_DIR}/benchmark_phaseA.csv"
    df.drop(columns=["yearly"], errors="ignore").to_csv(csv_path, sep="\t", index=False)
    _print_phase_a(df)
    print(f"\n[+] Phase A 排名已存 {csv_path}", flush=True)
    ok = df[df["status"] == "ok"]
    topb = ok["model"].head(config.BENCH_TOPB).tolist()
    print(f"[+] Phase A 前 {config.BENCH_TOPB} 名 (进入 Phase B 建议): {topb}", flush=True)
    return df


def _print_phase_a(df):
    # 标准三段式输出: 1)分年收益表 2)图1九指标表 3)排名; 分年数据来自各 worker JSON 的 yearly 字段
    print(f"\n{'='*96}\n  Phase A 分年收益表 (单切分, 信号期 2022-12~2026-06)\n{'='*96}", flush=True)
    years = ["2023", "2024", "2025", "2026"]
    print(f"{'模型':<16}" + "".join(f"{y:>9}" for y in years), flush=True)
    for _, r in df.iterrows():
        y = r.get("yearly")
        if not isinstance(y, dict):
            continue
        print(f"{str(r['model']):<16}" +
              "".join(f"{y[k]*100:>8.1f}%" if k in y else f"{'-':>9}" for k in years), flush=True)
    print(f"\n{'='*96}\n  Phase A 全模型初筛排名 (单切分 test 2023~2026H1, 按 OOS RankIC 降序)\n{'='*96}",
          flush=True)
    hdr = (f"{'模型':<16}{'状态':<9}{'RankIC':>8}{'IC':>8}{'年化':>9}{'波动':>8}"
           f"{'Sharpe':>8}{'回撤':>9}{'Calmar':>8}{'换手':>8}{'次数':>7}{'超额300':>9}{'秒':>8}")
    print(hdr, flush=True)
    for _, r in df.iterrows():
        def g(k, pct=False, d=3):
            v = r.get(k)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return f"{'-':>8}"
            return f"{v*100:>7.1f}%" if pct else f"{v:>8.{d}f}"
        line = (f"{str(r['model']):<16}{str(r['status']):<9}"
                f"{g('oos_rankic', d=4)}{g('oos_ic', d=4)}{g('ar', pct=True)}{g('vol', pct=True)}"
                f"{g('sharpe', d=2)}{g('mdd', pct=True)}{g('calmar', d=2)}"
                f"{g('turnover', pct=True)}{str(r.get('n_buys', '-')):>7}"
                f"{g('excess_ar', pct=True)}"
                f"{str(r.get('seconds', '-')):>8}")
        print(line, flush=True)
        if r.get("status") != "ok" and r.get("error"):
            print(f"    └─ {r['error']}", flush=True)


# ==================================================================
#  Phase B: 指定模型进入完整 7 窗口滚动, 与 value_comp/pool 头对头
# ==================================================================
def run_phase_b(models, tag=""):
    os.makedirs(OUT_DIR, exist_ok=True)
    if not models:
        raise SystemExit("Phase B 需 --models 指定模型 (通常取 Phase A 前几名)")
    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()
    val_piv = load_valuation()
    windows = build_windows()

    tags = list(models) + list(SPECIAL)
    rebalances = {t: [] for t in tags}
    ic_records = []
    for win in windows:
        y = win["year"]
        xs, xe = win["test"]
        codes = build_dynamic_universe(y, fcf_df, profit_df)
        universe = [format_qlib_code(c) for c in codes]
        print(f"\n{'='*70}\n[Phase B {win['name']}] 池{len(universe)}只 | train{win['train']} "
              f"valid{win['valid']}(embargo) 信号{xs}~{xe}\n{'='*70}", flush=True)
        limit_up, susp, liq = build_tradability(universe, xs, xe)
        fwd_mat = datalayer.forward_return_matrix(universe, xs, xe)
        seg = {"train": win["train"], "valid": win["valid"], "test": win["test"]}
        preds = {}
        for name in models:
            try:
                dataset = build_datasetH(seg, universe, ds_class=config.MODEL_CONFIGS[name]["ds"])
                _, preds[name] = train_qlib_model(name, dataset)
                del dataset
                gc.collect()
            except Exception as e:
                print(f"    [Phase B] {win['name']} {name} 训练失败: {e} → 该窗口跳过该模型", flush=True)
                preds[name] = None
        for name in models:
            if preds[name] is None:
                continue
            _signals_for_window(name, preds[name], universe, cal, val_piv, xs, xe,
                                limit_up, susp, liq, fwd_mat, TOPK,
                                ic_records=ic_records, rebalances=rebalances, win_name=win["name"])
        _signals_for_window("VALUE20", None, universe, cal, val_piv, xs, xe,
                            limit_up, susp, liq, fwd_mat, 20,
                            ic_records=ic_records, rebalances=rebalances, win_name=win["name"])
        _signals_for_window("POOL_EW", None, universe, cal, val_piv, xs, xe,
                            limit_up, susp, liq, None, TOPK,
                            ic_records=None, rebalances=rebalances, win_name=win["name"])
        # 每窗口落盘 checkpoint, 长任务中途崩溃/重启不丢已完成窗口
        import pickle
        suffix = f"_{tag}" if tag else ""
        with open(f"{OUT_DIR}/_phaseB_ckpt{suffix}.pkl", "wb") as f:
            pickle.dump({"done_windows": win["name"], "rebalances": rebalances,
                         "ic_records": ic_records}, f)

    _finalize_phase_b(tags, models, rebalances, ic_records, tag=tag)


def _finalize_phase_b(tags, models, rebalances, ic_records, tag=""):
    all_insts = sorted({i for t in tags for _, tops in rebalances[t] for i in tops})
    price_mat = datalayer.load_price_matrix(all_insts)
    bench = datalayer.load_benchmark()
    ic_df = pd.DataFrame(ic_records)
    ic_agg = ic_df.groupby("model")[["ic", "rank_ic"]].mean().to_dict("index") if len(ic_df) else {}

    BT_START = pd.Timestamp(config.BT_START)
    ALIGN = config.ALIGN_START
    LABEL = {**{m: f"{m} Top10" for m in models},
             "VALUE20": "value_comp Top20", "POOL_EW": "十年双正池等权"}
    metric_rows, yearly_all = [], {}
    for tag in tags:
        if not rebalances[tag]:
            continue
        rets, avg_to, n_buys = portfolio_backtest(rebalances[tag], price_mat)
        rets = rets[rets.index >= BT_START]
        m = calc_metrics(rets, bench)
        d = ic_agg.get(tag, {})
        metric_rows.append({
            "策略": LABEL.get(tag, tag), "年化收益": m["ar"],
            "年化(剔2020)": annualized_since(rets, ALIGN), "年化波动": m["vol"],
            "Sharpe": m["sharpe"], "最大回撤": m["mdd"], "Calmar": m["calmar"],
            "IC": d.get("ic", float("nan")), "RankIC": d.get("rank_ic", float("nan")),
            "换手率": avg_to, "交易次数": n_buys, "超额vsHS300": m.get("excess_ar", float("nan"))})
        yearly_all[tag] = {yr: (1 + g).prod() - 1 for yr, g in rets.groupby(rets.index.year)}

    bench_bt = bench[bench.index >= BT_START]
    mb = calc_metrics(bench_bt)
    yearly_all["HS300"] = {yr: (1 + g).prod() - 1 for yr, g in bench_bt.groupby(bench_bt.index.year)}
    metric_rows.append({"策略": "沪深300基准", "年化收益": mb["ar"],
                        "年化(剔2020)": annualized_since(bench_bt, ALIGN), "年化波动": mb["vol"],
                        "Sharpe": mb["sharpe"], "最大回撤": mb["mdd"], "Calmar": mb["calmar"],
                        "IC": float("nan"), "RankIC": float("nan"), "换手率": 0.0,
                        "交易次数": 0, "超额vsHS300": 0.0})

    order = [t for t in tags if rebalances[t]] + ["HS300"]
    labels = {**LABEL, "HS300": "沪深300基准"}
    print_yearly_table(yearly_all, order, labels, title="Phase B 分年收益 (完整7窗口滚动)")
    mdf = pd.DataFrame(metric_rows)
    print_metric_table(mdf, title="Phase B 整体指标 (图1口径; 与 value_comp Top20 头对头)")
    suffix = f"_{tag}" if tag else ""
    csv_path = f"{OUT_DIR}/benchmark_phaseB{suffix}.csv"
    mdf.to_csv(csv_path, sep="\t", index=False)
    pd.DataFrame(yearly_all).T.to_csv(f"{OUT_DIR}/benchmark_phaseB_yearly{suffix}.csv", sep="\t")
    if len(ic_df):
        ic_df.to_csv(f"{OUT_DIR}/benchmark_phaseB_ic{suffix}.csv", sep="\t", index=False)
    print(f"\n[+] Phase B 结果已存 {csv_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B"], default="A")
    ap.add_argument("--models", default="", help="逗号分隔模型名; 空=全部(Phase A)")
    ap.add_argument("--budget", type=int, default=config.BENCH_MODEL_BUDGET_SEC,
                    help="Phase A 单模型 wall-clock 秒预算")
    ap.add_argument("--worker", default="", help="内部: 单模型 worker 名")
    ap.add_argument("--out", default="", help="内部: worker JSON 输出路径")
    ap.add_argument("--tag", default="", help="Phase B 输出文件后缀, 多进程并行时隔离结果")
    args = ap.parse_args()

    if args.worker:
        run_worker(args.worker, args.out)
        return
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.phase == "A":
        run_phase_a(models, args.budget)
    else:
        run_phase_b(models, tag=args.tag)


if __name__ == "__main__":
    main()
