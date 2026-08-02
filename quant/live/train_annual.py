#!/usr/bin/env python3
"""
live/train_annual.py —— 年度训练 + 双跑校验 + 模型冻结 (上线前唯一的训练入口)
================================================================================
上线纪律: 模型一年只训练一次(每年 1 月, 用最新窗口), 之后 12 个月只做推理。
本脚本负责:
  1. 按 year 重建 PIT 股票池 (year-2 规则, 与回测同一函数)
  2. 训练并冻结模型 → pkl + manifest(SHA256/config_hash/git commit)
  3. --verify: 再训一遍到临时目录, 逐股逐分比对两次预测 → 不一致直接失败
     (这是"复现机制"的验收闸门, 通不过就不允许上线)

用法:
    cd quant
    PYTHONPATH=. python3 live/train_annual.py --year 2026 --seeds 3          # 训练+冻结
    PYTHONPATH=. python3 live/train_annual.py --year 2026 --seeds 3 --verify # 冻结+双跑校验
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

# 复现性: PYTHONHASHSEED 必须在解释器启动前生效 (详见 run_benchmark.py 同段注释)
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import shutil
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from core import config
from core import data as datalayer
from core import pipeline
from core.universe import build_dynamic_universe, format_qlib_code, load_pit_caches


def build_universe(year):
    fcf_df, profit_df = load_pit_caches()
    codes = build_dynamic_universe(year, fcf_df, profit_df)
    universe = [format_qlib_code(c) for c in codes]
    print(f"[universe] {year} 年 PIT 池 (仅用 ≤{year-2} 年报): {len(universe)} 只", flush=True)
    return universe


def verify_reproducible(name, year, universe, n_seeds, frozen_models, manifest):
    """再训一遍到临时目录, 与已冻结模型的预测逐值比对 (复现性验收闸门)"""
    tmp = f"{pipeline.FROZEN_DIR}/_verify_tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    print(f"\n{'='*70}\n[verify] 第二次训练 (临时目录), 用于双跑一致性校验\n{'='*70}", flush=True)
    pipeline.freeze(name, year, universe, n_seeds=n_seeds, out_dir=tmp)
    models2, man2 = pipeline.load_frozen(name, year, out_dir=tmp)

    test_seg = tuple(manifest["segments"]["test"])
    p1 = pipeline.predict_frozen(frozen_models, manifest, universe, test_seg)
    p2 = pipeline.predict_frozen(models2, man2, universe, test_seg)
    shutil.rmtree(tmp, ignore_errors=True)

    a, b = p1.align(p2, join="outer")
    if a.isna().any() or b.isna().any():
        raise RuntimeError("[VERIFY-FAIL] 两次训练的预测索引不一致 (股票/日期集合不同)!")
    dmax = float((a - b).abs().max())
    print(f"\n[verify] 两次预测最大绝对差 = {dmax:.3e} ({len(a)} 个 (日期,股票) 点)", flush=True)
    if dmax > 1e-10:
        raise RuntimeError(f"[VERIFY-FAIL] 双跑不一致 (max diff={dmax:.3e}), 存在未封锁的随机源, "
                           f"禁止上线!")
    print("[verify] ✅ 双跑逐值完全一致 → 复现机制有效, 允许上线", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DoubleEnsemble", help="模型名 (config.MODEL_CONFIGS)")
    ap.add_argument("--year", type=int, required=True, help="训练窗口年份 (如 2026)")
    ap.add_argument("--seeds", type=int, default=3, help="种子集成数 (降噪, 默认3)")
    ap.add_argument("--verify", action="store_true", help="再训一遍做双跑一致性校验")
    args = ap.parse_args()

    if args.model not in config.MODEL_CONFIGS:
        raise SystemExit(f"未知模型 {args.model}, 可选: {list(config.MODEL_CONFIGS)}")

    datalayer.init_qlib()
    universe = build_universe(args.year)

    print(f"\n{'='*70}\n[train_annual] {args.model} W{args.year} | 种子{args.seeds} | "
          f"冻结目录 {pipeline.FROZEN_DIR}\n{'='*70}", flush=True)
    pipeline.freeze(args.model, args.year, universe, n_seeds=args.seeds)
    models, manifest = pipeline.load_frozen(args.model, args.year)
    print(f"[train_annual] 冻结完成, SHA256 校验通过 ({len(models)} 个模型文件)", flush=True)

    if args.verify:
        verify_reproducible(args.model, args.year, universe, args.seeds, models, manifest)

    print(f"\n下一步: PYTHONPATH=. python3 live/monthly_signal.py --capital 50000", flush=True)


if __name__ == "__main__":
    main()
