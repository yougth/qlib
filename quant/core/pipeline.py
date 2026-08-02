"""
core.pipeline —— 训练管道: 年度训练 → 模型冻结 → manifest 凭证 (上线复现性核心)
================================================================================
研究阶段每次重跑都重新训练; 上线后必须"年度训练一次 + 月度只推理", 否则每月重训
会让持仓随机漂移。本模块提供:
  · zscore_mean:  多种子预测的截面 z-score 平均 (降噪, 非挑种子; 与 run_benchmark 共用)
  · freeze:       训练指定窗口模型 → pickle 落盘 + manifest(SHA256/配置哈希/git commit)
  · load_frozen:  加载冻结模型并校验 SHA256, 不一致直接报错 (防模型文件被悄悄替换)
  · predict_frozen: 用冻结模型对最新数据推理 (train/valid 分段与训练时逐字一致,
                    标准化参数只 fit train 段 → 无穿越且与训练时口径相同)

复现性契约: 同一 (model, year, n_seeds, 代码版本, 数据快照) → 逐股逐分完全一致。
manifest 里记录 config 哈希与 git commit, 任何一项变化都能被追溯。

关于 pickle: 模型文件仅由本模块 freeze() 自产自用, 且 load 前强制校验 manifest 中的
SHA256 —— 文件被替换或损坏会直接报错, 不接受任何外部来源的 pkl。
"""
import hashlib
import json
import os
import pickle
import subprocess
import time

import numpy as np
import pandas as pd

from . import config
from .dataset import build_datasetH
from .models import train_qlib_model
from .universe import build_windows

FROZEN_DIR = os.environ.get("QUANT_FROZEN_DIR") or f"{config.QUANT_DIR}/live/models"


def zscore_mean(preds):
    """多种子/多模型预测融合: 各自按日截面 z-score 标准化后取均值。
    直接平均原始分数是错的 —— 不同模型/种子的分数量纲不同, 会被方差最大的那个主导。"""
    acc = None
    for p in preds:
        g = p.groupby(level=0)
        z = (p - g.transform("mean")) / g.transform("std")
        acc = z if acc is None else acc + z
    return acc / len(preds)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=config.QUANT_DIR, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _config_hash(name):
    """模型超参 + 关键回测口径的指纹: 任何一处被改动, 冻结模型即视为失效"""
    payload = {"model": config.MODEL_CONFIGS[name], "topk": config.TOPK,
               "label_horizon": config.LABEL_HORIZON, "fee": config.FEE_ROUNDTRIP,
               "liq": config.LIQ_THRESHOLD}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _window_of(year):
    for w in build_windows():
        if w["year"] == year:
            return w
    raise ValueError(f"[CHECK] 无 W{year} 窗口定义 (build_windows 覆盖 2020~2026)")


def freeze(name, year, universe, n_seeds=1, out_dir=None):
    """训练 W{year} 窗口的 name 模型并冻结落盘。返回 (paths, manifest)。

    universe: 该年度 PIT 股票池 (由 build_dynamic_universe(year) 得出, 调用方传入以便复用)
    n_seeds:  >1 则冻结多个种子的模型, 推理时 zscore_mean 融合
    """
    out_dir = out_dir or FROZEN_DIR
    os.makedirs(out_dir, exist_ok=True)
    win = _window_of(year)
    seg = {"train": win["train"], "valid": win["valid"], "test": win["test"]}
    ds_class = config.MODEL_CONFIGS[name]["ds"]

    paths, t0 = [], time.time()
    for k in range(n_seeds):
        # seed_key 与 run_benchmark 完全同规则 → 冻结模型与基准回测是同一个模型
        seed_key = f"{name}:{win['name']}" if k == 0 else f"{name}:{win['name']}#s{k}"
        dataset = build_datasetH(seg, universe, ds_class=ds_class)
        model, _ = train_qlib_model(name, dataset, seed_key=seed_key)
        p = f"{out_dir}/{name}_{win['name']}_s{k}.pkl"
        with open(p, "wb") as f:
            pickle.dump(model, f)
        paths.append(p)
        print(f"  [freeze] {name} {win['name']} 种子{k + 1}/{n_seeds} → {os.path.basename(p)}",
              flush=True)
        del dataset, model

    manifest = {
        "model": name, "window": win["name"], "year": year,
        "n_seeds": n_seeds, "segments": {k: list(v) for k, v in seg.items()},
        "universe_size": len(universe), "universe": sorted(universe),
        "ds_class": ds_class,
        "config_hash": _config_hash(name), "git_commit": _git_commit(),
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_sec": round(time.time() - t0, 1),
        "files": [{"path": os.path.basename(p), "sha256": _sha256(p),
                   "bytes": os.path.getsize(p)} for p in paths],
    }
    mpath = f"{out_dir}/{name}_{win['name']}_manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  [freeze] manifest → {os.path.basename(mpath)} "
          f"(config_hash={manifest['config_hash']}, git={manifest['git_commit']})", flush=True)
    return paths, manifest


def load_frozen(name, year, out_dir=None, verify=True):
    """加载冻结模型 + manifest, 校验 SHA256 与 config_hash。返回 (models, manifest)。"""
    out_dir = out_dir or FROZEN_DIR
    win_name = f"W{year}"
    mpath = f"{out_dir}/{name}_{win_name}_manifest.json"
    if not os.path.exists(mpath):
        raise RuntimeError(f"[CHECK] 未找到冻结模型 manifest: {mpath}\n"
                           f"    请先运行: python3 live/train_annual.py --model {name} --year {year}")
    with open(mpath) as f:
        manifest = json.load(f)
    if verify:
        cur = _config_hash(name)
        if cur != manifest["config_hash"]:
            raise RuntimeError(
                f"[CHECK] config 已变更 (冻结时 {manifest['config_hash']} → 现在 {cur})!\n"
                f"    超参或回测口径被改动, 冻结模型已失效 → 必须重新年度训练。")
    models = []
    for fi in manifest["files"]:
        p = f"{out_dir}/{fi['path']}"
        if not os.path.exists(p):
            raise RuntimeError(f"[CHECK] 冻结模型文件缺失: {p}")
        if verify and _sha256(p) != fi["sha256"]:
            raise RuntimeError(f"[CHECK] {fi['path']} SHA256 不匹配, 模型文件已被修改!")
        with open(p, "rb") as f:
            models.append(pickle.load(f))
    return models, manifest


def predict_frozen(models, manifest, universe, test_seg, ds_class=None):
    """用冻结模型对 test_seg 推理。

    train/valid 分段取自 manifest (与训练时逐字一致) → handler 标准化参数只 fit train 段,
    与训练时同口径且不含未来信息; test_seg 换成最新区间即可得到当期打分。
    """
    seg = {"train": tuple(manifest["segments"]["train"]),
           "valid": tuple(manifest["segments"]["valid"]),
           "test": tuple(test_seg)}
    ds_class = ds_class or manifest.get("ds_class", "DatasetH")
    dataset = build_datasetH(seg, universe, ds_class=ds_class)
    preds = []
    for m in models:
        try:
            p = m.predict(dataset, segment="test")
        except TypeError:
            p = m.predict(dataset)
        if isinstance(p, pd.DataFrame):
            p = p.iloc[:, 0]
        preds.append(p)
    return preds[0] if len(preds) == 1 else zscore_mean(preds)
