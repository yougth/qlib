#!/usr/bin/env python3
"""
tools/feat_importance —— 特征重要性 (原生增益 gain + 置换重要性 perm)
================================================================================
两种口径回答的是**不同的问题**, 千万别混用:

  · gain (原生增益, 只有树模型有): 训练过程中这一列被用来分裂了多少次、带来多少
    增益。秒级即得, 但它衡量的是"拟合 **train** 段时的使用度" —— 高不代表对未来
    有预测力 (噪声列也能在训练集上刷出高 gain)。用途: 看模型在关注什么。

  · perm (置换重要性, 所有模型都能用): 在 **valid** 段把某一列按日截面内打乱,
    重新预测, 看 valid RankIC 掉多少。掉得多 = 这列真的在贡献样本外预测力;
    掉得少甚至上升 = 这列可以删。慢(每列一次 predict), 但这才是"该不该留这个
    特征"的唯一合理判据。DL 模型 (GRU/GATs/...) 无 gain, 只能用 perm。

无穿越: 复用 core.universe 的滚动窗口与 core.dataset.build_datasetH, 训练只用
train 段, 置换评估只用 valid 段, **全程不读 test 段**。所以本工具的结论可以拿来
指导特征取舍而不污染样本外评价。

用法:
    cd quant
    # 树模型原生增益 (最快, 先跑这个建立直觉)
    PYTHONPATH=. python3 tools/feat_importance.py --model LightGBM --year 2026
    PYTHONPATH=. python3 tools/feat_importance.py --model DoubleEnsemble --year 2026

    # 置换重要性: 全部列 (慢), 或只测 gain 前 40 列 (推荐)
    PYTHONPATH=. python3 tools/feat_importance.py --model LightGBM --year 2026 \
        --mode perm --perm-topn 40
    # DL 模型只能走 perm
    PYTHONPATH=. python3 tools/feat_importance.py --model GATs --year 2026 \
        --mode perm --perm-topn 20

产出: outputs/featimp_{model}_W{year}_{mode}.csv (tab 分隔, 与其它产出同格式)
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import warnings
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from qlib.data.dataset.handler import DataHandlerLP

from core import config
from core import data as datalayer
from core.universe import build_windows, build_dynamic_universe, format_qlib_code, load_pit_caches
from core.dataset import build_datasetH
from core.models import train_qlib_model

logging.getLogger("qlib").setLevel(logging.ERROR)


# ==================================================================
#  gain: 树模型原生增益 (注意: qlib 各 wrapper 训练时都传 x.values,
#        列名在 booster 里丢失成 f0/Column_0, 必须按**位置**映射回真名)
# ==================================================================
def native_gain(name, model, feat_cols):
    """返回 pd.Series(index=真实特征名, values=gain), 降序。不支持则返回 None"""
    n = len(feat_cols)

    def by_pos(idx_like, vals, prefix_strip):
        """把 f12 / Column_12 这类位置索引映射回 feat_cols[12]"""
        out = {}
        for k, v in zip(idx_like, vals):
            s = str(k)
            for p in prefix_strip:
                if s.startswith(p):
                    s = s[len(p):]
                    break
            if s.isdigit() and int(s) < n:
                out[feat_cols[int(s)]] = out.get(feat_cols[int(s)], 0.0) + float(v)
            elif s in feat_cols:
                out[s] = out.get(s, 0.0) + float(v)
        return pd.Series(out)

    if name == "DoubleEnsemble":
        # 官方 get_feature_importance 有坑: 它用 _model.feature_name() 得到
        # Column_k, 而 k 是**子特征集内**的位置, 不同子模型的 Column_0 指向不同
        # 真实列 → 直接相加会串味。这里用 sub_features 逐子模型还原真名。
        acc = {}
        for sub, feats, w in zip(model.ensemble, model.sub_features, model.sub_weights):
            imp = sub.feature_importance(importance_type="gain")
            for f, v in zip(list(feats), imp):
                acc[f] = acc.get(f, 0.0) + float(v) * float(w)
        return pd.Series(acc).sort_values(ascending=False)

    booster = getattr(model, "model", None)
    if booster is None:
        return None
    # LightGBM Booster
    if hasattr(booster, "feature_importance") and hasattr(booster, "feature_name"):
        vals = booster.feature_importance(importance_type="gain")
        return by_pos(booster.feature_name(), vals, ("Column_", "f")).sort_values(ascending=False)
    # XGBoost Booster
    if hasattr(booster, "get_score"):
        d = booster.get_score(importance_type="gain")
        return by_pos(list(d), list(d.values()), ("f",)).sort_values(ascending=False)
    # CatBoost
    if hasattr(booster, "get_feature_importance"):
        vals = booster.get_feature_importance()
        names = getattr(booster, "feature_names_", None) or list(range(len(vals)))
        return by_pos(names, vals, ("f",)).sort_values(ascending=False)
    # Linear: 用 |coef| (特征已标准化, 系数绝对值可比)
    coef = getattr(model, "coef_", None)
    if coef is not None and len(coef) == n:
        return pd.Series(np.abs(coef), index=feat_cols).sort_values(ascending=False)
    return None


# ==================================================================
#  perm: valid 段按日截面置换 → valid RankIC 下降幅度
# ==================================================================
def _rank_ic(pred, label):
    """按日截面 Spearman 相关的均值 (与 core.models.RankICEval 同口径)"""
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    if len(df) == 0:
        return float("nan")
    ics = df.groupby(level=0).apply(
        lambda g: g["p"].corr(g["y"], method="spearman") if len(g) > 5 else np.nan)
    return float(np.nanmean(ics.values))


def _predict_valid(model, dataset):
    try:
        p = model.predict(dataset, segment="valid")
    except TypeError:
        # pytorch TS 模型的 predict 没有 segment 形参, 只能预测 test 段 →
        # 调用方已把 dataset 的 test 段改指向 valid 区间, 这里直接调用即可
        p = model.predict(dataset)
    if isinstance(p, pd.DataFrame):
        p = p.iloc[:, 0]
    return p


def perm_importance(model, dataset, feat_cols, label_va, n_repeat, base_ic, rng):
    """在 handler 的 infer 矩阵上原地按日打乱一列 → 重算 valid RankIC → 复原。

    为什么打乱要**按日截面内**打乱而不是全局打乱: 全局打乱会同时破坏该列的时间
    结构(如动量的整体水平漂移), 相当于同时改了两件事; 按日内打乱只切断"这一天
    谁排前面"的信息, 正好对应 RankIC 衡量的东西。
    """
    dfi = dataset.handler._infer          # MultiIndex(datetime,instrument) × (feature/label,name)
    rows = []
    for i, f in enumerate(feat_cols, 1):
        col = ("feature", f)
        if col not in dfi.columns:
            print(f"  [{i}/{len(feat_cols)}] {f}: 不在 infer 矩阵中, 跳过", flush=True)
            continue
        orig = dfi[col].copy()
        drops = []
        for r in range(n_repeat):
            shuffled = dfi[col].groupby(level=0).transform(
                lambda s: rng.permutation(s.values))
            dfi[col] = shuffled
            ic = _rank_ic(_predict_valid(model, dataset), label_va)
            drops.append(base_ic - ic)
        dfi[col] = orig                    # 必须复原, 否则污染后续列的评估
        mean_drop = float(np.mean(drops))
        rows.append({"feature": f, "ic_drop": mean_drop,
                     "ic_drop_std": float(np.std(drops)) if n_repeat > 1 else 0.0,
                     "ic_after": base_ic - mean_drop, "n_repeat": n_repeat})
        print(f"  [{i}/{len(feat_cols)}] {f:<24} RankIC {base_ic:.4f} → "
              f"{base_ic - mean_drop:.4f}  (掉 {mean_drop:+.4f})", flush=True)
    return pd.DataFrame(rows).sort_values("ic_drop", ascending=False)


# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LightGBM", help="config.MODEL_CONFIGS 里的模型名")
    ap.add_argument("--year", type=int, default=2026, help="滚动窗口年 (build_windows 的 W{year})")
    ap.add_argument("--mode", choices=["gain", "perm", "both"], default="gain")
    ap.add_argument("--topn", type=int, default=40, help="终端展示前 N 行")
    ap.add_argument("--perm-topn", type=int, default=0,
                    help="perm 只测 gain 前 N 列 (0=全部列, 很慢)")
    ap.add_argument("--n-repeat", type=int, default=1, help="每列重复打乱次数 (取均值降噪)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.model not in config.MODEL_CONFIGS:
        raise SystemExit(f"未知模型 {a.model}; 可选: {list(config.MODEL_CONFIGS)}")

    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    win = next((w for w in build_windows() if w["year"] == a.year), None)
    if win is None:
        raise SystemExit(f"没有 {a.year} 年的窗口; 可选: {[w['year'] for w in build_windows()]}")
    universe = [format_qlib_code(c) for c in build_dynamic_universe(a.year, fcf_df, profit_df)]

    mcfg = config.MODEL_CONFIGS[a.model]
    is_ts = mcfg["ds"] == "TSDatasetH"
    # pytorch TS 模型的 predict() 不接受 segment → 把 test 段指向 valid 区间,
    # 这样 predict(dataset) 出来的就是 valid 段预测, 依旧一次都没读真正的 test 段。
    seg = {"train": win["train"], "valid": win["valid"],
           "test": win["valid"] if (is_ts and a.mode != "gain") else win["test"]}
    print(f"[featimp] {a.model} W{a.year} | 池 {len(universe)} 只 | "
          f"train{seg['train']} valid{seg['valid']} (评估段={seg['valid']}, 不读 test)",
          flush=True)

    dataset = build_datasetH(seg, universe, ds_class=mcfg["ds"])
    feat_cols = list(dataset.handler.get_cols(col_set="feature"))
    print(f"[featimp] 特征列 {len(feat_cols)} 个", flush=True)
    model, _ = train_qlib_model(a.model, dataset, seed_key=f"{a.model}:W{a.year}")

    os.makedirs(config.OUT_DIR, exist_ok=True)
    gain = None
    if a.mode in ("gain", "both"):
        gain = native_gain(a.model, model, feat_cols)
        if gain is None:
            print(f"[featimp] {a.model} 无原生 importance (非树模型) → 请用 --mode perm",
                  flush=True)
            if a.mode == "gain":
                raise SystemExit(2)
        else:
            gain = gain[gain > 0]
            out = f"{config.OUT_DIR}/featimp_{a.model}_W{a.year}_gain.csv"
            gain.rename("gain").to_frame().to_csv(out, sep="\t")
            print(f"\n=== {a.model} W{a.year} 原生增益 gain 前 {a.topn} ===", flush=True)
            tot = gain.sum()
            for i, (f, v) in enumerate(gain.head(a.topn).items(), 1):
                print(f"  {i:>3}. {f:<26}{v:>14.1f}  {v/tot*100:>6.2f}%", flush=True)
            print(f"[+] 已存 {out}  (共 {len(gain)}/{len(feat_cols)} 列被真正用到)",
                  flush=True)

    if a.mode in ("perm", "both"):
        label_va = dataset.handler.fetch(
            selector=slice(*win["valid"]), col_set="label",
            data_key=DataHandlerLP.DK_L).iloc[:, 0].dropna()
        base_ic = _rank_ic(_predict_valid(model, dataset), label_va)
        print(f"\n[featimp] 未扰动 valid RankIC = {base_ic:.4f} (基线)", flush=True)
        if abs(base_ic) < 1e-6:
            raise RuntimeError("[CHECK] 基线 valid RankIC≈0, 模型没学到东西, "
                               "置换重要性无意义 —— 先修模型收敛问题")
        cols = feat_cols
        if a.perm_topn > 0:
            g = gain if gain is not None else native_gain(a.model, model, feat_cols)
            cols = list(g.head(a.perm_topn).index) if g is not None else feat_cols[:a.perm_topn]
            print(f"[featimp] 只测 {len(cols)} 列 (--perm-topn)", flush=True)
        rng = np.random.default_rng(a.seed)
        df = perm_importance(model, dataset, cols, label_va, a.n_repeat, base_ic, rng)
        out = f"{config.OUT_DIR}/featimp_{a.model}_W{a.year}_perm.csv"
        df.to_csv(out, sep="\t", index=False)
        print(f"\n=== {a.model} W{a.year} 置换重要性 (valid RankIC 下降, 降序) ===", flush=True)
        print(df.head(a.topn).to_string(index=False), flush=True)
        neg = df[df["ic_drop"] <= 0]
        print(f"\n[!] {len(neg)}/{len(df)} 列打乱后 RankIC 不降反升 → 这些列大概率是"
              f"噪声, 是删特征的第一候选:", flush=True)
        print("    " + ", ".join(neg["feature"].head(20).tolist()), flush=True)
        print(f"[+] 已存 {out}", flush=True)


if __name__ == "__main__":
    main()
