#!/usr/bin/env python3
"""
滚动十年双正股票池 × XGB/LGB 双模型 × 月频Top10 —— 最近6年+ (2020~2026H1) 回测
================================================================================
不穿越设计 (逐项对应长期记忆《量化回测必须检查特征穿越与前视偏差清单》):
1. 股票池PIT: 每个回测年Y用 PIT 缓存 (含退市股) 重建 "Y-11..Y-2 十年净利+FCF双正" 池;
   年报Y-1在Y年4月底才披露完 → 1月调仓最多只能用到Y-2年报 (year-2规则)
2. 滚动5年训练: train = Y-5..Y-2 (4年), valid = Y-1 (截至11-30, embargo 1个月,
   防止20日label在valid末期偷看测试期价格); 无任何全局/跨窗口特征选择
3. 信号T+1执行: 月末收盘信号, 次一交易日收盘价成交; 测试段信号从(Y-1)年12月末起,
   修复"每年1月空仓"缺陷
4. 交易成本: 单边换手 × 0.4% 往返摩擦
5. 收敛保证: XGB/LGB 均用验证集日度RankIC最大化早停 (严禁RMSE早停退化);
   best_iter==0 直接报错, best_iter<30 或 valid RankIC<0.01 高亮告警
6. 数据check: 关键数据拉取失败/缓存重复键/池子过小/候选不足/流动性单位异常
   一律 RuntimeError 或高亮告警, 拒绝静默跳过
7. 基本面因子PIT: 年报Y数据仅在 [Y+1-05-01, 下一年报披露) 生效, 按日截面MAD-zscore
"""
import os
import sys
import gc
import random
import warnings
import logging

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
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158
import xgboost as xgb
import lightgbm as lgbm

logging.getLogger("qlib").setLevel(logging.ERROR)

DATA_DIR = "/Users/11164591/Documents/Qoder目录"
OUT_DIR = "/Users/11164591/Documents/Qoder目录/qlib/quant"

TOPK = 10
FEE_ROUNDTRIP = 0.004          # A股往返成本至少0.4%
LIQ_THRESHOLD = 20_000_000     # 20日均成交额 ≥ 2000万 (close*volume*100, 手→股)
N_ROUNDS, EARLY_STOP = 1000, 100
MIN_POOL, MIN_CAND = 50, 2 * TOPK
LABEL_HORIZON = 20             # 标签/前瞻窗口(交易日), 与 label Ref($close,-20) 一致

# 参与回测的策略 → 持仓数(None=全池等权). VAL* = value_comp 纯估值基准(对照系),
# 与 online_value/value_comp_M20 完全同口径, 用于回答"模型是否真能打过 value comp top20"
STRATS = {"XGB": TOPK, "LGB": TOPK, "ENS": TOPK,
          "VAL20": 20, "VAL10": 10, "POOL_EW": None}
MODEL_STRATS = ["XGB", "LGB", "ENS"]            # 需要 OOS IC 的 ML 模型
VAL_CACHE = f"{DATA_DIR}/valuation_cache.csv"
VALUE_FACTORS = ["ep", "bp", "cfp", "sp"]       # value_comp 4 个估值倒数

XGB_PARAMS = {
    "objective": "reg:squarederror", "learning_rate": 0.005,
    "max_depth": 4, "colsample_bytree": 0.8879, "subsample": 0.8789,
    "reg_alpha": 10.0, "reg_lambda": 50.0,
    "tree_method": "hist", "nthread": 4, "seed": 42,
    "disable_default_eval_metric": 1,
}
LGB_PARAMS = {
    "objective": "regression", "metric": "None",
    "learning_rate": 0.02, "max_depth": 8, "num_leaves": 128,
    "colsample_bytree": 0.8879, "subsample": 0.8789, "subsample_freq": 1,
    "lambda_l1": 50.0, "lambda_l2": 200.0,
    "num_threads": 4, "seed": 42, "verbose": -1,
}

FUND_FEATS = ["F_fcf_growth", "F_profit_growth", "F_fcf_profit_ratio",
              "F_fcf_avg_3y_norm", "F_fcf_cv_3y"]


# ==================== 窗口: 滚动5年 (4年train+1年valid) → 回测1年 ====================
def build_windows():
    wins = []
    for y in range(2020, 2027):
        bt_end = "2026-07-23" if y == 2026 else f"{y}-12-31"
        sig_end = "2026-06-30" if y == 2026 else f"{y}-11-30"
        wins.append({
            "year": y, "name": f"W{y}",
            "train": (f"{y-5}-01-01", f"{y-2}-12-31"),
            "valid": (f"{y-1}-01-01", f"{y-1}-11-30"),      # embargo: 12月留白
            "test":  (f"{y-1}-12-01", sig_end),              # 信号段(含上年12月末)
            "bt_end": bt_end,
        })
    return wins


# ==================== 十年双正 PIT 股票池 ====================
def format_qlib_code(code):
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"


def load_pit_caches():
    fcf = pd.read_csv(f"{DATA_DIR}/fcf_cache_pit.csv", sep="\t", dtype={"code": str})
    prof = pd.read_csv(f"{DATA_DIR}/profit_cache_pit.csv", sep="\t", dtype={"code": str})
    # ---- check: 缓存完整性 ----
    if len(fcf) == 0 or len(prof) == 0:
        raise RuntimeError("[CHECK] PIT缓存为空, 拒绝继续!")
    if fcf.duplicated(["code", "year"]).sum() or prof.duplicated(["code", "year"]).sum():
        raise RuntimeError("[CHECK] PIT缓存存在重复(code,year)键!")
    for df, col in [(fcf, "fcf"), (prof, "net_profit")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df["year"] = df["year"].astype(int)
    print(f"[CHECK] fcf_pit: {fcf.shape}, {fcf['code'].nunique()}只, "
          f"{fcf['year'].min()}~{fcf['year'].max()} | profit_pit: {prof.shape}, "
          f"{prof['code'].nunique()}只, {prof['year'].min()}~{prof['year'].max()}", flush=True)
    return fcf, prof


def build_dynamic_universe(backtest_year, fcf_df, profit_df):
    """Y-11..Y-2 十年窗口, FCF全正 + 净利全正(净利数据2016年起才全覆盖);
    数据覆盖≥80%年份即可入池 (既有放宽条款)"""
    fcf_years = list(range(backtest_year - 11, backtest_year - 1))
    f = fcf_df[fcf_df["year"].isin(fcf_years)].dropna(subset=["fcf"])
    fcf_pos = f.groupby("code").filter(
        lambda g: len(g) >= len(fcf_years) * 0.8 and (g["fcf"] > 0).all())
    fcf_codes = set(fcf_pos["code"])
    p_start = max(backtest_year - 11, 2016)
    p_years = list(range(p_start, backtest_year - 1))
    if p_years:
        p = profit_df[profit_df["year"].isin(p_years)].dropna(subset=["net_profit"])
        prof_pos = p.groupby("code").filter(
            lambda g: len(g) >= len(p_years) * 0.8 and (g["net_profit"] > 0).all())
        codes = fcf_codes & set(prof_pos["code"])
    else:
        codes = fcf_codes
    if len(codes) < MIN_POOL:
        raise RuntimeError(f"[CHECK] {backtest_year}年股票池仅{len(codes)}只(<{MIN_POOL}), 数据异常!")
    return sorted(codes)


# ==================== value_comp 纯估值对照 (与 online_value 逐字同口径) ====================
def load_valuation():
    """valuation_cache.csv → {ep,bp,cfp,sp} 的 date×instrument 透视表(倒数, ffill).
    与 online_value/core/valuation.load_raw_valuation 完全一致口径."""
    if not os.path.exists(VAL_CACHE):
        raise RuntimeError(f"[CHECK] {VAL_CACHE} 缺失, value_comp 对照无法计算!")
    v = pd.read_csv(VAL_CACHE, sep="\t", dtype={"code": str})
    v["date"] = pd.to_datetime(v["date"])
    for c in ["pe_ttm", "pb", "ps_ttm", "pcf"]:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v["ep"] = 1.0 / v["pe_ttm"]; v["bp"] = 1.0 / v["pb"]
    v["sp"] = 1.0 / v["ps_ttm"]; v["cfp"] = 1.0 / v["pcf"]
    v["instrument"] = v["code"].map(format_qlib_code)
    v = v.replace([np.inf, -np.inf], np.nan).drop_duplicates(["date", "instrument"], keep="last")
    piv = {c: v.pivot(index="date", columns="instrument", values=c).sort_index().ffill()
           for c in VALUE_FACTORS}
    print(f"[CHECK] 估值缓存: {piv['ep'].shape[0]}日 × {piv['ep'].shape[1]}股, "
          f"{piv['ep'].index[0].date()}~{piv['ep'].index[-1].date()}", flush=True)
    return piv


def value_comp_score(candidates, sig, val_piv):
    """信号日 value_comp: 4估值倒数在池内截面 rank 百分位均值(越大越便宜)"""
    f = pd.DataFrame(index=pd.Index(sorted(candidates), name="instrument"))
    for c in VALUE_FACTORS:
        pv = val_piv[c].loc[:sig]
        row = pv.iloc[-1] if len(pv) else pd.Series(dtype=float)
        f[c] = row.reindex(f.index)
    vc = pd.concat([f[c].rank(pct=True) for c in VALUE_FACTORS], axis=1).mean(axis=1)
    return vc


# ==================== 特征: Alpha158 + 长周期扩展 ====================
class Alpha158Enhanced(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()
        extra_fields = [
            "Ref($close, 120)/$close", "Ref($close, 240)/$close",
            "Mean($close, 120)/$close", "Mean($close, 240)/$close",
            "Std($close, 120)/$close", "Std($close, 240)/$close",
            "($close - Mean($close, 60))/(Std($close, 60)+1e-12)",
            "($close - Mean($close, 120))/(Std($close, 120)+1e-12)",
            "Mean($volume, 120)/($volume+1e-12)",
            "Mean($volume, 240)/($volume+1e-12)",
            "Corr($close, Log($volume+1), 20)",
            "Corr($close, Log($volume+1), 60)",
            "Mean($volume, 5)/(Mean($volume, 60)+1e-12)",
            "Mean($volume, 5)/(Mean($volume, 120)+1e-12)",
        ]
        extra_names = [
            "ROC120", "ROC240", "MA120", "MA240", "STD120", "STD240",
            "BOLL60", "BOLL120", "VMA120", "VMA240",
            "CORR_PV20", "CORR_PV60", "VRATIO_5_60", "VRATIO_5_120",
        ]
        return fields + extra_fields, names + extra_names


def load_fund_features(universe, fcf_df, profit_df, cal, start, end):
    """FCF系基本面因子, PIT: 年报Y → [Y+1-05-01, 下一年报披露前) 生效"""
    ucodes = {c[2:] for c in universe}
    m = fcf_df[fcf_df["code"].isin(ucodes)][["code", "year", "fcf"]].merge(
        profit_df[profit_df["code"].isin(ucodes)][["code", "year", "net_profit"]],
        on=["code", "year"], how="inner").dropna().sort_values(["code", "year"])
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    cal_win = cal[(cal >= lo) & (cal <= hi)]
    rows = []
    for code, grp in m.groupby("code"):
        qc = format_qlib_code(code)
        grp = grp.sort_values("year").copy()
        grp["fcf_growth"] = grp["fcf"].pct_change()
        grp["profit_growth"] = grp["net_profit"].pct_change()
        grp["fcf_profit_ratio"] = grp["fcf"] / (grp["net_profit"].abs() + 1e-8)
        grp["fcf_avg_3y"] = grp["fcf"].rolling(3, min_periods=1).mean()
        grp["fcf_cv_3y"] = grp["fcf"].rolling(3, min_periods=2).std() / \
            (grp["fcf"].rolling(3, min_periods=2).mean().abs() + 1e-8)
        yrs = sorted(grp["year"].unique())
        for i, (_, r) in enumerate(grp.iterrows()):
            if pd.isna(r["fcf_growth"]):
                continue
            year = int(r["year"])
            af = pd.Timestamp(f"{year+1}-05-01")
            at = pd.Timestamp(f"{yrs[i+1]+1}-04-30") if i + 1 < len(yrs) \
                else pd.Timestamp(f"{year+2}-04-30")
            for d in cal_win[(cal_win >= af) & (cal_win <= at)]:
                rows.append((d, qc, r["fcf_growth"], r["profit_growth"],
                             r["fcf_profit_ratio"], r["fcf_avg_3y"] / 1e8, r["fcf_cv_3y"]))
    if not rows:
        raise RuntimeError("[CHECK] 基本面PIT特征为空, 拒绝静默跳过!")
    fdf = pd.DataFrame(rows, columns=["datetime", "instrument"] + FUND_FEATS)
    fdf = fdf.set_index(["datetime", "instrument"])
    fdf = fdf[~fdf.index.duplicated(keep="last")]
    # 按日截面 MAD-zscore (避免全局标准化时间泄露)
    for col in fdf.columns:
        g = fdf[col].replace([np.inf, -np.inf], np.nan).groupby(level=0)
        med = g.transform("median")
        mad = g.transform(lambda x: (x - x.median()).abs().median()).replace(0, np.nan)
        fdf[col] = ((fdf[col] - med) / (1.4826 * mad)).clip(-3, 3)
    return fdf.astype(np.float32)


# ==================== RankIC 早停 (预计算分组, 每轮boosting高速评估) ====================
class RankICEval:
    def __init__(self, dates, labels):
        dates = np.asarray(dates)
        order = np.argsort(dates, kind="stable")
        self.order = order
        d_sorted = dates[order]
        _, starts = np.unique(d_sorted, return_index=True)
        bounds = np.append(starts, len(d_sorted))
        self.slices = [(bounds[i], bounds[i + 1]) for i in range(len(starts))
                       if bounds[i + 1] - bounds[i] > 5]
        y_sorted = np.asarray(labels)[order]
        self.y_ranks = []
        for lo, hi in self.slices:
            seg = y_sorted[lo:hi]
            r = np.empty(len(seg))
            r[np.argsort(seg, kind="stable")] = np.arange(len(seg))
            self.y_ranks.append(r)

    def __call__(self, preds):
        p_sorted = np.asarray(preds)[self.order]
        ics = []
        for (lo, hi), yr in zip(self.slices, self.y_ranks):
            seg = p_sorted[lo:hi]
            pr = np.empty(len(seg))
            pr[np.argsort(seg, kind="stable")] = np.arange(len(seg))
            ics.append(np.corrcoef(pr, yr)[0, 1])
        return float(np.nanmean(ics))


def train_xgb(X_tr, y_tr, X_va, y_va, ic_eval):
    dtr = xgb.DMatrix(X_tr.values, label=y_tr.values)
    dva = xgb.DMatrix(X_va.values, label=y_va.values)

    def feval(predt, dmat):
        return "rank_ic", ic_eval(predt)

    def _fit(params):
        m = xgb.train(params, dtr, num_boost_round=N_ROUNDS,
                      evals=[(dva, "valid")], custom_metric=feval, maximize=True,
                      early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        return m, m.best_iteration, float(m.best_score)

    m, bi, ic = _fit(XGB_PARAMS)
    # 收敛不足 fallback: 极小 best_iter 说明该窗口下超参不匹配(早停撞上噪声峰值),
    # 用备选超参重训, 按 valid RankIC 择优(仍只用 valid 集, 不碰 test, 无穿越)
    if bi < 30:
        for alt in ({"learning_rate": 0.02, "max_depth": 6, "reg_lambda": 20.0},
                    {"learning_rate": 0.01, "max_depth": 5, "reg_alpha": 3.0}):
            p2 = {**XGB_PARAMS, **alt}
            m2, bi2, ic2 = _fit(p2)
            print(f"    [XGB-fallback] {alt} → best_iter={bi2}, RankIC={ic2:.4f}", flush=True)
            if ic2 > ic and bi2 >= 30:
                m, bi, ic = m2, bi2, ic2
                break
            if ic2 > ic:
                m, bi, ic = m2, bi2, ic2
    return m, bi, ic


def train_lgb(X_tr, y_tr, X_va, y_va, ic_eval):
    tr = lgbm.Dataset(X_tr.values, label=y_tr.values)
    va = lgbm.Dataset(X_va.values, label=y_va.values, reference=tr)

    def feval(predt, ds):
        return "rank_ic", ic_eval(predt), True

    def _fit(params):
        m = lgbm.train(params, tr, num_boost_round=N_ROUNDS,
                       valid_sets=[va], valid_names=["valid"], feval=feval,
                       callbacks=[lgbm.early_stopping(EARLY_STOP, first_metric_only=True,
                                                      verbose=False)])
        return m, m.best_iteration, float(m.best_score["valid"]["rank_ic"])

    m, bi, ic = _fit(LGB_PARAMS)
    # 收敛不足 fallback: 同 XGB, 用更慢学习率/更浅树重训, 按 valid RankIC 择优(无穿越)
    if bi < 30:
        for alt in ({"learning_rate": 0.01, "max_depth": 6, "num_leaves": 48},
                    {"learning_rate": 0.005, "max_depth": 5, "num_leaves": 32}):
            p2 = {**LGB_PARAMS, **alt}
            m2, bi2, ic2 = _fit(p2)
            print(f"    [LGB-fallback] {alt} → best_iter={bi2}, RankIC={ic2:.4f}", flush=True)
            if ic2 > ic and bi2 >= 30:
                m, bi, ic = m2, bi2, ic2
                break
            if ic2 > ic:
                m, bi, ic = m2, bi2, ic2
    return m, bi, ic


def check_convergence(tag, best_iter, valid_ic):
    if best_iter is None or best_iter <= 0:
        raise RuntimeError(f"[CHECK] {tag} best_iter={best_iter}, 模型退化为常数输出!")
    warn = []
    if best_iter < 30:
        warn.append(f"best_iter={best_iter}<30 收敛轮数偏少")
    if valid_ic < 0.01:
        warn.append(f"valid RankIC={valid_ic:.4f}<0.01 预测力弱")
    msg = " | ".join(warn) if warn else "OK"
    print(f"    [{tag}] best_iter={best_iter}, valid RankIC={valid_ic:.4f} → {msg}", flush=True)


# ==================== 可交易性 / 流动性 ====================
def build_tradability(universe, start, end):
    px = D.features(list(universe), ["$close", "$open", "$high", "$low", "$volume"],
                    start_time=pd.Timestamp(start) - pd.Timedelta(days=60), end_time=end)
    if px is None or len(px) == 0:
        raise RuntimeError(f"[CHECK] 可交易性数据拉取失败 ({start}~{end}), 拒绝静默跳过!")
    px = px.reset_index()
    px.columns = ["instrument", "datetime", "close", "open", "high", "low", "volume"]
    px = px.sort_values(["instrument", "datetime"])
    px["prev_close"] = px.groupby("instrument")["close"].shift(1)
    ret = (px["close"] - px["prev_close"]) / px["prev_close"]
    one_line = (px["open"] == px["high"]) & (px["high"] == px["low"]) & \
               (px["low"] == px["close"]) & (ret > 0.09)
    limit_up = set(zip(px.loc[one_line, "datetime"], px.loc[one_line, "instrument"]))
    susp = px[(px["volume"].isna()) | (px["volume"] == 0)]
    suspension = set(zip(susp["datetime"], susp["instrument"]))
    # 流动性: $volume单位为手 → 成交额 = close*volume*100
    px["amount"] = px["close"] * px["volume"] * 100
    px["avg20"] = px.groupby("instrument")["amount"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    liq = px.set_index(["datetime", "instrument"])["avg20"]
    # ---- check: 单位自检, 池内中位数成交额应在合理量级 (1e6~1e11) ----
    med = liq.dropna().median()
    if not (1e6 < med < 1e11):
        raise RuntimeError(f"[CHECK] 流动性单位异常: 池内20日均成交额中位数={med:.3g}元!")
    print(f"    [流动性] 中位数20日均成交额={med/1e8:.2f}亿, 一字涨停{len(limit_up)}条, "
          f"停牌{len(suspension)}条", flush=True)
    return limit_up, suspension, liq


def get_month_end_dates(cal, start, end):
    dates = [d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    out = []
    for i, d in enumerate(dates):
        if i + 1 == len(dates) or dates[i + 1].month != d.month:
            out.append(d)
    return out


# ==================== 组合回测引擎 (T+1, 全期连续NAV) ====================
def portfolio_backtest(rebalances, price_mat):
    """rebalances: [(exec_dt, [等权持仓]), ...] 时间升序; 成本=单边换手×0.4%.
    返回 (日收益, 月均单边换手, 总买入笔数)"""
    rebalances = sorted(rebalances, key=lambda x: x[0])
    dates = [d for d in price_mat.index if d >= rebalances[0][0]]
    shares, value = {}, 1.0
    navs, ds, turnovers = [], [], []
    n_buys = 0
    ri = 0
    for d in dates:
        px = price_mat.loc[d]
        if shares:
            value = sum(sh * px[i] for i, sh in shares.items() if pd.notna(px.get(i)))
        while ri < len(rebalances) and rebalances[ri][0] == d:
            targets = [t for t in rebalances[ri][1]
                       if pd.notna(px.get(t)) and px.get(t, 0) > 0]
            if targets:
                n_buys += len(set(targets) - set(shares.keys()))   # 新建仓笔数
                w_tgt = {t: 1.0 / len(targets) for t in targets}
                w_cur = {i: sh * px[i] / value for i, sh in shares.items()
                         if pd.notna(px.get(i))} if shares and value > 0 else {}
                to = 0.5 * sum(abs(w_tgt.get(i, 0) - w_cur.get(i, 0))
                               for i in set(w_tgt) | set(w_cur))
                value -= value * to * FEE_ROUNDTRIP
                shares = {t: w_tgt[t] * value / px[t] for t in targets}
                turnovers.append(to)
            ri += 1
        navs.append(value)
        ds.append(d)
    nav = pd.Series(navs, index=pd.DatetimeIndex(ds))
    return (nav.pct_change().fillna(0),
            (np.mean(turnovers) if turnovers else 0.0), n_buys)


def calc_metrics(returns, bench=None):
    if len(returns) == 0:
        return {}
    n_years = len(returns) / 244
    ar = (1 + returns).prod() ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(244)
    nav = (1 + returns).cumprod()
    mdd = ((nav / nav.cummax()) - 1).min()
    out = {"ar": ar, "vol": vol, "sharpe": ar / vol if vol > 0 else 0,
           "mdd": mdd, "calmar": ar / abs(mdd) if mdd < 0 else float("nan"),
           "total": nav.iloc[-1] - 1}
    if bench is not None:
        b = bench.reindex(returns.index).fillna(0)
        ex = (1 + returns).prod() ** (1 / n_years) - (1 + b).prod() ** (1 / n_years)
        out["excess_ar"] = ex
    return out


def annualized_since(returns, start):
    """从 start 起(含)子区间的年化收益, 用于对齐 value_comp 基线(跳过2020)口径"""
    sub = returns[returns.index >= pd.Timestamp(start)]
    if len(sub) < 60:
        return float("nan")
    return (1 + sub).prod() ** (244 / len(sub)) - 1


# ==================== 单窗口: 训练 + 生成月度调仓指令 ====================
def run_window(win, fcf_df, profit_df, cal, results, rebalances, holdings_log,
               val_piv, ic_records):
    y = win["year"]
    ts, te = win["train"]
    vs, ve = win["valid"]
    xs, xe = win["test"]
    print(f"\n{'='*70}\n[{win['name']}] 池:{y-11}~{y-2}十年双正 | train {ts}~{te} | "
          f"valid {vs}~{ve}(embargo) | 信号 {xs}~{xe}\n{'='*70}", flush=True)

    codes = build_dynamic_universe(y, fcf_df, profit_df)
    universe = [format_qlib_code(c) for c in codes]
    print(f"  股票池: {len(universe)} 只", flush=True)

    dhc = {"start_time": ts, "end_time": xe, "fit_start_time": ts, "fit_end_time": te,
           "instruments": universe,
           "infer_processors": [
               {"class": "RobustZScoreNorm",
                "kwargs": {"fields_group": "feature", "clip_outlier": True}},
               {"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
           "learn_processors": [
               {"class": "DropnaLabel"},
               {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}],
           "label": ["Ref($close, -20) / $close - 1"]}
    dsc = {"class": "DatasetH", "module_path": "qlib.data.dataset",
           "kwargs": {"handler": {"class": "Alpha158Enhanced",
                                  "module_path": "__main__", "kwargs": dhc},
                      "segments": {"train": (ts, te), "valid": (vs, ve),
                                   "test": (xs, xe)}}}
    dataset = init_instance_by_config(dsc)
    train_df = dataset.prepare("train", col_set=["feature", "label"],
                               data_key=DataHandlerLP.DK_L)
    valid_df = dataset.prepare("valid", col_set=["feature", "label"],
                               data_key=DataHandlerLP.DK_L)
    test_X = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    del dataset
    gc.collect()

    # ---- check: 行情覆盖率 ----
    n_inst = train_df.index.get_level_values(1).nunique()
    cov = n_inst / len(universe)
    print(f"  [CHECK] 行情覆盖: {n_inst}/{len(universe)} ({cov*100:.0f}%)", flush=True)
    if cov < 0.6:
        raise RuntimeError(f"[CHECK] {y}年行情覆盖率仅{cov*100:.0f}%, qlib数据异常!")

    X_tr, y_tr = train_df["feature"], train_df["label"].iloc[:, 0]
    X_va, y_va = valid_df["feature"], valid_df["label"].iloc[:, 0]
    mask = y_va.notna()
    X_va, y_va = X_va[mask], y_va[mask]
    if len(X_tr) < 10000 or len(X_va) < 1000:
        raise RuntimeError(f"[CHECK] 样本量异常: train={len(X_tr)}, valid={len(X_va)}!")

    # 基本面PIT因子注入 (train/valid/test 统一列)
    fund = load_fund_features(universe, fcf_df, profit_df, cal, ts, xe)
    X_tr = pd.concat([X_tr, fund.reindex(X_tr.index).fillna(0)], axis=1).astype(np.float32)
    X_va = pd.concat([X_va, fund.reindex(X_va.index).fillna(0)], axis=1).astype(np.float32)
    test_X = pd.concat([test_X, fund.reindex(test_X.index).fillna(0)], axis=1).astype(np.float32)
    nan_pct = X_tr.isna().values.mean() * 100
    print(f"  特征矩阵: train {X_tr.shape}, valid {X_va.shape}, test {test_X.shape}, "
          f"NaN={nan_pct:.2f}%", flush=True)
    if nan_pct > 5:
        raise RuntimeError(f"[CHECK] 特征NaN比例{nan_pct:.1f}%过高!")
    del train_df, valid_df, fund
    gc.collect()

    va_dates = X_va.index.get_level_values(0).values
    ic_eval = RankICEval(va_dates, y_va.values)

    models = {}
    for tag, trainer in [("XGB", train_xgb), ("LGB", train_lgb)]:
        m, best_iter, valid_ic = trainer(X_tr, y_tr, X_va, y_va, ic_eval)
        check_convergence(f"{win['name']}-{tag}", best_iter, valid_ic)
        models[tag] = m
        results.append({"window": win["name"], "year": y, "model": tag,
                        "pool_size": len(universe), "n_inst": n_inst,
                        "best_iter": best_iter, "valid_rank_ic": round(valid_ic, 4)})

    # 测试段预测
    preds = {}
    preds["XGB"] = pd.Series(models["XGB"].predict(xgb.DMatrix(test_X.values)),
                             index=test_X.index)
    preds["LGB"] = pd.Series(models["LGB"].predict(
        test_X.values, num_iteration=models["LGB"].best_iteration), index=test_X.index)
    del X_tr, X_va, y_tr, y_va, test_X, models
    gc.collect()

    # 可交易性与流动性 (仅测试段)
    limit_up, susp, liq = build_tradability(universe, xs, xe)

    # ---- OOS IC 用: 信号段 20日前瞻真实收益矩阵 (date × instrument) ----
    icpx = D.features(universe, ["$close"],
                      start_time=pd.Timestamp(xs) - pd.Timedelta(days=10),
                      end_time=pd.Timestamp(xe) + pd.Timedelta(days=60))
    icclose = icpx["$close"].unstack(level=0).sort_index()
    fwd_mat = icclose.shift(-LABEL_HORIZON) / icclose - 1     # 20交易日前瞻收益

    sig_dates = get_month_end_dates(cal, xs, xe)
    cal_idx = pd.DatetimeIndex(cal)
    for sig_dt in sig_dates:
        for tag in ["XGB", "LGB"]:
            if sig_dt not in preds[tag].index.get_level_values(0):
                raise RuntimeError(f"[CHECK] {sig_dt.date()} 无{tag}预测截面!")
        # 可交易候选集 (过滤 model-agnostic: 一字涨停/停牌/流动性不足, 缺字段即剔除)
        base_idx = preds["XGB"].xs(sig_dt, level=0).index
        cand = []
        for inst in base_idx:
            if (sig_dt, inst) in limit_up or (sig_dt, inst) in susp:
                continue
            a = liq.get((sig_dt, inst), np.nan)
            if pd.isna(a) or a < LIQ_THRESHOLD:
                continue
            cand.append(inst)
        cand = pd.Index(cand)
        if len(cand) < TOPK:
            raise RuntimeError(f"[CHECK] {sig_dt.date()} 可交易候选仅{len(cand)}只(<{TOPK})!")
        if len(cand) < MIN_CAND:
            print(f"    [WARN] {sig_dt.date()} 可交易候选{len(cand)}只偏少", flush=True)

        # 各策略在同一候选集上的打分
        score = {}
        score["XGB"] = preds["XGB"].xs(sig_dt, level=0).reindex(cand)
        score["LGB"] = preds["LGB"].xs(sig_dt, level=0).reindex(cand)
        score["ENS"] = (score["XGB"].rank(pct=True) + score["LGB"].rank(pct=True)) / 2
        score["VALUE"] = value_comp_score(cand, sig_dt, val_piv).reindex(cand)

        # OOS IC: 每模型 该信号日截面 (score vs 20日前瞻真实收益) 的 Pearson/Spearman
        fwd = fwd_mat.loc[sig_dt].reindex(cand) if sig_dt in fwd_mat.index else pd.Series(np.nan, index=cand)
        for mdl in MODEL_STRATS + ["VALUE"]:
            df = pd.concat([score[mdl], fwd], axis=1).dropna()
            if len(df) >= 10:
                ic_records.append({"window": win["name"], "model": mdl,
                                   "sig_date": sig_dt,
                                   "ic": df.iloc[:, 0].corr(df.iloc[:, 1]),
                                   "rank_ic": df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")})

        # T+1: 次一交易日执行
        pos = int(cal_idx.searchsorted(sig_dt)) + 1
        if pos >= len(cal_idx):
            continue
        exec_dt = cal_idx[pos]
        vc_valid = score["VALUE"].dropna()
        picks = {
            "XGB": score["XGB"].nlargest(TOPK).index.tolist(),
            "LGB": score["LGB"].nlargest(TOPK).index.tolist(),
            "ENS": score["ENS"].nlargest(TOPK).index.tolist(),
            "VAL20": vc_valid.nlargest(20).index.tolist(),
            "VAL10": vc_valid.nlargest(10).index.tolist(),
            "POOL_EW": cand.tolist(),
        }
        for tag, top in picks.items():
            if not top:
                raise RuntimeError(f"[CHECK] {sig_dt.date()} {tag} 无可买标的!")
            rebalances[tag].append((exec_dt, top))
            if tag in ("XGB", "LGB", "ENS", "VAL20"):
                holdings_log.append({"signal_date": sig_dt.date(), "exec_date": exec_dt.date(),
                                     "model": tag, "holdings": ",".join(top)})
    del preds
    gc.collect()


# ==================== 主流程 ====================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # 使用修补后的数据副本(fix_qlib_seam.py 已消除 2020-09-28 两批拼接断点)
    fixed_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data_fixed")
    if not os.path.exists(f"{fixed_uri}/features"):
        raise RuntimeError("[CHECK] cn_data_fixed 不存在, 请先运行 fix_qlib_seam.py!")
    qlib.init(provider_uri=fixed_uri, region=REG_CN)
    fcf_df, profit_df = load_pit_caches()
    cal = D.calendar(start_time="2014-01-01", end_time="2026-07-23")
    print(f"[CHECK] 交易日历: {cal[0].date()} ~ {cal[-1].date()} ({len(cal)}天)", flush=True)
    if cal[-1] < pd.Timestamp("2026-07-01"):
        raise RuntimeError("[CHECK] qlib日历未覆盖2026H1!")

    windows = build_windows()
    results, holdings_log, ic_records = [], [], []
    rebalances = {s: [] for s in STRATS}
    val_piv = load_valuation()
    for win in windows:
        run_window(win, fcf_df, profit_df, cal, results, rebalances, holdings_log,
                   val_piv, ic_records)

    # ---- 全期连续回测 (跨窗口换手自然衔接) ----
    all_insts = sorted({i for tag in rebalances for _, tops in rebalances[tag]
                        for i in tops})
    px = D.features(all_insts, ["$close"], start_time="2019-11-01",
                    end_time="2026-07-23")
    if px is None or len(px) == 0:
        raise RuntimeError("[CHECK] 回测价格矩阵拉取失败!")
    px = px.reset_index()
    px.columns = ["instrument", "datetime", "close"]
    price_mat = px.pivot(index="datetime", columns="instrument",
                         values="close").sort_index().ffill()

    # 基准: qlib 内 SH000300 指数数据止于 2020-09-25(断点后全0), 改用外部干净缓存
    bcsv = "/Users/11164591/Documents/Qoder目录/csi300_cache.csv"
    if not os.path.exists(bcsv):
        raise RuntimeError("[CHECK] csi300_cache.csv 不存在, 沪深300基准缺失!")
    braw = pd.read_csv(bcsv, sep="\t")
    braw.columns = ["date", "close"]
    braw["date"] = pd.to_datetime(braw["date"])
    bclose = braw.set_index("date")["close"].astype(float).sort_index()
    if bclose.index[0] > pd.Timestamp("2019-12-01") or bclose.index[-1] < pd.Timestamp("2026-07-01"):
        raise RuntimeError(f"[CHECK] csi300_cache 覆盖不足: {bclose.index[0]}~{bclose.index[-1]}")
    if bclose.pct_change().abs().max() > 0.12:
        raise RuntimeError("[CHECK] csi300_cache 存在异常日收益(>12%), 数据可疑!")
    bench = bclose.pct_change().fillna(0)
    print(f"[CHECK] 沪深300基准(csi300_cache): {bclose.index[0].date()} ~ "
          f"{bclose.index[-1].date()} ({len(bclose)}天)", flush=True)

    # ---- OOS 信号期 IC / RankIC 聚合 (逐信号日截面, 跨窗口取均值) ----
    ic_df = pd.DataFrame(ic_records)
    ic_agg = ic_df.groupby("model")[["ic", "rank_ic"]].mean().to_dict("index") if len(ic_df) else {}

    def ic_of(strat):
        key = "VALUE" if strat in ("VAL10", "VAL20") else strat
        d = ic_agg.get(key)
        return (d["ic"], d["rank_ic"]) if d else (float("nan"), float("nan"))

    BT_START = pd.Timestamp("2020-01-01")
    ALIGN_START = pd.Timestamp("2021-01-01")     # 对齐 value_comp 冻结基线(跳过2020异常年)
    LABEL = {"XGB": "XGB Top10", "LGB": "LGB Top10", "ENS": "集成ENS Top10",
             "VAL20": "value_comp Top20", "VAL10": "value_comp Top10", "POOL_EW": "十年双正池等权"}

    # ---- 逐策略回测 + 全指标 ----
    nav_out, metric_rows, yearly_all = {}, [], {}
    for tag in STRATS:
        rets, avg_to, n_buys = portfolio_backtest(rebalances[tag], price_mat)
        rets = rets[rets.index >= BT_START]
        m = calc_metrics(rets, bench)
        ic, rankic = ic_of(tag)
        nav_out[tag] = (1 + rets).cumprod()
        metric_rows.append({
            "策略": LABEL[tag], "年化收益": m["ar"], "年化(剔2020)": annualized_since(rets, ALIGN_START),
            "年化波动": m["vol"], "Sharpe": m["sharpe"], "最大回撤": m["mdd"],
            "Calmar": m["calmar"], "IC": ic, "RankIC": rankic,
            "换手率": avg_to, "交易次数": n_buys, "超额vsHS300": m.get("excess_ar", float("nan"))})
        yearly_all[tag] = {yr: (1 + g).prod() - 1 for yr, g in rets.groupby(rets.index.year)}

    bench_bt = bench[bench.index >= BT_START]
    mb = calc_metrics(bench_bt)
    yearly_all["HS300"] = {yr: (1 + g).prod() - 1 for yr, g in bench_bt.groupby(bench_bt.index.year)}
    metric_rows.append({
        "策略": "沪深300基准", "年化收益": mb["ar"], "年化(剔2020)": annualized_since(bench_bt, ALIGN_START),
        "年化波动": mb["vol"], "Sharpe": mb["sharpe"], "最大回撤": mb["mdd"],
        "Calmar": mb["calmar"], "IC": float("nan"), "RankIC": float("nan"),
        "换手率": 0.0, "交易次数": 0, "超额vsHS300": 0.0})

    # ==================== 标准输出 ====================
    def fp(x):   # 百分比
        return "  n/a  " if pd.isna(x) else f"{x*100:+7.2f}%"
    def fpp(x):
        return "  n/a  " if pd.isna(x) else f"{x*100:7.2f}%"
    def fn(x, d=3):
        return " n/a " if pd.isna(x) else f"{x:.{d}f}"

    years = sorted({yr for m in yearly_all.values() for yr in m})
    print(f"\n{'='*96}\n  一、分年收益 (2020~2026H1, 月频, T+1, 往返成本0.4%)\n{'='*96}", flush=True)
    hdr = "策略".ljust(20) + "".join(f"{yr:>9}" for yr in years)
    print(hdr, flush=True)
    order = list(STRATS) + ["HS300"]
    for tag in order:
        lab = LABEL.get(tag, "沪深300基准")
        line = lab.ljust(18) + "".join(f"{yearly_all[tag].get(yr, float('nan'))*100:>8.1f}%"
                                        if yr in yearly_all[tag] else f"{'-':>9}" for yr in years)
        print(line, flush=True)

    print(f"\n{'='*96}\n  二、整体指标 (图1口径; 年化(剔2020)用于对齐 value_comp 冻结基线)\n{'='*96}", flush=True)
    mdf = pd.DataFrame(metric_rows)
    disp = pd.DataFrame({
        "策略": mdf["策略"],
        "年化收益": mdf["年化收益"].map(fpp),
        "年化(剔20)": mdf["年化(剔2020)"].map(fpp),
        "年化波动": mdf["年化波动"].map(fpp),
        "Sharpe": mdf["Sharpe"].map(lambda x: fn(x, 2)),
        "最大回撤": mdf["最大回撤"].map(fpp),
        "Calmar": mdf["Calmar"].map(lambda x: fn(x, 2)),
        "IC": mdf["IC"].map(lambda x: fn(x, 4)),
        "RankIC": mdf["RankIC"].map(lambda x: fn(x, 4)),
        "换手率": mdf["换手率"].map(fpp),
        "交易次数": mdf["交易次数"],
    })
    print(disp.to_string(index=False), flush=True)

    # ---- 关键对比结论 ----
    ens = mdf.set_index("策略").loc["集成ENS Top10"]
    v20 = mdf.set_index("策略").loc["value_comp Top20"]
    print(f"\n{'='*96}\n  三、模型 vs value_comp 对照 (回答'模型是否真能打过 value comp top20')\n{'='*96}", flush=True)
    print(f"  · 全期(含2020)  : ENS年化 {ens['年化收益']*100:.2f}%  vs  value_comp20 {v20['年化收益']*100:.2f}%", flush=True)
    print(f"  · 对齐(剔2020)  : ENS年化 {ens['年化(剔2020)']*100:.2f}%  vs  value_comp20 {v20['年化(剔2020)']*100:.2f}%"
          f"   ← 与冻结基线(25.75%)同口径", flush=True)
    print(f"  · 风险调整      : ENS Sharpe {ens['Sharpe']:.2f}/Calmar {ens['Calmar']:.2f}  vs  "
          f"value_comp20 Sharpe {v20['Sharpe']:.2f}/Calmar {v20['Calmar']:.2f}", flush=True)
    print(f"  · ENS 是 Top10(集中度2倍于Top20); OOS RankIC ENS={ens['RankIC']:.4f} value={v20['RankIC']:.4f}", flush=True)

    # ---- 保存 ----
    pd.DataFrame(results).to_csv(f"{OUT_DIR}/rolling10y_train_diag.csv", sep="\t", index=False)
    mdf.to_csv(f"{OUT_DIR}/rolling10y_summary.csv", sep="\t", index=False)
    pd.DataFrame(holdings_log).to_csv(f"{OUT_DIR}/rolling10y_holdings.csv", sep="\t", index=False)
    yearly_df = pd.DataFrame(yearly_all).T
    yearly_df.to_csv(f"{OUT_DIR}/rolling10y_yearly.csv", sep="\t")
    if len(ic_df):
        ic_df.to_csv(f"{OUT_DIR}/rolling10y_ic.csv", sep="\t", index=False)
    nav_df = pd.DataFrame(nav_out)
    nav_df["HS300"] = (1 + bench_bt).cumprod().reindex(nav_df.index)
    nav_df.to_csv(f"{OUT_DIR}/rolling10y_nav.csv", sep="\t")
    print(f"\n[+] 结果已保存: rolling10y_summary.csv / _yearly.csv / _ic.csv / "
          f"_train_diag.csv / _holdings.csv / _nav.csv", flush=True)


if __name__ == "__main__":
    main()
