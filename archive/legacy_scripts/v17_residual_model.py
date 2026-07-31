"""
V17 残差增强模型: value_comp 基本盘 + ML残差(催化剂) 融合
=================================================================
思路(用户提供, 已对齐时间轴):
  标签 y_res: 每换仓点, [未来60天超额收益] 对 value_comp 截面回归取残差
             → 模型只学"便宜之外"的催化剂, 与估值正交
  特征 X    : 全部在换仓点之前 (量价反转/动量/换手/波动 + 边际基本面SUE + 日历)
             严格剔除估值因子(避免与value_comp共线)
  模型      : XGB rank:pairwise, 浅树(depth2-3), 高min_child_weight,
             colsample 0.4, RankIC验证集早停(修掉RMSE早停退化的坑)
  融合      : Final = a*rank(value_comp) + (1-a)*rank(y_hat_res), a=0.5/0.7/0.85
  验收      : 同口径回测, OOS(2023-26)净超越 value_comp Top20 的 19.98%
时间轴锁定: 特征全在过去, 标签全在未来, 不交叉。训练用历史配对, 预测用当日X。
"""
import os, sys, random
import warnings, logging
def set_seed(seed=42):
    random.seed(seed); import numpy as np; np.random.seed(seed)
set_seed(42)
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")
import numpy as np
import pandas as pd
import qlib
import xgboost as xgb
from qlib.constant import REG_CN
from qlib.data import D
warnings.filterwarnings("ignore")
logging.getLogger('qlib.data.data').setLevel(logging.ERROR)

from v5_validation import build_limit_up_set
from v5_xgb_turnover_comparative import build_dynamic_universe, format_qlib_code
from v14_ablation import (build_liquidity_table, portfolio_backtest, calc_metrics,
                          quarter_signal_exec_dates, load_finind_table,
                          WORK_DIR, DATA_DIR)
from v16_value_rules import load_raw_valuation, LIQ_THRESHOLD
from v15_model_fixed import get_bench_returns_ak

YEARS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
IS_YEARS = {2019, 2021, 2022}
OOS_YEARS = {2023, 2024, 2025, 2026}
ALPHAS = [0.5, 0.7, 0.85]
TOPK = 20
FWD_DAYS = 60   # 标签: 未来60天(约一季)超额

# 催化剂特征(全部基于换仓点之前的信息, 与估值正交)
FEATURES = ["mom_60", "mom_120", "rev_5", "rev_20", "turn_z", "turn_slope",
            "ivol_20", "beta_60", "profit_accel", "rev_accel", "roe_q_chg", "month"]

# 浅树 rank 模型
XGB_RANK_PARAMS = dict(
    objective="rank:pairwise", eta=0.03, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.4, reg_alpha=1.0, reg_lambda=5.0,
    disable_default_eval_metric=1,
)
N_ROUNDS, EARLY_STOP = 400, 40
IC_DEGEN_TH = 0.005


def load_price_volume(insts, start, end):
    """价格+成交量矩阵, 用于量价特征。返回 (close_mat, volume_mat) 均已ffill"""
    px = D.features(list(insts), ["$close", "$volume"], start_time=start, end_time=end)
    if px is None or len(px) == 0:
        raise RuntimeError(f"[价量] 拉取失败 ({start}~{end})!")
    px = px.reset_index()
    px.columns = ["instrument", "datetime", "close", "volume"]
    close = px.pivot(index="datetime", columns="instrument", values="close").sort_index().ffill()
    vol = px.pivot(index="datetime", columns="instrument", values="volume").sort_index()
    return close, vol


def load_index_close():
    """沪深300收盘, 用于beta与大盘均线位置"""
    df = D.features(["SH000300"], ["$close"], start_time="2013-01-01", end_time="2026-07-31")
    if df is None or len(df) == 0:
        return None
    s = df.reset_index().set_index("datetime")["$close"].sort_index()
    return s


def compute_value_comp(universe, sig, val_piv):
    """信号日 value_comp: 4估值倒数的池内截面rank百分位均值 (与v16一致)"""
    f = pd.DataFrame(index=pd.Index(sorted(universe), name="instrument"))
    for c in ["ep", "bp", "cfp", "sp"]:
        pv = val_piv[c]
        row = pv.loc[:sig].iloc[-1] if len(pv.loc[:sig]) else pd.Series(dtype=float)
        f[c] = row.reindex(f.index)
    rk = lambda s: s.rank(pct=True)
    vc = pd.concat([rk(f[c]) for c in ["ep", "bp", "cfp", "sp"]], axis=1).mean(axis=1)
    return vc


def compute_catalyst_features(universe, sig, exec_dt, close, vol, idx_close, fin):
    """换仓点之前的催化剂特征 (量价+边际基本面+日历), index=instrument"""
    f = pd.DataFrame(index=pd.Index(sorted(universe), name="instrument"))
    px = close.loc[:sig]
    if len(px) > 130:
        p = px.iloc[-1]
        f["mom_60"] = (px.iloc[-1] / px.iloc[-61] - 1).reindex(f.index)      # 中期动量
        f["mom_120"] = (px.iloc[-1] / px.iloc[-121] - 1).reindex(f.index)
        f["rev_5"] = -(px.iloc[-1] / px.iloc[-6] - 1).reindex(f.index)       # 短期反转
        f["rev_20"] = -(px.iloc[-1] / px.iloc[-21] - 1).reindex(f.index)
        rets = px.pct_change()
        f["ivol_20"] = rets.iloc[-20:].std().reindex(f.index)               # 特质波动(近似)
        # beta_60: 个股对大盘的敏感度
        if idx_close is not None:
            ir = idx_close.reindex(px.index).pct_change()
            win = rets.iloc[-60:]; iw = ir.iloc[-60:]
            cov = win.apply(lambda col: col.cov(iw))
            f["beta_60"] = (cov / iw.var()).reindex(f.index)
        else:
            f["beta_60"] = np.nan
    else:
        for c in ["mom_60", "mom_120", "rev_5", "rev_20", "ivol_20", "beta_60"]:
            f[c] = np.nan
    # 换手率趋势 (成交量zscore与斜率, 短期流通股本近似不变)
    vv = vol.loc[:sig]
    if len(vv) > 40:
        v20 = vv.iloc[-20:]
        f["turn_z"] = ((vv.iloc[-1] - v20.mean()) / v20.std()).reindex(f.index)
        x = np.arange(20)
        slope = v20.apply(lambda col: np.polyfit(x, col.values, 1)[0] / (col.mean() + 1e-9)
                          if col.notna().sum() >= 10 else np.nan)
        f["turn_slope"] = slope.reindex(f.index)
    else:
        f["turn_z"] = np.nan; f["turn_slope"] = np.nan
    # 边际基本面: 用PIT可用的财务(avail<=sig), 单季度加速度
    f = _add_fundamental_accel(f, universe, sig, fin)
    # 日历
    f["month"] = exec_dt.month
    return f


def _add_fundamental_accel(f, universe, sig, fin):
    """业绩加速度: profit_growth/rev_growth 环比变化, 单季ROE变化。只用avail<=sig的财报"""
    sub = fin[(fin["avail"] <= sig) & (fin["instrument"].isin(universe))].copy()
    sub = sub.sort_values(["instrument", "avail"])
    prof_a, rev_a, roe_c = {}, {}, {}
    for inst, g in sub.groupby("instrument"):
        g = g.tail(3)
        pg = g["profit_growth"].values; rg = g["rev_growth"].values; roe = g["roe"].values
        prof_a[inst] = (pg[-1] - pg[-2]) if len(pg) >= 2 and np.isfinite(pg[-1]) and np.isfinite(pg[-2]) else np.nan
        rev_a[inst] = (rg[-1] - rg[-2]) if len(rg) >= 2 and np.isfinite(rg[-1]) and np.isfinite(rg[-2]) else np.nan
        roe_c[inst] = (roe[-1] - roe[-2]) if len(roe) >= 2 and np.isfinite(roe[-1]) and np.isfinite(roe[-2]) else np.nan
    f["profit_accel"] = pd.Series(prof_a).reindex(f.index)
    f["rev_accel"] = pd.Series(rev_a).reindex(f.index)
    f["roe_q_chg"] = pd.Series(roe_c).reindex(f.index)
    return f


def residual_label(fwd_excess, value_comp):
    """标签: 未来超额 对 value_comp 截面回归取残差 (剥离便宜因素)"""
    df = pd.concat([fwd_excess.rename("y"), value_comp.rename("vc")], axis=1).dropna()
    if len(df) < 10:
        return pd.Series(np.nan, index=fwd_excess.index)
    x = df["vc"].values; y = df["y"].values
    b1 = np.cov(x, y)[0, 1] / (np.var(x) + 1e-12)
    b0 = y.mean() - b1 * x.mean()
    resid = df["y"] - (b0 + b1 * df["vc"])
    return resid.reindex(fwd_excess.index)


def cs_zscore(df):
    """按列(单截面)zscore, 对特征做稳健标准化"""
    return ((df - df.mean()) / (df.std() + 1e-9)).clip(-3, 3)


def rank_ic(y_true, y_pred):
    a = pd.Series(y_true); b = pd.Series(y_pred)
    m = a.notna() & b.notna()
    if m.sum() < 10:
        return 0.0
    return a[m].rank().corr(b[m].rank())


def make_ic_feval(y_va):
    """RankIC custom metric, maximize"""
    def feval(preds, dtrain):
        return "rank_ic", rank_ic(y_va, preds)
    return feval


# ==================== 主流程 ====================
def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv(f"{DATA_DIR}/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv(f"{DATA_DIR}/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31")
    cal_list = list(cal)
    val_piv = load_raw_valuation()
    fin = load_finind_table()
    idx_close = load_index_close()

    print(f"\n{'='*100}\n  V17 残差增强模型 (value_comp + ML催化剂残差)  标签=未来60d超额剥离value_comp\n{'='*100}", flush=True)

    # ---- 逐期准备: 特征X / 残差标签y / value_comp / 未来收益 / 可交易集 ----
    # samples[(y,qi)] = dict(sig, exec, X(df,index=inst), vc(series), yres(series), fwd(series), tradable(index))
    samples = {}
    year_ctx = {}
    for y in YEARS:
        codes = build_dynamic_universe(y, fcf_df, profit_df)
        universe = sorted(format_qlib_code(c) for c in codes)
        bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        sig_exec = quarter_signal_exec_dates(y, cal_list)
        ext_start = sig_exec[0][0] - pd.Timedelta(days=400)
        ext_end = pd.Timestamp(bt_end) + pd.Timedelta(days=20)
        close, vol = load_price_volume(universe, ext_start, min(ext_end, pd.Timestamp("2026-07-31")))
        liq_table = build_liquidity_table(universe, sig_exec[0][0] - pd.Timedelta(days=10), bt_end)
        lu, sus = build_limit_up_set(universe, cal)
        year_ctx[y] = (universe, sig_exec, close, liq_table, lu, sus, bt_end)

        for qi, (sig, exec_dt) in enumerate(sig_exec):
            vc = compute_value_comp(universe, sig, val_piv)
            X = compute_catalyst_features(universe, sig, exec_dt, close, vol, idx_close, fin)
            # 可交易过滤 (与E0/v16一致)
            liq = liq_table.xs(sig, level=0).reindex(X.index) if sig in liq_table.index.get_level_values(0) else pd.Series(np.nan, index=X.index)
            tradable = X.index[(liq.notna()) & (liq >= LIQ_THRESHOLD)
                               & (~pd.Series(X.index, index=X.index).map(lambda i: (exec_dt, i) in lu))
                               & (~pd.Series(X.index, index=X.index).map(lambda i: (exec_dt, i) in sus))]
            # 未来收益: exec_dt -> 约FWD_DAYS后交易日 (标签用, 训练期才有)
            fut = [d for d in close.index if d > exec_dt]
            if exec_dt in close.index and len(fut) >= 1:
                target_day = min(fut, key=lambda d: abs((d - exec_dt).days - FWD_DAYS))
                fwd = close.loc[target_day] / close.loc[exec_dt] - 1
                pool_ret = fwd.reindex(tradable).mean()
                fwd_excess = (fwd - pool_ret).reindex(X.index)
            else:
                fwd_excess = pd.Series(np.nan, index=X.index)
            yres = residual_label(fwd_excess.reindex(tradable), vc.reindex(tradable))
            samples[(y, qi)] = dict(sig=sig, exec=exec_dt, X=X, vc=vc,
                                    yres=yres.reindex(X.index), tradable=tradable)
        print(f"  [{y}] 池{len(universe)}只 特征/残差标签就绪", flush=True)

    # 训练点顺序(时间序): 用于expanding
    keys = sorted(samples.keys())

    # ---- expanding 训练: 每个OOS点用其之前全部历史训练 ----
    # 预测: 对每个(y,qi)产出 yhat_res (仅当有足够历史)
    preds = {}
    for i, k in enumerate(keys):
        hist = keys[:i]
        # 至少累计600样本才训练
        Xtr_list, ytr_list, grp = [], [], []
        for hk in hist:
            s = samples[hk]
            tr = s["tradable"]
            xx = cs_zscore(s["X"].loc[tr, FEATURES].astype(float))
            yy = s["yres"].loc[tr]
            m = yy.notna() & xx.notna().all(axis=1)
            if m.sum() < 15:
                continue
            Xtr_list.append(xx[m]); ytr_list.append(yy[m]); grp.append(int(m.sum()))
        if sum(grp) < 400 or len(grp) < 4:
            preds[k] = None
            continue
        Xtr = pd.concat(Xtr_list); ytr = pd.concat(ytr_list)
        # 最后一个历史组作验证(RankIC早停)
        va_n = grp[-1]
        Xva = Xtr.iloc[-va_n:]; yva = ytr.iloc[-va_n:]
        Xtr2 = Xtr.iloc[:-va_n]; ytr2 = ytr.iloc[:-va_n]; grp2 = grp[:-1]
        dtr = xgb.DMatrix(Xtr2.values, label=ytr2.rank().values); dtr.set_group(grp2)
        dva = xgb.DMatrix(Xva.values, label=yva.rank().values); dva.set_group([va_n])
        model = xgb.train(XGB_RANK_PARAMS, dtr, num_boost_round=N_ROUNDS,
                          evals=[(dva, "va")], custom_metric=make_ic_feval(yva.values),
                          maximize=True, early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        va_ic = rank_ic(yva.values, model.predict(dva))
        # 预测当期
        s = samples[k]; tr = s["tradable"]
        xk = cs_zscore(s["X"].loc[tr, FEATURES].astype(float)).fillna(0)
        yhat = pd.Series(model.predict(xgb.DMatrix(xk.values)), index=tr)
        preds[k] = dict(yhat=yhat, va_ic=va_ic, best_it=model.best_iteration)

    # ---- 回测: value_comp基准 + 各alpha融合 ----
    strat_names = ["value_comp_Top20"] + [f"fusion_a{int(a*100)}" for a in ALPHAS]
    all_rets = {n: [] for n in strat_names}
    annual = {n: {} for n in strat_names}
    to_stats = {n: [] for n in strat_names}
    ic_log = []

    for y in YEARS:
        universe, sig_exec, close, liq_table, lu, sus, bt_end = year_ctx[y]
        price_mat = close.loc[sig_exec[0][0]:pd.Timestamp(bt_end)]
        rebal = {n: [] for n in strat_names}
        for qi, (sig, exec_dt) in enumerate(sig_exec):
            s = samples[(y, qi)]; tr = s["tradable"]
            vc = s["vc"].reindex(tr)
            vc_rank = vc.rank(pct=True)
            # value_comp 基准
            rebal["value_comp_Top20"].append((exec_dt, vc.nlargest(TOPK).index.tolist()))
            # 融合
            pr = preds.get((y, qi))
            for a in ALPHAS:
                nm = f"fusion_a{int(a*100)}"
                if pr is None or pr["va_ic"] < IC_DEGEN_TH:
                    hold = vc.nlargest(TOPK).index.tolist()   # 退化保护: 回退纯value_comp
                else:
                    yhat_rank = pr["yhat"].reindex(tr).rank(pct=True)
                    final = a * vc_rank + (1 - a) * yhat_rank
                    hold = final.nlargest(TOPK).index.tolist()
                rebal[nm].append((exec_dt, hold))
            if pr is not None:
                ic_log.append({"year": y, "qi": qi, "va_ic": pr["va_ic"], "best_it": pr["best_it"]})
        for nm in strat_names:
            rets, avg_to = portfolio_backtest(rebal[nm], price_mat, cal_list)
            all_rets[nm].append(rets); to_stats[nm].append(avg_to)
            annual[nm][y] = (1 + rets).prod() - 1

    # ---- 基准收益 ----
    bench_all = []
    for y in YEARS:
        be = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        bench_all.append(get_bench_returns_ak(f"{y}-01-01", be))
    bench = pd.concat(bench_all).sort_index(); bench = bench[~bench.index.duplicated(keep="last")]

    # ---- 汇总 ----
    icd = pd.DataFrame(ic_log)
    print(f"\n  验证RankIC: 均值{icd['va_ic'].mean():.4f} 中位{icd['va_ic'].median():.4f} "
          f">0比例{(icd['va_ic']>0).mean():.2f} 退化(<{IC_DEGEN_TH})次数{(icd['va_ic']<IC_DEGEN_TH).sum()}/{len(icd)}", flush=True)
    print(f"\n{'='*130}", flush=True)
    header = "  ".join(f"{y:>7}" for y in YEARS)
    print(f"{'策略':<18} | {header} | {'全期年化':>8} {'夏普':>5} {'回撤':>7} {'季换手':>6} {'IS年化':>7} {'OOS年化':>7}", flush=True)
    print(f"{'-'*130}", flush=True)
    out_rows = []
    for nm in strat_names:
        s = pd.concat(all_rets[nm]).sort_index(); s = s[~s.index.duplicated(keep="last")]
        m = calc_metrics(s)
        m_is = calc_metrics(s[s.index.year.isin(IS_YEARS)])
        m_oos = calc_metrics(s[s.index.year.isin(OOS_YEARS)])
        row = "  ".join(f"{annual[nm][y]*100:>6.2f}%" for y in YEARS)
        print(f"{nm:<18} | {row} | {m['ar']*100:>7.2f}% {m['sharpe']:>5.2f} {m['max_dd']*100:>6.1f}% "
              f"{np.mean(to_stats[nm])*100:>5.1f}% {m_is['ar']*100:>6.2f}% {m_oos['ar']*100:>6.2f}%", flush=True)
        out_rows.append({"strategy": nm, "ar": m["ar"], "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                         "is_ar": m_is["ar"], "oos_ar": m_oos["ar"], **{str(y): annual[nm][y] for y in YEARS}})
        s.to_csv(f"{WORK_DIR}/v17_returns_{nm}.csv", sep='\t', header=False)
    bm = calc_metrics(bench)
    print(f"{'沪深300(ak)':<18} | {'':>55} | {bm['ar']*100:>7.2f}% {bm['sharpe']:>5.2f} {bm['max_dd']*100:>6.1f}%", flush=True)
    print(f"{'='*130}", flush=True)
    pd.DataFrame(out_rows).to_csv(f"{WORK_DIR}/v17_results.csv", sep='\t', index=False)
    icd.to_csv(f"{WORK_DIR}/v17_ic_log.csv", sep='\t', index=False)
    print("\n[+] 已保存: v17_results.csv, v17_ic_log.csv, v17_returns_*.csv", flush=True)


if __name__ == "__main__":
    run()
