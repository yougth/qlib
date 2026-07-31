"""
V18 全栈增强: L0存活偏差修正 + L1月度换仓 + L2扩展正交特征 + L3动态α融合/集中
=================================================================================
在 V17 残差增强(value_comp + ML催化剂残差)基础上, 一次性叠加四层优化:
  L0 存活偏差: 用 PIT 增广股票池缓存(含退市股) 构建候选池, 消除幸存者偏差
  L1 月度换仓: 季度(每年4点) -> 月度(每年~12点), 训练样本翻3倍, 缓解模型退化
  L2 扩展特征: 在V17的12个催化剂特征上, 追加6个与估值低相关的正交因子
              (max_ret彩票 / amihud非流动性 / skew偏度 / 52周高点邻近 /
               毛利率加速度 / 负债率变化)
  L3 融合升级: alpha 按当期验证RankIC动态调节(模型强则多信); Top15集中版对比Top20
时间轴锁定同V17: 特征全在换仓点之前, 标签=未来60天超额剥离value_comp, 训练用历史配对。
产出: 图1全部指标(年化收益/年化波动/Sharpe/最大回撤/Calmar/IC/RankIC/换手率/交易次数)
     + 每年收益 + IS/OOS 拆分。
"""
import os, sys, random, warnings, logging
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
                          load_finind_table, WORK_DIR, DATA_DIR)
from v16_value_rules import load_raw_valuation, LIQ_THRESHOLD
from v15_model_fixed import get_bench_returns_ak
# 复用 V17 已验证的组件
from v17_residual_model import (load_price_volume, load_index_close, compute_value_comp,
                                residual_label, cs_zscore, rank_ic, make_ic_feval)

YEARS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
IS_YEARS = {2019, 2021, 2022}
OOS_YEARS = {2023, 2024, 2025, 2026}
TOPK = 20
TOPK_CONC = 15          # L3集中版
FWD_DAYS = 60           # 标签: 未来60天(约一季)超额
IC_DEGEN_TH = 0.005

# L2 扩展催化剂特征 (V17的12个 + 6个正交增量), 全部基于换仓点之前信息, 与估值正交
FEATURES = ["mom_60", "mom_120", "rev_5", "rev_20", "turn_z", "turn_slope",
            "ivol_20", "beta_60", "profit_accel", "rev_accel", "roe_q_chg", "month",
            "max_ret_20", "amihud_20", "skew_20", "high_prox", "gm_accel", "debt_chg"]

XGB_RANK_PARAMS = dict(
    objective="rank:pairwise", eta=0.03, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.4, reg_alpha=1.0, reg_lambda=5.0,
    disable_default_eval_metric=1,
)
N_ROUNDS, EARLY_STOP = 400, 40


def monthly_signal_exec_dates(year, cal_list):
    """L1: 月度换仓。返回[(信号日, 执行日), ...]。信号=月首前最后交易日, 执行=月首交易日"""
    months = range(1, 8) if year == 2026 else range(1, 13)
    out = []
    for m in months:
        q = pd.Timestamp(f"{year}-{m:02d}-01")
        prev = [d for d in cal_list if d < q]
        nxt = [d for d in cal_list if d >= q]
        if not prev or not nxt:
            continue
        out.append((max(prev), min(nxt)))
    return out


def compute_catalyst_features(universe, sig, exec_dt, close, vol, idx_close, fin):
    """L2: 换仓点之前的催化剂特征 (V17量价+边际基本面+日历, 再加6个正交增量)"""
    f = pd.DataFrame(index=pd.Index(sorted(universe), name="instrument"))
    px = close.loc[:sig]
    if len(px) > 260:
        rets = px.pct_change()
        f["mom_60"] = (px.iloc[-1] / px.iloc[-61] - 1).reindex(f.index)
        f["mom_120"] = (px.iloc[-1] / px.iloc[-121] - 1).reindex(f.index)
        f["rev_5"] = -(px.iloc[-1] / px.iloc[-6] - 1).reindex(f.index)
        f["rev_20"] = -(px.iloc[-1] / px.iloc[-21] - 1).reindex(f.index)
        f["ivol_20"] = rets.iloc[-20:].std().reindex(f.index)
        # L2新增: 彩票效应(近20日单日最大涨幅, 负向预测)/收益偏度/52周高点邻近
        f["max_ret_20"] = rets.iloc[-20:].max().reindex(f.index)
        f["skew_20"] = rets.iloc[-60:].skew().reindex(f.index)
        f["high_prox"] = (px.iloc[-1] / px.iloc[-250:].max()).reindex(f.index)
        if idx_close is not None:
            ir = idx_close.reindex(px.index).pct_change()
            win = rets.iloc[-60:]; iw = ir.iloc[-60:]
            cov = win.apply(lambda col: col.cov(iw))
            f["beta_60"] = (cov / iw.var()).reindex(f.index)
        else:
            f["beta_60"] = np.nan
    else:
        for c in ["mom_60", "mom_120", "rev_5", "rev_20", "ivol_20",
                  "max_ret_20", "skew_20", "high_prox", "beta_60"]:
            f[c] = np.nan
    # 换手率趋势 + L2新增 amihud 非流动性(|日收益|/成交额)
    vv = vol.loc[:sig]
    if len(vv) > 40:
        v20 = vv.iloc[-20:]
        f["turn_z"] = ((vv.iloc[-1] - v20.mean()) / v20.std()).reindex(f.index)
        x = np.arange(20)
        slope = v20.apply(lambda col: np.polyfit(x, col.values, 1)[0] / (col.mean() + 1e-9)
                          if col.notna().sum() >= 10 else np.nan)
        f["turn_slope"] = slope.reindex(f.index)
        r20 = close.loc[:sig].pct_change().iloc[-20:].abs()
        amt = (close.loc[:sig].iloc[-20:] * vv.iloc[-20:])
        f["amihud_20"] = ((r20 / (amt + 1e-9)).mean() * 1e9).reindex(f.index)
    else:
        f["turn_z"] = np.nan; f["turn_slope"] = np.nan; f["amihud_20"] = np.nan
    f = _add_fundamental(f, universe, sig, fin)
    f["month"] = exec_dt.month
    return f


def _add_fundamental(f, universe, sig, fin):
    """边际基本面: 只用avail<=sig的PIT财报。V17三项 + L2毛利率加速度/负债率变化"""
    sub = fin[(fin["avail"] <= sig) & (fin["instrument"].isin(universe))].copy()
    sub = sub.sort_values(["instrument", "avail"])
    has_gm = "gross_margin" in sub.columns
    has_debt = "debt_ratio" in sub.columns
    prof_a, rev_a, roe_c, gm_a, debt_c = {}, {}, {}, {}, {}
    for inst, g in sub.groupby("instrument"):
        g = g.tail(3)
        def d2(col):
            v = g[col].values if col in g.columns else np.array([])
            return (v[-1] - v[-2]) if len(v) >= 2 and np.isfinite(v[-1]) and np.isfinite(v[-2]) else np.nan
        prof_a[inst] = d2("profit_growth")
        rev_a[inst] = d2("rev_growth")
        roe_c[inst] = d2("roe")
        gm_a[inst] = d2("gross_margin") if has_gm else np.nan
        debt_c[inst] = d2("debt_ratio") if has_debt else np.nan
    f["profit_accel"] = pd.Series(prof_a).reindex(f.index)
    f["rev_accel"] = pd.Series(rev_a).reindex(f.index)
    f["roe_q_chg"] = pd.Series(roe_c).reindex(f.index)
    f["gm_accel"] = pd.Series(gm_a).reindex(f.index)
    f["debt_chg"] = pd.Series(debt_c).reindex(f.index)
    return f


def dyn_alpha(va_ic):
    """L3: 动态融合权重。va_ic越高越多信模型 -> alpha(value权重)越低。范围[0.55,0.80]"""
    return float(np.clip(0.75 - 2.0 * max(va_ic, 0.0), 0.55, 0.80))


def trade_count(rebal_list):
    """总买入次数(容量压力代理): 每次换仓相对上期新进的名字数之和"""
    prev, total = set(), 0
    for _, hold in rebal_list:
        cur = set(hold); total += len(cur - prev); prev = cur
    return total


def full_metrics(returns):
    """图1指标(组合层): 年化收益/年化波动/Sharpe/最大回撤/Calmar"""
    if len(returns) == 0:
        return dict(ar=0, vol=0, sharpe=0, max_dd=0, calmar=0)
    m = calc_metrics(returns)
    vol = returns.std() * np.sqrt(252)
    calmar = m["ar"] / abs(m["max_dd"]) if m["max_dd"] < 0 else 0.0
    return dict(ar=m["ar"], vol=vol, sharpe=m["sharpe"], max_dd=m["max_dd"], calmar=calmar)



# ==================== 主流程 ====================
def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    # L0: 用 PIT 增广缓存(含退市股)构建候选池, 修存活偏差; 缺失则回退存量缓存
    fcf_path = f"{DATA_DIR}/fcf_cache_pit.csv" if os.path.exists(f"{DATA_DIR}/fcf_cache_pit.csv") else f"{DATA_DIR}/fcf_cache.csv"
    prof_path = f"{DATA_DIR}/profit_cache_pit.csv" if os.path.exists(f"{DATA_DIR}/profit_cache_pit.csv") else f"{DATA_DIR}/profit_cache.csv"
    fcf_df = pd.read_csv(fcf_path, sep='\t'); fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df = pd.read_csv(prof_path, sep='\t'); profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    print(f"  [L0] 股票池缓存: {os.path.basename(fcf_path)} / {os.path.basename(prof_path)}", flush=True)

    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31")
    cal_list = list(cal)
    val_piv = load_raw_valuation()
    fin = load_finind_table()
    idx_close = load_index_close()

    print(f"\n{'='*110}\n  V18 全栈: L0存活偏差 + L1月度换仓 + L2扩展特征({len(FEATURES)}个) + L3动态α/集中\n{'='*110}", flush=True)

    # ---- 逐期准备: 特征/残差标签/value_comp/未来超额/可交易集 ----
    samples = {}      # (y, mi) -> dict
    year_ctx = {}
    for y in YEARS:
        codes = build_dynamic_universe(y, fcf_df, profit_df)
        universe = sorted(format_qlib_code(c) for c in codes)
        bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        sig_exec = monthly_signal_exec_dates(y, cal_list)
        ext_start = sig_exec[0][0] - pd.Timedelta(days=400)
        # 前向缓冲: 让年末月份的未来60d标签也能取到(次年初), 上限=数据末端
        ext_end = min(pd.Timestamp(f"{y+1}-03-31"), pd.Timestamp("2026-07-31"))
        close, vol = load_price_volume(universe, ext_start, ext_end)
        liq_table = build_liquidity_table(universe, sig_exec[0][0] - pd.Timedelta(days=10), bt_end)
        lu, sus = build_limit_up_set(universe, cal)
        year_ctx[y] = (universe, sig_exec, close, bt_end)

        for mi, (sig, exec_dt) in enumerate(sig_exec):
            vc = compute_value_comp(universe, sig, val_piv)
            X = compute_catalyst_features(universe, sig, exec_dt, close, vol, idx_close, fin)
            liq = (liq_table.xs(sig, level=0).reindex(X.index)
                   if sig in liq_table.index.get_level_values(0) else pd.Series(np.nan, index=X.index))
            tradable = X.index[(liq.notna()) & (liq >= LIQ_THRESHOLD)
                               & (~pd.Series(X.index, index=X.index).map(lambda i: (exec_dt, i) in lu))
                               & (~pd.Series(X.index, index=X.index).map(lambda i: (exec_dt, i) in sus))]
            fut = [d for d in close.index if d > exec_dt]
            if exec_dt in close.index and len(fut) >= 1:
                target_day = min(fut, key=lambda d: abs((d - exec_dt).days - FWD_DAYS))
                # 需真正跨越~FWD_DAYS才算有效标签, 否则(最近月份)置NaN
                if (target_day - exec_dt).days >= FWD_DAYS * 0.6:
                    fwd = close.loc[target_day] / close.loc[exec_dt] - 1
                    pool_ret = fwd.reindex(tradable).mean()
                    fwd_excess = (fwd - pool_ret).reindex(X.index)
                else:
                    fwd_excess = pd.Series(np.nan, index=X.index)
            else:
                fwd_excess = pd.Series(np.nan, index=X.index)
            yres = residual_label(fwd_excess.reindex(tradable), vc.reindex(tradable))
            samples[(y, mi)] = dict(sig=sig, exec=exec_dt, X=X, vc=vc,
                                    yres=yres.reindex(X.index),
                                    fwd=fwd_excess.reindex(X.index), tradable=tradable)
        print(f"  [{y}] 池{len(universe)}只 月度点{len(sig_exec)}个 就绪", flush=True)

    keys = sorted(samples.keys())

    # ---- expanding 训练 (每点用其之前全部历史, 最后一组做RankIC早停验证) ----
    preds = {}
    for i, k in enumerate(keys):
        hist = keys[:i]
        Xtr_list, ytr_list, grp = [], [], []
        for hk in hist:
            s = samples[hk]; tr = s["tradable"]
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
        va_n = grp[-1]
        Xva = Xtr.iloc[-va_n:]; yva = ytr.iloc[-va_n:]
        Xtr2 = Xtr.iloc[:-va_n]; ytr2 = ytr.iloc[:-va_n]; grp2 = grp[:-1]
        dtr = xgb.DMatrix(Xtr2.values, label=ytr2.rank().values); dtr.set_group(grp2)
        dva = xgb.DMatrix(Xva.values, label=yva.rank().values); dva.set_group([va_n])
        model = xgb.train(XGB_RANK_PARAMS, dtr, num_boost_round=N_ROUNDS,
                          evals=[(dva, "va")], custom_metric=make_ic_feval(yva.values),
                          maximize=True, early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        va_ic = rank_ic(yva.values, model.predict(dva))
        s = samples[k]; tr = s["tradable"]
        xk = cs_zscore(s["X"].loc[tr, FEATURES].astype(float)).fillna(0)
        yhat = pd.Series(model.predict(xgb.DMatrix(xk.values)), index=tr)
        preds[k] = dict(yhat=yhat, va_ic=va_ic, best_it=model.best_iteration)

    # ---- 回测: 4个策略 ----
    #  value_comp_M20   : 月度纯估值Top20 (隔离L1换仓频率效应)
    #  V18_full_Top20   : 全栈(L0+L1+L2+L3动态α) Top20
    #  V18_full_Top15   : 全栈 + L3集中Top15
    strat = {
        "value_comp_M20": dict(topk=TOPK, use_model=False),
        "V18_full_Top20": dict(topk=TOPK, use_model=True),
        "V18_full_Top15": dict(topk=TOPK_CONC, use_model=True),
    }
    all_rets = {n: [] for n in strat}
    annual = {n: {} for n in strat}
    to_stats = {n: [] for n in strat}
    trades = {n: 0 for n in strat}
    ic_pairs = {n: [] for n in strat}   # (score, fwd_excess) per period -> IC/RankIC
    va_ic_log = []

    for y in YEARS:
        universe, sig_exec, close, bt_end = year_ctx[y]
        price_mat = close.loc[sig_exec[0][0]:pd.Timestamp(bt_end)]
        rebal = {n: [] for n in strat}
        for mi, (sig, exec_dt) in enumerate(sig_exec):
            s = samples[(y, mi)]; tr = s["tradable"]
            vc = s["vc"].reindex(tr); vc_rank = vc.rank(pct=True)
            fwd = s["fwd"].reindex(tr)
            pr = preds.get((y, mi))
            if pr is not None:
                va_ic_log.append(pr["va_ic"])
            for nm, cfg in strat.items():
                if (not cfg["use_model"]) or pr is None or pr["va_ic"] < IC_DEGEN_TH:
                    score = vc_rank                              # 退化/纯估值 -> value_comp
                else:
                    a = dyn_alpha(pr["va_ic"])
                    score = a * vc_rank + (1 - a) * pr["yhat"].reindex(tr).rank(pct=True)
                hold = score.nlargest(cfg["topk"]).index.tolist()
                rebal[nm].append((exec_dt, hold))
                # 因子IC: 该期打分 vs 未来超额
                pair = pd.concat([score.rename("s"), fwd.rename("f")], axis=1).dropna()
                if len(pair) >= 10:
                    ic_pairs[nm].append((pair["s"].corr(pair["f"]),
                                         pair["s"].rank().corr(pair["f"].rank())))
        for nm in strat:
            rets, avg_to = portfolio_backtest(rebal[nm], price_mat, cal_list)
            all_rets[nm].append(rets); to_stats[nm].append(avg_to)
            annual[nm][y] = (1 + rets).prod() - 1
            trades[nm] += trade_count(rebal[nm])

    # ---- 原始季度基准(参考锚, 25.11%): 直接读V17产物若在, 否则跳过 ----
    # 基准指数
    bench_all = []
    for y in YEARS:
        be = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        bench_all.append(get_bench_returns_ak(f"{y}-01-01", be))
    bench = pd.concat(bench_all).sort_index(); bench = bench[~bench.index.duplicated(keep="last")]

    # ---- 汇总输出 ----
    va_ic_arr = np.array(va_ic_log)
    print(f"\n  [模型] 验证RankIC 均值{va_ic_arr.mean():.4f} 中位{np.median(va_ic_arr):.4f} "
          f">0比例{(va_ic_arr>0).mean():.2f} 退化(<{IC_DEGEN_TH})次数{(va_ic_arr<IC_DEGEN_TH).sum()}/{len(va_ic_arr)}", flush=True)

    print(f"\n{'='*150}", flush=True)
    yhdr = "  ".join(f"{y:>7}" for y in YEARS)
    print(f"{'策略':<16} | {yhdr} | {'年化':>7} {'波动':>6} {'夏普':>5} {'回撤':>7} {'Calmar':>6} "
          f"{'IC':>6} {'RankIC':>6} {'换手':>5} {'交易数':>5} {'IS':>7} {'OOS':>7}", flush=True)
    print("-" * 150, flush=True)
    out_rows = []
    for nm in strat:
        s = pd.concat(all_rets[nm]).sort_index(); s = s[~s.index.duplicated(keep="last")]
        m = full_metrics(s)
        m_is = full_metrics(s[s.index.year.isin(IS_YEARS)])
        m_oos = full_metrics(s[s.index.year.isin(OOS_YEARS)])
        ic_arr = np.array([p[0] for p in ic_pairs[nm]])
        ric_arr = np.array([p[1] for p in ic_pairs[nm]])
        ic = np.nanmean(ic_arr) if len(ic_arr) else 0.0
        ric = np.nanmean(ric_arr) if len(ric_arr) else 0.0
        avg_to = np.mean(to_stats[nm]) * 100
        yrow = "  ".join(f"{annual[nm][y]*100:>6.2f}%" for y in YEARS)
        print(f"{nm:<16} | {yrow} | {m['ar']*100:>6.2f}% {m['vol']*100:>5.1f}% {m['sharpe']:>5.2f} "
              f"{m['max_dd']*100:>6.1f}% {m['calmar']:>6.2f} {ic:>6.3f} {ric:>6.3f} "
              f"{avg_to:>4.1f}% {trades[nm]:>5d} {m_is['ar']*100:>6.2f}% {m_oos['ar']*100:>6.2f}%", flush=True)
        out_rows.append(dict(strategy=nm, ar=m["ar"], vol=m["vol"], sharpe=m["sharpe"],
                             max_dd=m["max_dd"], calmar=m["calmar"], ic=ic, rank_ic=ric,
                             turnover=avg_to/100, trades=trades[nm],
                             is_ar=m_is["ar"], oos_ar=m_oos["ar"],
                             **{str(y): annual[nm][y] for y in YEARS}))
        s.to_csv(f"{WORK_DIR}/v18_returns_{nm}.csv", sep='\t', header=False)
    bm = full_metrics(bench)
    print(f"{'沪深300(ak)':<16} | {'':>63} | {bm['ar']*100:>6.2f}% {bm['vol']*100:>5.1f}% "
          f"{bm['sharpe']:>5.2f} {bm['max_dd']*100:>6.1f}% {bm['calmar']:>6.2f}", flush=True)
    print("=" * 150, flush=True)
    print("  注: 原季度基准 value_comp Top20 = 全期25.11%/夏普1.21/OOS19.98% (对比锚)", flush=True)
    pd.DataFrame(out_rows).to_csv(f"{WORK_DIR}/v18_results.csv", sep='\t', index=False)
    print("\n[+] 已保存 v18_results.csv, v18_returns_*.csv", flush=True)


if __name__ == "__main__":
    run()
