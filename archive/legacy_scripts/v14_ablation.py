"""
V14 消融实验: 季频Top20 + 60天超额label + 估值/基本面因子逐层叠加
=================================================================
E0: 等权持有动态股票池 (季频再平衡, 无模型) —— 基准
E1: XGB + Alpha158量价 + 60天label + 季频Top20
E2: E1 + 估值因子 (EP/BP/SP/CFP/PEG, 日频, 2018起)
E3: E2 + 深度基本面 (ROE/毛利率/营收增速/净利增速/负债率/应收周转+FCF系5因子, PIT)

继承V13全部修复: T+1执行 | 往返0.4% | 500万流动性(close*volume,失败报错) |
窗口内动态Top80特征选择 | embargo(valid提前3个月,覆盖60天label)
新增修复: 季度初即满仓建仓 (V13每年1月空仓缺陷)
Label说明: Ref($close,-60)/$close-1 + CSZScoreNorm横截面标准化,
           数学上等价于60天相对大盘超额收益的截面zscore (基准分量为截面常数)
"""
import os, sys, random
import warnings, logging

def set_seed(seed=42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")
logging.getLogger('qlib.data.data').setLevel(logging.ERROR)

from v5_validation import build_limit_up_set, filter_pred_by_tradability
from v5_pipeline_fix import load_fundamental_features_fixed
from v5_xgb_turnover_comparative import build_dynamic_universe, format_qlib_code

WORK_DIR = "/Users/11164591/Documents/Qoder目录/qlib"
DATA_DIR = "/Users/11164591/Documents/Qoder目录"

XGB_PARAMS = {
    "objective": "reg:squarederror", "learning_rate": 0.005,
    "max_depth": 4, "colsample_bytree": 0.8879, "subsample": 0.8789,
    "reg_alpha": 10.0, "reg_lambda": 50.0,
    "tree_method": "hist", "nthread": 4, "seed": 42,
}
N_ROUNDS, EARLY_STOP = 1000, 100
FS_ROUNDS, FS_EARLY_STOP, FS_LR = 200, 30, 0.05
TOP_N_FEATURES = 80
TOPK = 20                      # 季频Top20
FEE_ROUNDTRIP = 0.004
LIQ_THRESHOLD = 5_000_000
LABEL_HORIZON_MONTHS = 3       # 60交易日label → embargo 3个月

VAL_FEATS = ["ep", "bp", "sp", "cfp", "peg"]
FIN_FEATS = ["roe", "gross_margin", "rev_growth", "profit_growth", "debt_ratio", "ar_turnover"]


def generate_quarterly_windows():
    windows = []
    for year in [2019, 2021, 2022, 2023, 2024, 2025, 2026]:
        periods = [(1, 3), (4, 6), (7, 9)] if year == 2026 else [(1, 3), (4, 6), (7, 9), (10, 12)]
        for start_m, end_m in periods:
            bt_start = pd.Timestamp(f"{year}-{start_m:02d}-01")
            bt_end = pd.Timestamp(f"{year}-{end_m:02d}-01") + relativedelta(months=1) - pd.Timedelta(days=1)
            train_end = bt_start - relativedelta(years=1) - pd.Timedelta(days=1)
            train_start = train_end - relativedelta(years=4)
            valid_start = train_end + pd.Timedelta(days=1)
            valid_end = bt_start - relativedelta(months=LABEL_HORIZON_MONTHS) - pd.Timedelta(days=1)
            # test段起点前移10天: 覆盖上季末信号日截面
            test_start = bt_start - pd.Timedelta(days=10)
            windows.append({
                "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
                "valid": (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
                "test": (test_start.strftime("%Y-%m-%d"), bt_end.strftime("%Y-%m-%d")),
                "bt_start": bt_start, "bt_end": bt_end,
                "name": f"{year}_Q{start_m//3 + 1}", "year": year})
    return windows


# ==================== 外部因子加载 (模块级缓存) ====================
def cs_mad_zscore(fdf):
    """按日截面 MAD zscore, clip±3, 缺失→0(中性)"""
    for col in fdf.columns:
        grp = fdf[col].groupby(level=0)
        med = grp.transform("median")
        mad = grp.transform(lambda x: (x - x.median()).abs().median()).replace(0, np.nan)
        fdf[col] = ((fdf[col] - med) / (1.4826 * mad)).clip(-3, 3)
    return fdf

_VAL_TABLE = None
def load_valuation_table():
    """估值因子日频表: EP/BP/SP/CFP/PEG, 索引(datetime, instrument)"""
    global _VAL_TABLE
    if _VAL_TABLE is not None:
        return _VAL_TABLE
    v = pd.read_csv(f"{DATA_DIR}/valuation_cache.csv", sep='\t', dtype={"code": str})
    if len(v) == 0:
        raise RuntimeError("[估值] valuation_cache.csv 为空!")
    v["date"] = pd.to_datetime(v["date"])
    for c in ["pe_ttm", "pb", "ps_ttm", "pcf", "peg"]:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v["ep"] = 1.0 / v["pe_ttm"]; v["bp"] = 1.0 / v["pb"]
    v["sp"] = 1.0 / v["ps_ttm"]; v["cfp"] = 1.0 / v["pcf"]
    v["instrument"] = v["code"].map(format_qlib_code)
    fdf = v.set_index(["date", "instrument"])[VAL_FEATS].replace([np.inf, -np.inf], np.nan)
    fdf = fdf[~fdf.index.duplicated(keep="last")].sort_index()
    _VAL_TABLE = cs_mad_zscore(fdf)
    print(f"  [估值表] {_VAL_TABLE.shape}, {fdf.index.get_level_values(1).nunique()}只, "
          f"{fdf.index.get_level_values(0).min().date()}~{fdf.index.get_level_values(0).max().date()}", flush=True)
    return _VAL_TABLE

_FIN_TABLE = None
def load_finind_table():
    """财务指标表: 按披露日PIT排序, 用于merge_asof"""
    global _FIN_TABLE
    if _FIN_TABLE is not None:
        return _FIN_TABLE
    f = pd.read_csv(f"{DATA_DIR}/finind_cache.csv", sep='\t', dtype={"code": str})
    if len(f) == 0:
        raise RuntimeError("[财务] finind_cache.csv 为空!")
    f["report_date"] = pd.to_datetime(f["report_date"], errors="coerce")
    f = f.dropna(subset=["report_date"])
    for c in FIN_FEATS:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    # PIT披露规则: Q1→05-01, H1→09-01, Q3→11-01, 年报→次年05-01
    def avail_date(rd):
        if rd.month == 3: return pd.Timestamp(rd.year, 5, 1)
        if rd.month == 6: return pd.Timestamp(rd.year, 9, 1)
        if rd.month == 9: return pd.Timestamp(rd.year, 11, 1)
        return pd.Timestamp(rd.year + 1, 5, 1)
    f["avail"] = f["report_date"].map(avail_date)
    f["instrument"] = f["code"].map(format_qlib_code)
    f = f.sort_values(["avail", "report_date"]).drop_duplicates(
        subset=["instrument", "avail"], keep="last")
    _FIN_TABLE = f[["instrument", "avail"] + FIN_FEATS].sort_values("avail")
    print(f"  [财务表] {_FIN_TABLE.shape}, {f['instrument'].nunique()}只", flush=True)
    return _FIN_TABLE

def join_valuation(X):
    val = load_valuation_table()
    add = val.reindex(X.index)
    add.columns = [f"v_{c}" for c in add.columns]
    return add.fillna(0)

def join_finind(X):
    fin = load_finind_table()
    idx = X.index.to_frame(index=False)
    idx.columns = ["datetime", "instrument"]
    idx["_row"] = np.arange(len(idx))
    idx = idx.sort_values("datetime")
    merged = pd.merge_asof(idx, fin, left_on="datetime", right_on="avail",
                           by="instrument", direction="backward")
    merged = merged.sort_values("_row")
    add = merged[FIN_FEATS].copy()
    add.index = X.index
    add.columns = [f"f_{c}" for c in add.columns]
    # 按日截面标准化
    add = add.replace([np.inf, -np.inf], np.nan)
    add = cs_mad_zscore(add).fillna(0)
    return add


# ==================== 窗口数据准备 (三实验共享) ====================
def prepare_window_data(win, fcf_df, profit_df, cal):
    codes = build_dynamic_universe(win["year"], fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    ts, te = win["train"]; vs, ve = win["valid"]; xs, xe = win["test"]

    dhc = {"start_time": ts, "end_time": xe, "fit_start_time": ts, "fit_end_time": te,
        "instruments": universe,
        "infer_processors": [{"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":True}},
                              {"class":"Fillna","kwargs":{"fields_group":"feature"}}],
        "learn_processors": [{"class":"DropnaLabel"}, {"class":"CSZScoreNorm","kwargs":{"fields_group":"label"}}],
        "label": ["Ref($close, -60) / $close - 1"]}   # 60天label(截面zscore≈超额)
    dsc = {"class":"DatasetH","module_path":"qlib.data.dataset",
        "kwargs":{"handler":{"class":"Alpha158Enhanced","module_path":"v5_validation","kwargs":dhc},
                  "segments":{"train":(ts,te),"valid":(vs,ve),"test":(xs,xe)}}}
    dataset = init_instance_by_config(dsc)

    train_df = dataset.prepare("train", col_set=["feature","label"], data_key=DataHandlerLP.DK_L)
    valid_df = dataset.prepare("valid", col_set=["feature","label"], data_key=DataHandlerLP.DK_L)
    test_X = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)

    X_tr, y_tr = train_df["feature"], train_df["label"].iloc[:, 0]
    X_va, y_va = valid_df["feature"], valid_df["label"].iloc[:, 0]
    va_mask = y_va.notna()
    X_va, y_va = X_va[va_mask], y_va[va_mask]
    del dataset

    # 外部因子 (一次性join, 各实验取列子集)
    fund = load_fundamental_features_fixed(universe, cal)   # FCF系5因子(日频PIT)
    ext = {}
    for name, X in [("tr", X_tr), ("va", X_va), ("te", test_X)]:
        val_add = join_valuation(X)
        fin_add = join_finind(X)
        fcf_add = fund.reindex(X.index).fillna(0) if fund is not None else None
        if fcf_add is not None:
            fcf_add.columns = [f"c_{c}" for c in fcf_add.columns]
        ext[name] = (val_add, fin_add, fcf_add)
    return X_tr, y_tr, X_va, y_va, test_X, universe, ext


def build_exp_matrices(exp, X_tr, X_va, test_X, ext):
    """E1=量价; E2=+估值; E3=+估值+财务+FCF系"""
    outs = []
    for name, X in [("tr", X_tr), ("va", X_va), ("te", test_X)]:
        val_add, fin_add, fcf_add = ext[name]
        parts = [X]
        if exp in ("E2", "E3"):
            parts.append(val_add)
        if exp == "E3":
            parts.append(fin_add)
            if fcf_add is not None:
                parts.append(fcf_add)
        outs.append(pd.concat(parts, axis=1))
    return outs


def select_and_train(X_tr, y_tr, X_va, y_va):
    """两阶段: stage1快速全特征取Top80 → stage2正式训练"""
    feats_all = list(X_tr.columns)
    p1 = dict(XGB_PARAMS); p1["learning_rate"] = FS_LR
    dtr = xgb.DMatrix(X_tr.values, label=y_tr.values, feature_names=feats_all)
    dva = xgb.DMatrix(X_va.values, label=y_va.values, feature_names=feats_all)
    m1 = xgb.train(p1, dtr, num_boost_round=FS_ROUNDS, evals=[(dva, "valid")],
                   early_stopping_rounds=FS_EARLY_STOP, verbose_eval=False)
    imp = pd.Series(m1.get_score(importance_type="gain")).reindex(feats_all).fillna(0.0)
    feats = imp.nlargest(TOP_N_FEATURES).index.tolist()

    dtr2 = xgb.DMatrix(X_tr[feats].values, label=y_tr.values, feature_names=feats)
    dva2 = xgb.DMatrix(X_va[feats].values, label=y_va.values, feature_names=feats)
    m2 = xgb.train(XGB_PARAMS, dtr2, num_boost_round=N_ROUNDS, evals=[(dva2, "valid")],
                   early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    return m2, feats, m2.best_iteration


# ==================== 流动性 ====================
def build_liquidity_table(universe, start, end):
    lookback_start = pd.Timestamp(start) - pd.Timedelta(days=60)
    px = D.features(list(universe), ["$close", "$volume"],
                    start_time=lookback_start, end_time=end)
    if px is None or len(px) == 0:
        raise RuntimeError(f"[流动性] 数据拉取失败 ({start}~{end}), 拒绝静默跳过!")
    px = px.reset_index()
    px.columns = ["instrument", "datetime", "close", "volume"]
    px["amount"] = px["close"] * px["volume"]
    px = px.sort_values(["instrument", "datetime"])
    px["avg20"] = px.groupby("instrument")["amount"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    return px.set_index(["datetime", "instrument"])["avg20"]


# ==================== 统一精确组合引擎 ====================
def load_price_matrix(insts, start, end):
    px = D.features(list(insts), ["$close"], start_time=start, end_time=end)
    if px is None or len(px) == 0:
        raise RuntimeError(f"[价格] 数据拉取失败 ({start}~{end})!")
    px = px.reset_index()
    px.columns = ["instrument", "datetime", "close"]
    mat = px.pivot(index="datetime", columns="instrument", values="close").sort_index()
    return mat.ffill()   # 停牌用最后已知价

def portfolio_backtest(rebalances, price_mat, cal_list):
    """
    rebalances: [(exec_dt, [目标等权持仓列表]), ...] 按时间排序
    T+1已体现在exec_dt的选取上; 成本=单边换手×0.4%
    返回: 日收益序列, 平均单边换手
    """
    dates = [d for d in price_mat.index if d >= rebalances[0][0]]
    shares = {}
    value = 1.0
    nav, nav_dates = [], []
    turnovers = []
    ri = 0
    for d in dates:
        px = price_mat.loc[d]
        # 先按当日收盘估值
        if shares:
            value = sum(sh * px[inst] for inst, sh in shares.items() if pd.notna(px.get(inst)))
        # 调仓日: 当日收盘价再平衡
        if ri < len(rebalances) and d == rebalances[ri][0]:
            targets = [t for t in rebalances[ri][1] if pd.notna(px.get(t)) and px[t] > 0]
            if targets:
                w_tgt = {t: 1.0 / len(targets) for t in targets}
                w_cur = {inst: sh * px[inst] / value for inst, sh in shares.items()
                         if pd.notna(px.get(inst))} if shares and value > 0 else {}
                all_inst = set(w_tgt) | set(w_cur)
                turnover = 0.5 * sum(abs(w_tgt.get(i, 0) - w_cur.get(i, 0)) for i in all_inst)
                cost = value * turnover * FEE_ROUNDTRIP
                value -= cost
                shares = {t: w_tgt[t] * value / px[t] for t in targets}
                turnovers.append(turnover)
            ri += 1
        nav.append(value)
        nav_dates.append(d)
    nav = pd.Series(nav, index=nav_dates)
    rets = nav.pct_change().fillna(0)
    return rets, (np.mean(turnovers) if turnovers else 0)


def calc_metrics(returns):
    if len(returns) == 0:
        return {"ar": 0, "sharpe": 0, "max_dd": 0}
    n_years = len(returns) / 252
    ar = (1 + returns).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = ar / vol if vol > 0 else 0
    nav = (1 + returns).cumprod()
    max_dd = ((nav / nav.cummax()) - 1).min()
    return {"ar": ar, "sharpe": sharpe, "max_dd": max_dd}


def get_bench_returns(start, end):
    df = D.features(["SH000300"], ["$close"], start_time=start, end_time=end)
    s = df.reset_index().set_index("datetime")["$close"].sort_index()
    return s.pct_change().fillna(0)


def quarter_signal_exec_dates(year, cal_list):
    """返回[(信号日, 执行日, 季度起点), ...]: 信号=季度首日前最后交易日, 执行=季度首交易日"""
    qs = [f"{year}-01-01", f"{year}-04-01", f"{year}-07-01"]
    if year != 2026:
        qs.append(f"{year}-10-01")
    out = []
    for q in qs:
        q = pd.Timestamp(q)
        sig = max([d for d in cal_list if d < q])
        exec_dt = min([d for d in cal_list if d >= q])
        out.append((sig, exec_dt))
    return out


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv(f"{DATA_DIR}/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv(f"{DATA_DIR}/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31")
    cal_list = list(cal)

    print(f"\n{'='*100}", flush=True)
    print(f"  V14 消融实验: E0等权基准 | E1量价+60d label | E2+估值 | E3+深度基本面  (季频Top{TOPK})", flush=True)
    print(f"{'='*100}", flush=True)

    windows = generate_quarterly_windows()
    years = sorted(set(w["year"] for w in windows))
    # preds[exp][year] = [各季度 (信号日截面pred)]
    preds = {e: {} for e in ["E1", "E2", "E3"]}
    year_universe = {}
    latest_models = {}

    for win in windows:
        print(f"\n[+] 窗口 {win['name']}: train{win['train']} valid{win['valid']} test{win['test']}", flush=True)
        X_tr, y_tr, X_va, y_va, test_X, universe, ext = prepare_window_data(win, fcf_df, profit_df, cal)
        year_universe[win["year"]] = universe
        print(f"    数据: train{X_tr.shape} valid{X_va.shape} test{test_X.shape}", flush=True)

        for exp in ["E1", "E2", "E3"]:
            Xt, Xv, Xe = build_exp_matrices(exp, X_tr, X_va, test_X, ext)
            model, feats, best_it = select_and_train(Xt, y_tr, Xv, y_va)
            scores = model.predict(xgb.DMatrix(Xe[feats].values, feature_names=feats))
            pred = pd.DataFrame({"score": scores}, index=Xe.index)
            preds[exp].setdefault(win["year"], []).append(pred)
            print(f"    [{exp}] {Xt.shape[1]}特征→Top{len(feats)}, best_iter={best_it}", flush=True)
            if win["name"] == "2026_Q3":
                latest_models[exp] = {"model": model, "features": feats}
                joblib.dump(latest_models[exp], f"{WORK_DIR}/v14_{exp}_latest_Q.pkl")

    # ==================== 回测 ====================
    print(f"\n{'='*100}\n  回测 (季频, T+1执行, 往返0.4%, 500万流动性)\n{'='*100}", flush=True)
    exp_list = ["E0", "E1", "E2", "E3"]
    all_rets = {e: [] for e in exp_list}
    annual = {e: {} for e in exp_list}
    to_stats = {e: [] for e in exp_list}

    for y in years:
        universe = year_universe[y]
        bt_start = f"{y}-01-01"
        bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        sig_exec = quarter_signal_exec_dates(y, cal_list)
        liq_table = build_liquidity_table(universe, sig_exec[0][0] - pd.Timedelta(days=10), bt_end)
        price_mat = load_price_matrix(universe, sig_exec[0][0], bt_end)
        limit_up_set, suspension_set = build_limit_up_set(universe, cal)

        row = f"  {y}:"
        for exp in exp_list:
            rebalances = []
            for sig, exec_dt in sig_exec:
                if exp == "E0":
                    # 等权全池 (有价格数据的)
                    holdings = [i for i in universe if i in price_mat.columns
                                and pd.notna(price_mat.loc[exec_dt, i])]
                else:
                    pred_y = pd.concat(preds[exp][y]).sort_index()
                    pred_y = pred_y[~pred_y.index.duplicated(keep="last")]
                    pred_f = filter_pred_by_tradability(pred_y, limit_up_set, suspension_set)
                    if sig not in pred_f.index.get_level_values(0):
                        raise RuntimeError(f"[{exp}] 信号日{sig}不在pred中!")
                    day_pred = pred_f.xs(sig, level=0)["score"].copy()
                    if sig not in liq_table.index.get_level_values(0):
                        raise RuntimeError(f"[流动性] 信号日{sig}无数据!")
                    day_liq = liq_table.xs(sig, level=0).reindex(day_pred.index)
                    day_pred[day_liq.isna() | (day_liq < LIQ_THRESHOLD)] = -np.inf
                    day_pred = day_pred[day_pred > -np.inf]
                    holdings = day_pred.nlargest(TOPK).index.tolist()
                rebalances.append((exec_dt, holdings))

            rets, avg_to = portfolio_backtest(rebalances, price_mat, cal_list)
            all_rets[exp].append(rets)
            to_stats[exp].append(avg_to)
            annual[exp][y] = (1 + rets).prod() - 1
            row += f"  {exp} {annual[exp][y]*100:>7.2f}%"
        print(row, flush=True)

    # ==================== 汇总 ====================
    bench_all = []
    for y in years:
        be = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        bench_all.append(get_bench_returns(f"{y}-01-01", be))
    bench = pd.concat(bench_all).sort_index()
    bench = bench[~bench.index.duplicated(keep="last")]
    bench_m = calc_metrics(bench)

    print(f"\n{'='*118}", flush=True)
    header = "  ".join(f"{y:>7}" for y in years)
    print(f"{'实验':<16} | {header} | {'全期年化':>8} {'夏普':>5} {'回撤':>7} {'季换手':>6} {'vs沪深300超额':>10}", flush=True)
    print(f"{'-'*118}", flush=True)
    results = {}
    labels = {"E0": "E0 等权基准", "E1": "E1 量价60d", "E2": "E2 +估值", "E3": "E3 +基本面"}
    for exp in exp_list:
        s = pd.concat(all_rets[exp]).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        m = calc_metrics(s)
        results[exp] = m
        excess_ar = m["ar"] - bench_m["ar"]
        row = "  ".join(f"{annual[exp][y]*100:>6.2f}%" for y in years)
        print(f"{labels[exp]:<14} | {row} | {m['ar']*100:>7.2f}% {m['sharpe']:>5.2f} {m['max_dd']*100:>6.1f}% "
              f"{np.mean(to_stats[exp])*100:>5.1f}% {excess_ar*100:>+9.2f}pp", flush=True)
        s.to_csv(f"{WORK_DIR}/v14_returns_{exp}.csv", sep='\t', header=False)
    row_b = "  ".join(f"{(1+bench[bench.index.year==y]).prod()-1:>7.2%}" for y in years)
    print(f"{'沪深300':<14} | {row_b} | {bench_m['ar']*100:>7.2f}% {bench_m['sharpe']:>5.2f} {bench_m['max_dd']*100:>6.1f}%", flush=True)
    print(f"{'='*118}", flush=True)

    # 消融增量
    print("\n消融增量 (全期年化):", flush=True)
    print(f"  模型排序增量  (E1-E0): {(results['E1']['ar']-results['E0']['ar'])*100:+.2f}pp", flush=True)
    print(f"  估值因子增量  (E2-E1): {(results['E2']['ar']-results['E1']['ar'])*100:+.2f}pp", flush=True)
    print(f"  基本面因子增量(E3-E2): {(results['E3']['ar']-results['E2']['ar'])*100:+.2f}pp", flush=True)

    rows = []
    for exp in exp_list:
        for y in years:
            rows.append({"exp": exp, "year": y, "annual_ret": annual[exp][y]})
        rows.append({"exp": exp, "year": "ALL", "annual_ret": results[exp]["ar"],
                     "sharpe": results[exp]["sharpe"], "max_dd": results[exp]["max_dd"]})
    pd.DataFrame(rows).to_csv(f"{WORK_DIR}/v14_ablation_results.csv", sep='\t', index=False)
    print(f"\n[+] 结果已保存: v14_ablation_results.csv", flush=True)


if __name__ == "__main__":
    run()
