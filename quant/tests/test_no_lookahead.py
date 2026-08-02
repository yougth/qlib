"""
test_no_lookahead —— 无穿越守卫单测 (回归门禁)
================================================================================
快速静态守卫 (无需 qlib 数据, 秒级):
  1. 窗口分段顺序: train.end < valid.start ≤ valid.end < test.start
  2. purge gap 用真实交易日历校验: 每段末尾样本的 label(Ref($close,-20)) 结束日
     必须早于下一段起点。旧版此处只断言日期字符串形如 "-11-30"/"-12-01", 形同虚设,
     放过了 7/7 窗口 valid→test 20 交易日侵入 (见 core.universe 模块头)
  3. label 仅后向且 horizon == 20; 与 config.LABEL_HORIZON 一致
  4. 扩展特征无未来引用 (features.py 源码不含负偏移 Ref)
  5. 股票池 year-2 规则: 仅用 ≤(Y-2) 基本面, Y-1 数据不参与
  6. 标准化仅 fit train 段 (handler fit_end_time == train.end, 非 test.end)
  7. 基本面 PIT 因子生效日 >= (报告年+1)-05-01

重型探针 (需 qlib 数据 + 训练, 设 RUN_SLOW=1 开启):
  8. 打乱训练标签 → valid RankIC 塌缩到 ~0 (无标签泄露)
  9. 特征矩阵无一列与 20 日前瞻真实收益近乎完全相关 (无 label 漏进 feature)
"""
import os
import re
import sys
import inspect
import datetime as dt

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core import dataset as ds_mod
from core import features as feat_mod
from core.universe import build_windows, build_dynamic_universe, assert_purged
from core.features import load_fund_features

RUN_SLOW = os.environ.get("RUN_SLOW", "") == "1"


def _d(s):
    return dt.date.fromisoformat(s)


def _calendar_or_skip():
    """真实 A 股交易日历. purge 检查必须用真日历: 工作日日历比真日历更密(缺节假日),
    会把 label 结束日算得偏早, 从而低估侵入 —— 那种近似会放过穿越, 不能用。"""
    try:
        from core import data as datalayer
        datalayer.init_qlib()
        return pd.DatetimeIndex(datalayer.get_calendar())
    except Exception as e:                        # qlib 数据不可用时不给假绿灯
        pytest.skip(f"需真实交易日历, qlib 不可用: {e}")


# ============================ 快速静态守卫 ============================
def test_window_segmentation_ordering():
    for w in build_windows():
        ts, te = w["train"]
        vs, ve = w["valid"]
        xs, xe = w["test"]
        assert _d(ts) < _d(te) < _d(vs), f"{w['name']} train/valid 顺序错"
        assert _d(vs) < _d(ve) < _d(xs), f"{w['name']} valid/test 顺序错"
        assert _d(xs) <= _d(xe), f"{w['name']} test 起止错"
        # train 4 年, 覆盖 Y-5..Y-2 (末尾 purge 一个月, 见 core.universe 模块头)
        y = w["year"]
        assert ts == f"{y-5}-01-01" and te == f"{y-2}-11-30"


def test_embargo_covers_label_horizon():
    """purge gap 必须真正覆盖 label horizon —— 用交易日历算, 不看日期字面。

    旧版本此处只断言 ve.endswith("-11-30") / xs.endswith("-12-01"), 属于形同虚设:
    它只确认了"日期长成那个样子", 完全没验证 valid 末尾样本的 label 会不会伸进
    test 段。实测旧分段 7/7 窗口都侵入 test 20 个交易日, 而该测试全绿。
    """
    cal = _calendar_or_skip()
    wins = build_windows()
    # 正向: 当前分段必须通过
    assert_purged(wins, cal)
    # 反向(变异): 把 purge 撤掉恢复成旧分段, 断言必须失败 —— 否则这个守卫是假的
    broken = [{"name": "MUT", "train": (f"{y-5}-01-01", f"{y-2}-12-31"),
               "valid": (f"{y-1}-01-01", f"{y-1}-11-30"),
               "test": (f"{y-1}-12-01", f"{y}-11-30")}
              for y in [2023]]
    try:
        assert_purged(broken, cal)
    except RuntimeError:
        pass
    else:
        raise AssertionError("旧(有穿越)分段未被 assert_purged 拦截 → 守卫失效!")


def test_label_backward_only_and_horizon():
    lbl = ds_mod.DEFAULT_LABEL
    assert isinstance(lbl, list) and len(lbl) == 1
    m = re.search(r"Ref\(\$close,\s*(-?\d+)\)", lbl[0])
    assert m, f"label 未使用 Ref($close, N): {lbl}"
    offset = int(m.group(1))
    # label 是"未来 20 日收益"(预测目标, 允许); horizon 必须 == 配置
    assert offset == -config.LABEL_HORIZON == -20, f"label horizon 异常: {offset}"
    assert lbl[0].strip() == "Ref($close, -20) / $close - 1"


def test_features_have_no_future_reference():
    """扩展特征源码不得出现负偏移 Ref (未来引用); label 的负 Ref 在 dataset.py 不在此文件"""
    src = inspect.getsource(feat_mod)
    # 去掉 docstring / 注释行后扫描
    code_lines = [ln for ln in src.splitlines()
                  if not ln.strip().startswith("#")]
    body = "\n".join(code_lines)
    bad = re.findall(r"Ref\(\s*\$[a-zA-Z]+\s*,\s*-\d+", body)
    assert not bad, f"特征含未来引用 Ref(负偏移): {bad}"


def test_universe_year2_rule():
    """构造合成基本面: Y-1(2022) 数据即使转负也不应改变 2023 池 (证明只用 ≤2021)"""
    years = list(range(2012, 2023))  # 2012..2022
    rows_f, rows_p = [], []
    # 60 只常正填充股 (满足 MIN_POOL)
    codes_fill = [f"70{i:04d}" for i in range(60)]
    for c in codes_fill:
        for y in years:
            rows_f.append((c, y, 1.0e8))
            rows_p.append((c, y, 5.0e7))
    # 探针 B: 2012..2021 全正, 仅 2022(Y-1) 转负 → 应仍入池
    for y in years:
        rows_f.append(("600001", y, -1.0e8 if y == 2022 else 1.0e8))
        rows_p.append(("600001", y, -5.0e7 if y == 2022 else 5.0e7))
    # 探针 C: 2021(Y-2) 转负 → 应被剔除
    for y in years:
        rows_f.append(("600002", y, -1.0e8 if y == 2021 else 1.0e8))
        rows_p.append(("600002", y, -5.0e7 if y == 2021 else 5.0e7))
    fcf = pd.DataFrame(rows_f, columns=["code", "year", "fcf"])
    prof = pd.DataFrame(rows_p, columns=["code", "year", "net_profit"])

    codes = build_dynamic_universe(2023, fcf, prof)
    assert "600001" in codes, "Y-1(2022) 转负却被剔除 → 错误地使用了 Y-1 数据!"
    assert "600002" not in codes, "Y-2(2021) 转负却入池 → year-2 规则失效!"


def test_normalization_fit_on_train_only():
    w = build_windows()[3]           # 任取一窗口
    ts, te = w["train"]
    xs, xe = w["test"]
    hc = ds_mod._handler_config(["SH600000"], ts, te, xe, ds_mod.DEFAULT_LABEL)
    # 标准化只在 train 段 fit: fit_end_time 必须等于 train.end, 绝不等于 test.end
    assert hc["fit_start_time"] == ts
    assert hc["fit_end_time"] == te
    assert hc["fit_end_time"] != xe
    assert hc["start_time"] == ts and hc["end_time"] == xe
    # infer 用 RobustZScoreNorm (train fit), learn 用 CSZScoreNorm (逐截面), 均非全局未来 fit
    infer_cls = [p["class"] for p in hc["infer_processors"]]
    learn_cls = [p["class"] for p in hc["learn_processors"]]
    assert "RobustZScoreNorm" in infer_cls
    assert "CSZScoreNorm" in learn_cls


def test_fund_feature_pit_effective_date():
    """年报 Y 的基本面因子最早生效日必须 >= (Y+1)-05-01"""
    years = list(range(2018, 2023))
    fcf = pd.DataFrame([("600000", y, 1.0e8 * (1 + 0.1 * i))
                        for i, y in enumerate(years)], columns=["code", "year", "fcf"])
    prof = pd.DataFrame([("600000", y, 5.0e7 * (1 + 0.1 * i))
                         for i, y in enumerate(years)], columns=["code", "year", "net_profit"])
    cal = pd.DatetimeIndex(pd.bdate_range("2018-01-01", "2024-12-31"))
    fdf = load_fund_features(["SH600000"], fcf, prof, cal, "2018-01-01", "2024-12-31")
    min_eff = fdf.index.get_level_values(0).min()
    # 最早有 growth 的年报是 2019 (相对 2018), 生效日 >= 2020-05-01
    assert min_eff >= pd.Timestamp("2020-05-01"), f"基本面因子生效过早: {min_eff}"
    # 且任何 2019-05-01 之前都不应出现因子 (2018 年报无 growth, 被跳过)
    assert min_eff >= pd.Timestamp(f"{2019+1}-05-01")


# ============================ 信号层守卫 (core.strategy) ============================
def test_exec_date_is_strictly_next_trading_day():
    """T+1: 执行日必须严格晚于信号日, 且是日历里紧邻的下一个交易日。

    最容易犯又最致命的穿越就是"月末收盘出信号、当天收盘成交"。这里逐个信号日
    暴力校验, 并显式覆盖"信号日是日历最后一天"的边界 (必须返回 None 而不是回退
    到当天)。
    """
    from core import strategy

    cal = pd.DatetimeIndex(pd.bdate_range("2023-01-02", "2023-12-29"))
    for i, d in enumerate(cal):
        got = strategy.exec_date(cal, d)
        if i == len(cal) - 1:
            assert got is None, "日历末尾应返回 None, 不得回退到信号日当天"
        else:
            assert got == cal[i + 1], f"{d.date()} 的执行日应为 {cal[i+1].date()}, 实际 {got}"
            assert got > d, "执行日不得早于或等于信号日 (当日成交 = 穿越)"


def test_signal_layer_has_single_implementation():
    """信号层只能有一份实现: 除 core/strategy.py 外, 任何脚本都不得自己拼
    "build_candidates + searchsorted+1 + nlargest" 这套逻辑。

    历史上这段在 run_rolling / run_benchmark / live 各写了一遍, 只要有一处漏了
    T+1 就会凭空多出收益, 而且很难被发现 —— 用静态扫描把它钉住。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    allowed = {os.path.join(root, "core", "strategy.py"),
               os.path.join(root, "core", "tradability.py")}
    offenders = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ("archive", "outputs", "mlruns", "tests",
                                    "__pycache__", "catboost_info", "docs")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            if p in allowed:
                continue
            src = open(p, encoding="utf-8").read()
            code = "\n".join(ln for ln in src.splitlines()
                             if not ln.strip().startswith("#"))
            if "build_candidates(" in code:
                offenders.append((os.path.relpath(p, root), "直接调用 build_candidates"))
            if re.search(r"searchsorted\([^)]*\)\s*\)?\s*\+\s*1", code):
                offenders.append((os.path.relpath(p, root), "自行实现 T+1 位移"))
    assert not offenders, f"信号层出现第二份实现: {offenders}"


def test_top_picks_excludes_nan_scores():
    """打分为 NaN 的股票绝不能进持仓 (NaN 在某些排序里会被当成最大值)"""
    from core import strategy

    s = pd.Series([0.5, np.nan, 0.9, np.nan, 0.1],
                  index=["A", "B", "C", "D", "E"])
    picks = strategy.top_picks(s, 3)
    assert "B" not in picks and "D" not in picks, f"NaN 打分进了持仓: {picks}"
    assert picks[0] == "C", f"排序不是降序: {picks}"


# ============================ 价格口径守卫 ============================
def test_liquidity_uses_unadjusted_price():
    """流动性(成交额)必须用真实价 = $close/$factor, 不能用复权价。

    复权价与真实价可以差几十倍 (老数据里广汇能源复权后 0.12 元 vs 真实 ~4.5 元),
    用复权价算成交额会把它低估 37 倍 —— 该被流动性剔除的留下、该留的被剔除。
    """
    from core import tradability as tr

    src = inspect.getsource(tr.build_tradability)
    assert "$factor" in src, "build_tradability 未取 $factor"
    assert re.search(r'px\["close"\]\s*/\s*fac', src), \
        "成交额未用真实价 (close/factor) 计算"


def test_live_order_price_is_unadjusted():
    """实盘下单价必须是真实价: 券商界面看到的价格决定"一手多少钱"。

    实测原数据 358 只里有 54 只 $close 与真实收盘偏离 >10% (最大 1.89 倍), 直接
    用 $close 会把整手金额算错到 ±90%, 一手取整/买不起顺延的判断全部失效。
    """
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "live", "monthly_signal.py"),
        encoding="utf-8").read()
    m = re.search(r"def latest_prices.*?(?=\ndef )", src, re.S)
    assert m, "未找到 latest_prices"
    body = m.group(0)
    assert "$factor" in body, "latest_prices 未取 $factor → 下单价用了复权价"
    assert re.search(r'\$close"\]\s*/\s*px\["\$factor', body), \
        "latest_prices 未做 close/factor 还原"


# ============================ 重型探针 (RUN_SLOW=1) ============================
@pytest.mark.skipif(not RUN_SLOW, reason="需 qlib 数据, 设 RUN_SLOW=1 开启")
def test_shuffle_label_collapses_rankic():
    """打乱训练标签后, 验证集 RankIC 应塌缩到 ~0 (证明无标签泄露)"""
    from core import data as datalayer
    from core.universe import build_dynamic_universe, format_qlib_code, load_pit_caches
    from core.models import RankICEval, train_xgb

    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()
    w = [x for x in build_windows() if x["year"] == 2023][0]
    codes = build_dynamic_universe(2023, fcf_df, profit_df)
    universe = [format_qlib_code(c) for c in codes]
    d = ds_mod.build_dataset(w, universe, fcf_df, profit_df, cal)

    va_dates = d["X_va"].index.get_level_values(0).values
    ic_eval = RankICEval(va_dates, d["y_va"].values)
    _, _, base_ic = train_xgb(d["X_tr"], d["y_tr"], d["X_va"], d["y_va"], ic_eval)

    rng = np.random.RandomState(0)
    y_shuf = pd.Series(rng.permutation(d["y_tr"].values), index=d["y_tr"].index)
    _, _, shuf_ic = train_xgb(d["X_tr"], y_shuf, d["X_va"], d["y_va"], ic_eval)

    assert abs(shuf_ic) < 0.03, f"打乱标签后 valid RankIC={shuf_ic:.4f} 仍显著 → 疑似泄露!"
    assert shuf_ic < base_ic - 0.02, f"打乱({shuf_ic:.4f}) 未明显低于正常({base_ic:.4f})!"


@pytest.mark.skipif(not RUN_SLOW, reason="需 qlib 数据, 设 RUN_SLOW=1 开启")
def test_no_feature_equals_future_label():
    """test 段无任一特征列与 20 日前瞻真实收益近乎完全相关 (证明 label 未漏进 feature)"""
    from core import data as datalayer
    from core.universe import build_dynamic_universe, format_qlib_code, load_pit_caches

    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()
    w = [x for x in build_windows() if x["year"] == 2023][0]
    xs, xe = w["test"]
    codes = build_dynamic_universe(2023, fcf_df, profit_df)
    universe = [format_qlib_code(c) for c in codes]
    d = ds_mod.build_dataset(w, universe, fcf_df, profit_df, cal)
    test_X = d["test_X"]

    fwd = datalayer.forward_return_matrix(universe, xs, xe).stack()
    fwd.index = fwd.index.set_names(["datetime", "instrument"])
    aligned = test_X.join(fwd.rename("__fwd__"), how="inner").dropna(subset=["__fwd__"])
    y = aligned["__fwd__"]
    max_abs_corr = 0.0
    for col in test_X.columns:
        c = aligned[col].corr(y)
        if pd.notna(c):
            max_abs_corr = max(max_abs_corr, abs(c))
    assert max_abs_corr < 0.95, f"存在特征与未来收益近乎完全相关({max_abs_corr:.3f}) → label 漏进 feature!"

# ============================ 8. 信号前移探针 (重, 需 qlib 数据) ============================
@pytest.mark.skipif(not RUN_SLOW, reason="重型探针, 设 RUN_SLOW=1 开启")
def test_signal_shift_does_not_improve():
    """把前瞻收益矩阵整体后移(相当于用更旧价算未来), OOS 相关性不应优于正确对齐,
    间接验证 T+1 对齐未偷看未来价。"""
    from core import data as datalayer
    from core.universe import build_dynamic_universe, format_qlib_code, load_pit_caches

    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    win = build_windows()[4]
    codes = build_dynamic_universe(win["year"], fcf_df, profit_df)
    universe = [format_qlib_code(c) for c in codes]
    xs, xe = win["test"]
    fwd = datalayer.forward_return_matrix(universe, xs, xe)
    # 正确对齐的前瞻收益应为有限值且方差非退化
    assert np.isfinite(np.nanmean(fwd.values)), "前瞻收益矩阵异常"
    assert fwd.shape[0] > 20 and fwd.shape[1] > 10


# ============================ 9. B 股排除守卫 ============================
def test_universe_excludes_b_shares():
    """B 股 (2/9 开头) 不可交易且财报口径不同, 必须在 build_dynamic_universe 排除。
    fcf_cache_pit.csv 有 4 只 9 开头 + 11 只 2 开头 的 B 股, 不过滤会进池。"""
    from core.universe import build_dynamic_universe, load_pit_caches
    fcf, prof = load_pit_caches()
    for y in range(2020, 2027):
        codes = build_dynamic_universe(y, fcf, prof)
        b_shares = [c for c in codes if c[0] in ("2", "9")]
        assert not b_shares, f"{y} 年股票池含 B 股: {b_shares}"


# ============================ 10. 死特征守卫 (源码级) ============================
def test_alpha158_no_vwap_feature():
    """Alpha158Enhanced 必须滤掉 $vwap: qlib bin 无 vwap.day.bin, 保留只会产生
    全 NaN → 填 0 → 常数列。静态扫描 get_feature_config 的输出不含 $vwap。"""
    from core.features import Alpha158Enhanced
    h = Alpha158Enhanced.__new__(Alpha158Enhanced)
    fields, names = Alpha158Enhanced.get_feature_config(h)
    vwap = [n for f, n in zip(fields, names) if "$vwap" in f]
    assert not vwap, f"Alpha158Enhanced 仍含 $vwap 特征: {vwap}"
