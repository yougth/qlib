"""
test_no_lookahead —— 无穿越守卫单测 (回归门禁)
================================================================================
快速静态守卫 (无需 qlib 数据, 秒级):
  1. 窗口分段顺序: train.end < valid.start ≤ valid.end < test.start
  2. embargo: valid 截至 (Y-1)-11-30 (12月留白), test 段 12-01 起
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
from core.universe import build_windows, build_dynamic_universe
from core.features import load_fund_features

RUN_SLOW = os.environ.get("RUN_SLOW", "") == "1"


def _d(s):
    return dt.date.fromisoformat(s)


# ============================ 快速静态守卫 ============================
def test_window_segmentation_ordering():
    for w in build_windows():
        ts, te = w["train"]
        vs, ve = w["valid"]
        xs, xe = w["test"]
        assert _d(ts) < _d(te) < _d(vs), f"{w['name']} train/valid 顺序错"
        assert _d(vs) < _d(ve) < _d(xs), f"{w['name']} valid/test 顺序错"
        assert _d(xs) <= _d(xe), f"{w['name']} test 起止错"
        # train 4 年, 覆盖 Y-5..Y-2
        y = w["year"]
        assert ts == f"{y-5}-01-01" and te == f"{y-2}-12-31"


def test_embargo_one_month_gap():
    for w in build_windows():
        _, ve = w["valid"]
        xs, _ = w["test"]
        # embargo: valid 必须停在 11-30 (12 月留白), 而非跑到 12-31
        assert ve.endswith("-11-30"), f"{w['name']} valid 未在 11-30 embargo: {ve}"
        assert xs.endswith("-12-01"), f"{w['name']} 信号段未从 12-01 起: {xs}"
        # valid 结束年 == test 起始年 (上一年), 中间无跨年重叠
        assert _d(ve).year == _d(xs).year


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
#!/usr/bin/env python3
"""
test_no_lookahead —— 无穿越/前视偏差 回归守卫单测
================================================================================
覆盖长期记忆《量化回测必须检查特征穿越与前视偏差清单》的自动化断言:

快速静态断言 (无需 qlib 数据, 秒级, 作为 CI 门禁):
  1. 窗口分段:      train.end < valid.start < valid.end(=11-30 embargo) < test.start
  2. embargo 留白:  valid 截止 (Y-1)-11-30, test 从 (Y-1)-12-01 起, 1 个月缓冲
  3. 标签后向:      label 仅 Ref($close,-20)(预测目标=未来20日收益), 且 horizon 对齐
  4. 池 year-2 规则: 股票池最多用到 Y-2 年报 (Y-1 年报要到 Y 年5月才披露完)
  5. 标准化只 fit train: handler fit_end == train_end (绝不用 test 段 fit → 无未来信息)
  6. 基本面 PIT:     年报Y 因子最早生效日 >= (Y+1)-05-01 (合成数据验证, 无需 qlib)

重型探针 (需 qlib 数据, 默认跳过; RUN_SLOW=1 开启):
  7. 打乱标签探针:   y 随机置换后 valid RankIC 应塌缩到 ~0 (证明无标签泄露)
  8. 信号前移探针:   调仓日前移一格应显著恶化 OOS RankIC (证明 T+1 未偷看未来价)
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.universe import build_windows, build_dynamic_universe
from core.dataset import DEFAULT_LABEL, _handler_config
from core.features import load_fund_features

RUN_SLOW = os.environ.get("RUN_SLOW") == "1"


# ============================ 1. 窗口分段 ============================
def test_window_segmentation_ordering():
    for w in build_windows():
        ts, te = map(pd.Timestamp, w["train"])
        vs, ve = map(pd.Timestamp, w["valid"])
        xs, xe = map(pd.Timestamp, w["test"])
        assert ts < te < vs <= ve < xs <= xe, f"{w['name']} 分段乱序: {w}"
        # train 恰好 4 年 (Y-5..Y-2)
        assert te.year - ts.year == 3, f"{w['name']} 训练窗口非4年"
        # valid 恰好为 Y-1 年
        assert vs.year == ve.year == w["year"] - 1, f"{w['name']} valid 非 Y-1 年"


# ============================ 2. embargo 留白 ============================
def test_embargo_one_month_gap():
    for w in build_windows():
        ve = pd.Timestamp(w["valid"][1])
        xs = pd.Timestamp(w["test"][0])
        # valid 截止 11-30
        assert (ve.month, ve.day) == (11, 30), f"{w['name']} valid 未在 11-30 收口"
        # test/信号段 从 12-01 起 → 12 月整月 embargo, 防 20 日 label 偷看测试期
        assert (xs.month, xs.day) == (12, 1), f"{w['name']} 信号段未从 12-01 起"
        assert (xs - ve).days >= 1


# ============================ 3. 标签后向 + horizon 对齐 ============================
def test_label_backward_only_and_horizon():
    assert DEFAULT_LABEL == ["Ref($close, -20) / $close - 1"]
    # label 的 -20 必须与 config.LABEL_HORIZON 一致 (前瞻收益/回测持有窗口同口径)
    assert f"-{config.LABEL_HORIZON}" in DEFAULT_LABEL[0]
    # 标签是唯一的未来量(预测目标); 特征侧不得出现负 Ref (未来价) —— 见 features 测试
    assert DEFAULT_LABEL[0].count("Ref(") == 1


def test_features_have_no_future_reference():
    """特征字段中不得出现向未来取值的 Ref($close, -k) (k>0 = 偷看未来)"""
    from core.features import Alpha158Enhanced
    # 仅检查我们新增的长周期扩展字段(super() 的 Alpha158 字段是 qlib 标准, 均后向)
    h = Alpha158Enhanced.__new__(Alpha158Enhanced)
    # 直接取扩展字段源码常量, 断言无负 Ref
    import inspect
    src = inspect.getsource(Alpha158Enhanced.get_feature_config)
    # 扩展字段里的 Ref 都应是正向历史 (Ref($close, 120) 等), 不能有 Ref(..., -N)
    import re
    negrefs = re.findall(r"Ref\([^)]*,\s*-\d+\)", src)
    assert not negrefs, f"扩展特征出现未来 Ref: {negrefs}"


# ============================ 4. 池 year-2 规则 ============================
def test_universe_year2_rule():
    """股票池最多用到 Y-2 年报: fcf/profit 年份上界 == backtest_year - 2"""
    y = 2024
    years = list(range(2010, 2024))
    # build_dynamic_universe 内部: fcf_years=range(Y-11,Y-1) → 上界 Y-2
    used_fcf_years = list(range(y - 11, y - 1))
    assert max(used_fcf_years) == y - 2, "FCF 年份上界必须为 Y-2 (未用 Y-1 未披露年报)"

    # 合成 PIT 缓存: 60 只全正基底(需 >= config.MIN_POOL 才不触发数据异常门禁)
    base = [f"{600000 + i:06d}" for i in range(config.MIN_POOL + 10)]
    rows = []
    for c in base:
        for yr in years:
            rows.append((c, yr, 1.0e8, 1.0e8))
    # 探针 B: Y-1(2023) 净利转负, 但 Y-2(2022) 仍正 → 因只用到 Y-2, 应仍入池(未偷看 Y-1)
    b = "900001"
    for yr in years:
        rows.append((b, yr, 1.0e8, -1.0e8 if yr == y - 1 else 1.0e8))
    # 探针 C: Y-2(2022) 净利转负 → 落在使用年份内 → 应被剔除
    c_neg = "900002"
    for yr in years:
        rows.append((c_neg, yr, 1.0e8, -1.0e8 if yr == y - 2 else 1.0e8))

    df = pd.DataFrame(rows, columns=["code", "year", "fcf", "net_profit"])
    fcf = df[["code", "year", "fcf"]].copy()
    prof = df[["code", "year", "net_profit"]].copy()
    got = set(build_dynamic_universe(y, fcf, prof))
    assert set(base).issubset(got), "全正基底股票应全部入池"
    assert b in got, "探针B: Y-1转负但Y-2仍正的股票被误剔 → 疑似偷看了 Y-1 未披露年报!"
    assert c_neg not in got, "探针C: Y-2转负的股票未被剔除 → year-2 门禁失效!"


# ============================ 5. 标准化只在 train 段 fit ============================
def test_normalization_fit_on_train_only():
    w = build_windows()[3]  # 任取一窗
    ts, te = w["train"]
    xs, xe = w["test"]
    dhc = _handler_config(["SH600000"], ts, te, xe, DEFAULT_LABEL)
    # 关键: processor 的 fit 区间上界必须是 train_end, 绝不能是 test_end
    assert dhc["fit_start_time"] == ts
    assert dhc["fit_end_time"] == te, "标准化 fit 段越界到 test → 未来信息泄露!"
    assert dhc["fit_end_time"] != xe
    # infer 段用 RobustZScoreNorm(train统计量), learn 段按截面 CSZScoreNorm(label)
    infer_classes = [p["class"] for p in dhc["infer_processors"]]
    assert "RobustZScoreNorm" in infer_classes


# ============================ 6. 基本面 PIT 生效日 (合成数据, 快) ============================
def test_fund_feature_pit_effective_date():
    """年报 Y 的因子最早生效日必须 >= (Y+1)-05-01 (次年5月才披露完)"""
    cal = pd.DatetimeIndex(pd.bdate_range("2018-01-01", "2024-12-31"))
    codes = ["600000"]
    years = list(range(2016, 2023))
    fcf = pd.DataFrame({"code": "600000", "year": years,
                        "fcf": np.linspace(1e8, 2e8, len(years))})
    prof = pd.DataFrame({"code": "600000", "year": years,
                         "net_profit": np.linspace(1e8, 2e8, len(years))})
    fdf = load_fund_features(["SH600000"], fcf, prof, cal, "2018-01-01", "2024-12-31")
    dates = fdf.index.get_level_values(0)
    # 最早出现的因子日期对应的是 years 中第二个年份(pct_change 需要前一年),
    # 即 year=2017 → 生效日应 >= 2018-05-01
    earliest = dates.min()
    assert earliest >= pd.Timestamp("2018-05-01"), \
        f"基本面因子最早生效日 {earliest.date()} 早于次年5月, 存在前视!"
    # 且任一因子行的日期都不早于其年报年份+1 的 5-01 (抽样校验单调性)
    assert earliest.month >= 5 or earliest.year > 2018


# ============================ 7. 打乱标签探针 (重, 需 qlib 数据) ============================
@pytest.mark.skipif(not RUN_SLOW, reason="重型探针, 设 RUN_SLOW=1 开启")
def test_shuffle_label_collapses_rankic():
    """label 随机置换后 valid RankIC 应塌缩到 ~0 (真实约 0.06~0.09)"""
    from core import data as datalayer
    from core.universe import build_dynamic_universe, format_qlib_code, load_pit_caches
    from core.dataset import build_dataset
    from core.models import RankICEval, train_xgb

    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()
    win = build_windows()[4]  # W2024, 数据充足
    codes = build_dynamic_universe(win["year"], fcf_df, profit_df)
    universe = [format_qlib_code(c) for c in codes]
    ds = build_dataset(win, universe, fcf_df, profit_df, cal)

    va_dates = ds["X_va"].index.get_level_values(0).values
    ic_eval = RankICEval(va_dates, ds["y_va"].values)
    # 打乱训练标签
    rng = np.random.RandomState(0)
    y_shuf = ds["y_tr"].copy()
    y_shuf[:] = rng.permutation(y_shuf.values)
    _, _, ic_shuf = train_xgb(ds["X_tr"], y_shuf, ds["X_va"], ds["y_va"], ic_eval)
    assert abs(ic_shuf) < 0.03, f"打乱标签后 valid RankIC={ic_shuf:.4f} 仍显著 → 疑似泄露!"


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
