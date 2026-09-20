"""
core.features —— Alpha158 长周期扩展 + 基本面 PIT 因子 (无穿越)
================================================================================
- Alpha158Enhanced: Alpha158 + 长周期动量/波动/量价因子
- load_fund_features: FCF系基本面因子, PIT 年报Y → [Y+1-05-01, 下一年报披露前) 生效,
  按日截面 MAD-zscore (避免全局标准化时间泄露)
"""
import os
import numpy as np
import pandas as pd
from qlib.contrib.data.handler import Alpha158

from . import config
from .universe import format_qlib_code


class Alpha158Enhanced(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()
        # 滤掉 $vwap 特征: qlib bin 里没有 vwap.day.bin (腾讯 kline 接口不返回成交额),
        # 保留它只会产生全 NaN → Fillna 填 0 → 常数列, 对树模型无客 (永不分裂),
        # 但对 MLP/NN 会浪费一个维度且在 CSZScoreNorm 时产生 NaN。
        keep = [i for i, f in enumerate(fields) if "$vwap" not in f]
        fields = [fields[i] for i in keep]
        names = [names[i] for i in keep]
        # ---- 消融实验: 滤掉 SUMD/SUMP 系特征 (RSI类, 高方差噪声嫌疑) ----
        if getattr(config, "ABLATE_SUMD", False):
            ablate = {"SUMD5", "SUMD10", "SUMD20", "SUMD30", "SUMD60",
                      "SUMP5", "SUMP10", "SUMP20", "SUMP30", "SUMP60"}
            keep2 = [i for i, n in enumerate(names) if n not in ablate]
            fields = [fields[i] for i in keep2]
            names = [names[i] for i in keep2]
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
            # 短期反转因子 (A股强alpha源)。注意: Alpha158 已内置 ROC1~ROC60
            # (Ref($close,d)/Ref($close,d+1)), 不能再添加同名特征 → 列名重复会让
            # RobustZScoreNorm 赋值崩溃 (Columns must be same length as key)。
            # 这里只补充 Alpha158 没有的量价反转类因子。
            # 短期成交量反转
            "$volume/(Ref($volume, 5)+1e-12)",
            # 量价偏离: 短期涨跌幅 vs 成交量变化
            "(Ref($close, 5)/$close) * ($volume/(Mean($volume, 20)+1e-12))",
            # 12-1 动量: 过去252日剔除最近21日的收益率
            # A股短周期反转强(近月负相关), 含近月的动量会被反转吃掉
            # 学术标准: Ref($close,252)/Ref($close,21) - 1 ≈ 过去12个月剔除最近1个月
            "Ref($close, 252)/Ref($close, 21)",
        ]
        extra_names = [
            "ROC120", "ROC240", "MA120", "MA240", "STD120", "STD240",
            "BOLL60", "BOLL120", "VMA120", "VMA240",
            "CORR_PV20", "CORR_PV60", "VRATIO_5_60", "VRATIO_5_120",
            "VREV5", "PVREV5",
            "MOM_12_1",
        ]
        # 守卫: 特征名必须唯一 (重复名会让 RobustZScoreNorm 赋值崩溃, 见上注)
        all_names = names + extra_names
        dup = sorted({n for n in all_names if all_names.count(n) > 1})
        if dup:
            raise ValueError(f"[CHECK] Alpha158Enhanced 特征名重复: {dup}, "
                            f"请勿添加 Alpha158 内置同名因子!")
        return fields + extra_fields, all_names


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
    fdf = pd.DataFrame(rows, columns=["datetime", "instrument", "F_fcf_growth",
                                      "F_profit_growth", "F_fcf_profit_ratio",
                                      "F_fcf_avg_3y_norm", "F_fcf_cv_3y"])
    fdf = fdf.set_index(["datetime", "instrument"])
    fdf = fdf[~fdf.index.duplicated(keep="last")]
    # 按日截面 MAD-zscore (避免全局标准化时间泄露)
    for col in fdf.columns:
        g = fdf[col].replace([np.inf, -np.inf], np.nan).groupby(level=0)
        med = g.transform("median")
        mad = g.transform(lambda x: (x - x.median()).abs().median()).replace(0, np.nan)
        fdf[col] = ((fdf[col] - med) / (1.4826 * mad)).clip(-3, 3)
    return fdf.astype(np.float32)


def load_asset_growth_feature(universe, cal, start, end,
                               fin_health_path=None):
    """CMA (Investment Factor): 总资产同比增速, PIT 无穿越

    FF 五因子里唯一没用的一块: "花钱纪律"。与"赚钱能力"(RMW/十年双正)正交互补。
    高盈利+激进扩张 = 帝国建造经典画像, 历史回报差。
    PIT: 年报Y → [Y+1-05-01, 下一年报披露前) 生效, 与 FCF 因子同口径。
    """
    if fin_health_path is None:
        fin_health_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "data_cache", "fin_health_pit.parquet")
    if not os.path.exists(fin_health_path):
        raise RuntimeError(f"[CHECK] {fin_health_path} 缺失, CMA特征无法计算!")
    fh = pd.read_parquet(fin_health_path)
    fh["code"] = fh["code"].astype(str)
    ucodes = {c[2:] for c in universe}
    m = fh[fh["code"].isin(ucodes)][["code", "year", "ta"]].dropna().sort_values(
        ["code", "year"])
    m["asset_growth"] = m.groupby("code")["ta"].pct_change()
    m = m.dropna(subset=["asset_growth"])
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    cal_win = cal[(cal >= lo) & (cal <= hi)]
    rows = []
    for code, grp in m.groupby("code"):
        qc = format_qlib_code(code)
        grp = grp.sort_values("year").copy()
        yrs = sorted(grp["year"].unique())
        for i, (_, r) in enumerate(grp.iterrows()):
            if pd.isna(r["asset_growth"]):
                continue
            year = int(r["year"])
            af = pd.Timestamp(f"{year+1}-05-01")
            at = pd.Timestamp(f"{yrs[i+1]+1}-04-30") if i + 1 < len(yrs) \
                else pd.Timestamp(f"{year+2}-04-30")
            for d in cal_win[(cal_win >= af) & (cal_win <= at)]:
                rows.append((d, qc, r["asset_growth"]))
    if not rows:
        return pd.DataFrame(columns=["F_asset_growth"]).set_index(
            pd.MultiIndex.from_tuples([], names=["datetime", "instrument"]))
    fdf = pd.DataFrame(rows, columns=["datetime", "instrument", "F_asset_growth"])
    fdf = fdf.set_index(["datetime", "instrument"])
    fdf = fdf[~fdf.index.duplicated(keep="last")]
    for col in fdf.columns:
        g = fdf[col].replace([np.inf, -np.inf], np.nan).groupby(level=0)
        med = g.transform("median")
        mad = g.transform(lambda x: (x - x.median()).abs().median()).replace(0, np.nan)
        fdf[col] = ((fdf[col] - med) / (1.4826 * mad)).clip(-3, 3)
    return fdf.astype(np.float32)
