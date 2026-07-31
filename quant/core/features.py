"""
core.features —— Alpha158 长周期扩展 + 基本面 PIT 因子 (无穿越)
================================================================================
- Alpha158Enhanced: Alpha158 + 长周期动量/波动/量价因子
- load_fund_features: FCF系基本面因子, PIT 年报Y → [Y+1-05-01, 下一年报披露前) 生效,
  按日截面 MAD-zscore (避免全局标准化时间泄露)
"""
import numpy as np
import pandas as pd
from qlib.contrib.data.handler import Alpha158

from .universe import format_qlib_code


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
