#!/usr/bin/env python3
"""归因: 被ROIC/债务规则剔除的股票 vs 保留股票, 各自等权组合表现如何?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from core import data as datalayer
from core.universe import build_windows, build_dynamic_universe, format_qlib_code, load_pit_caches
from core.fin_health import load_fin_health, load_fin_codes, filter_pool


def main():
    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    fin_df = load_fin_health(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data_cache",
        "fin_health_pit.parquet"))
    fin_codes = load_fin_codes()

    drop_rets, keep_rets = [], []
    print(f"{'窗口':<8}{'剔除数':>6}  被剔除组合当年收益   保留组合当年收益")
    for win in build_windows():
        y, (xs, xe) = win["year"], win["test"]
        codes = build_dynamic_universe(y, fcf_df, profit_df)
        kept, stats = filter_pool(codes, y, fin_df, fin_codes)
        dropped = [c for c in codes if c not in set(kept)]

        def port_ret(cl):
            if not cl:
                return float("nan")
            insts = [format_qlib_code(c) for c in cl]
            pm = datalayer.load_price_matrix(insts).ffill()
            pm = pm[(pm.index >= xs) & (pm.index <= xe)]
            return pm.iloc[-1].mean() / pm.iloc[0].mean() - 1

        dr, kr = port_ret(dropped), port_ret(kept)
        drop_rets.append(dr)
        keep_rets.append(kr)
        print(f"{win['name']:<8}{len(dropped):>6}  {dr:+9.1%}          {kr:+9.1%}")

    s = pd.Series(drop_rets).dropna()
    k = pd.Series(keep_rets).dropna()
    print(f"\n被剔除组合 平均年收益: {s.mean():+.1%}  |  保留组合: {k.mean():+.1%}")
    print(f"被剔除组合 胜率(>0): {(s > 0).mean():.0%}  |  保留组合: {(k > 0).mean():.0%}")


if __name__ == "__main__":
    main()
