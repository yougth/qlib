"""临时脚本: DE 上线评估 — 与 value_comp 相关性/组合指标 + 成本压力测试."""
import pickle
import pandas as pd
from core import config
from core import data as datalayer
from core.backtest import portfolio_backtest, calc_metrics, annualized_since


def main():
    d = pickle.load(open('outputs/_phaseB_ckpt_tree.pkl', 'rb'))
    reb = d['rebalances']
    insts = sorted({i for t in ['DoubleEnsemble', 'VALUE20'] for _, tops in reb[t] for i in tops})
    datalayer.init_qlib()
    pm = datalayer.load_price_matrix(insts)
    bench = datalayer.load_benchmark()
    BT = pd.Timestamp(config.BT_START)
    r_de, to_de, _ = portfolio_backtest(reb['DoubleEnsemble'], pm)
    r_v, to_v, _ = portfolio_backtest(reb['VALUE20'], pm)
    r_de, r_v = r_de[r_de.index >= BT], r_v[r_v.index >= BT]
    print('=== DE vs value_comp 日收益相关性:', round(r_de.corr(r_v), 3))
    combo = (r_de + r_v) / 2
    for name, r in [('DE Top10        ', r_de), ('value_comp Top20', r_v), ('50/50 组合       ', combo)]:
        m = calc_metrics(r, bench)
        ex20 = annualized_since(r, config.ALIGN_START)
        print('%s 年化%6.1f%%  剔2020 %5.1f%%  Sharpe %.2f  回撤 %6.1f%%' %
              (name, m['ar'] * 100, ex20 * 100, m['sharpe'], m['mdd'] * 100))
    print()
    print('=== 成本压力测试(近似): 月换手82% x 12个月')
    for extra in [0.2, 0.4, 0.6]:
        drag = 0.82 * 12 * extra / 100
        print('  往返成本 0.4%%->%.1f%%: 年化拖累 -%.1fpt -> DE剔2020约 %.1f%%' %
              (0.4 + extra, drag * 100, 27.6 - drag * 100))


if __name__ == '__main__':
    main()
