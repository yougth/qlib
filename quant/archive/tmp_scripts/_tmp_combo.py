"""临时: DE/XGB/V20 相关性矩阵与组合方案对比."""
import pickle
import pandas as pd
from core import config
from core import data as datalayer
from core.backtest import portfolio_backtest, calc_metrics, annualized_since


def main():
    d = pickle.load(open('outputs/_phaseB_ckpt_tree.pkl', 'rb'))
    reb = d['rebalances']
    names = ['DoubleEnsemble', 'XGBoost', 'VALUE20']
    insts = sorted({i for t in names for _, tops in reb[t] for i in tops})
    datalayer.init_qlib()
    pm = datalayer.load_price_matrix(insts)
    bench = datalayer.load_benchmark()
    BT = pd.Timestamp(config.BT_START)
    rets = {}
    for t in names:
        r, _, _ = portfolio_backtest(reb[t], pm)
        rets[t] = r[r.index >= BT]
    df = pd.DataFrame(rets)
    print('=== 日收益相关性矩阵 ===')
    print(df.corr().round(3))
    print()
    combos = {
        'DE 100%':            df['DoubleEnsemble'],
        'DE50 + V20 50':      0.5*df['DoubleEnsemble'] + 0.5*df['VALUE20'],
        'DE40 + XGB20 + V40': 0.4*df['DoubleEnsemble'] + 0.2*df['XGBoost'] + 0.4*df['VALUE20'],
        'DE33+XGB33+V33':     (df['DoubleEnsemble'] + df['XGBoost'] + df['VALUE20'])/3,
    }
    print('=== 组合方案对比 ===')
    for name, r in combos.items():
        m = calc_metrics(r, bench)
        ex = annualized_since(r, config.ALIGN_START)
        print('%-20s 年化%6.1f%%  剔2020 %5.1f%%  Sharpe %.2f  回撤 %6.1f%%  Calmar %.2f' %
              (name, m['ar']*100, ex*100, m['sharpe'], m['mdd']*100, m['calmar']))


if __name__ == '__main__':
    main()
