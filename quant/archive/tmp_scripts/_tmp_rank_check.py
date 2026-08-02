"""临时: DE 排序能力分层检验 + 收益集中度检查 (回应'30.5%是否吓人')."""
import pickle
import pandas as pd
from core import config
from core import data as datalayer
from core.backtest import portfolio_backtest, calc_metrics, annualized_since


def main():
    d = pickle.load(open('outputs/_phaseB_ckpt_tree.pkl', 'rb'))
    reb = d['rebalances']
    legs = ['DoubleEnsemble', 'VALUE20'] + (['POOL_EW'] if 'POOL_EW' in reb else [])
    insts = sorted({i for t in legs for _, tops in reb[t] for i in tops})
    datalayer.init_qlib()
    pm = datalayer.load_price_matrix(insts)
    bench = datalayer.load_benchmark()
    BT = pd.Timestamp(config.BT_START)

    def slice_bt(tag, lo, hi):
        rl = [(dt, tops[lo:hi]) for dt, tops in reb[tag]]
        r, to, _ = portfolio_backtest(rl, pm)
        return r[r.index >= BT], to

    print('=== 1) DE 排序分层检验 (若排序有效应单调递减) ===')
    for name, lo, hi in [('第1-3名', 0, 3), ('第1-5名', 0, 5), ('第4-7名', 3, 7),
                         ('第6-10名', 5, 10), ('全部1-10', 0, 10)]:
        r, to = slice_bt('DoubleEnsemble', lo, hi)
        ex = annualized_since(r, config.ALIGN_START)
        m = calc_metrics(r, bench)
        print('%-10s 剔2020 %6.1f%%  Sharpe %5.2f  回撤 %6.1f%%' % (name, ex * 100, m['sharpe'], m['mdd'] * 100))

    print()
    print('=== 2) V20 排序分层检验 ===')
    for name, lo, hi in [('第1-10名', 0, 10), ('第11-20名', 10, 20)]:
        r, to = slice_bt('VALUE20', lo, hi)
        ex = annualized_since(r, config.ALIGN_START)
        m = calc_metrics(r, bench)
        print('%-10s 剔2020 %6.1f%%  Sharpe %5.2f  回撤 %6.1f%%' % (name, ex * 100, m['sharpe'], m['mdd'] * 100))

    print()
    print('=== 3) DE5+V10 50/50 组合收益集中度 (是否靠幸运月) ===')
    r_de5, _ = slice_bt('DoubleEnsemble', 0, 5)
    r_v10, _ = slice_bt('VALUE20', 0, 10)
    combo = 0.5 * r_de5 + 0.5 * r_v10
    mon = (1 + combo).resample('M').prod() - 1
    mon = mon[mon.index >= pd.Timestamp('2021-01-01')]  # 剔2020口径
    pos = (mon > 0).mean()
    print(f'月度胜率: {pos*100:.0f}%  ({(mon>0).sum()}/{len(mon)}个月)')
    print(f'最好月 {mon.max()*100:+.1f}%  最差月 {mon.min()*100:+.1f}%  月均 {mon.mean()*100:+.2f}%')
    top3 = mon.nlargest(3)
    total_log = (1 + mon).prod() - 1
    ex_top3 = (1 + mon.drop(top3.index)).prod() - 1
    n_yr = len(mon) / 12
    print(f'剔2020累计 {total_log*100:.0f}%;  去掉最好3个月后年化 {((1+ex_top3)**(1/n_yr)-1)*100:.1f}% (原 {((1+total_log)**(1/n_yr)-1)*100:.1f}%)')
    print('最好3个月:', ', '.join(f'{i.strftime("%Y-%m")} {v*100:+.1f}%' for i, v in top3.items()))

    print()
    print('=== 4) 对照: 同池等权基线 (超额是否来自选股) ===')
    # POOL_EW 在 ckpt 里
    r_pool, _, _ = portfolio_backtest(reb['POOL_EW'], pm) if 'POOL_EW' in reb else (None, None, None)
    if r_pool is not None:
        r_pool = r_pool[r_pool.index >= BT]
        exp = annualized_since(r_pool, config.ALIGN_START)
        exc = annualized_since(combo, config.ALIGN_START)
        print(f'池子等权 剔2020 {exp*100:.1f}%  vs 组合 {exc*100:.1f}%  → 选股超额 {exc*100-exp*100:.1f}pt/年')


if __name__ == '__main__':
    main()
