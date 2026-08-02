"""临时: 5万小资金方案测算 — 少持仓(10~15只)变体回测对比."""
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

    def leg(tag, topn):
        """截取每期前 topn 只重新回测(名单按分数排序)."""
        rl = [(dt, tops[:topn]) for dt, tops in reb[tag]]
        r, to, _ = portfolio_backtest(rl, pm)
        return r[r.index >= BT], to

    r_de10, to_de10 = leg('DoubleEnsemble', 10)
    r_de5, to_de5 = leg('DoubleEnsemble', 5)
    r_de4, to_de4 = leg('DoubleEnsemble', 4)
    r_v20, to_v20 = leg('VALUE20', 20)
    r_v10, to_v10 = leg('VALUE20', 10)
    r_v8, to_v8 = leg('VALUE20', 8)
    r_v5, to_v5 = leg('VALUE20', 5)

    print('=== 各腿单独表现(剔2020年化 / 月换手) ===')
    for n, (r, to) in [('DE Top10', (r_de10, to_de10)), ('DE Top5', (r_de5, to_de5)),
                       ('DE Top4', (r_de4, to_de4)), ('V Top20', (r_v20, to_v20)),
                       ('V Top10', (r_v10, to_v10)), ('V Top8', (r_v8, to_v8)),
                       ('V Top5', (r_v5, to_v5))]:
        ex = annualized_since(r, config.ALIGN_START)
        m = calc_metrics(r, bench)
        print('%-10s 剔2020 %6.1f%%  Sharpe %.2f  回撤 %6.1f%%  换手 %5.1f%%/月' %
              (n, ex * 100, m['sharpe'], m['mdd'] * 100, to * 100))

    print()
    print('=== 小资金组合方案对比 (5万, 目标10~15只, 换手≤50%) ===')
    plans = [
        ('DE5+V10 50/50 (15只)', 0.5, r_de5, to_de5, r_v10, to_v10),
        ('DE5+V10 40/60 (15只)', 0.4, r_de5, to_de5, r_v10, to_v10),
        ('DE5+V10 60/40 (15只)', 0.6, r_de5, to_de5, r_v10, to_v10),
        ('DE4+V8  50/50 (12只)', 0.5, r_de4, to_de4, r_v8, to_v8),
        ('DE5+V5  50/50 (10只)', 0.5, r_de5, to_de5, r_v5, to_v5),
        ('纯V10       (10只)', 0.0, r_de5, to_de5, r_v10, to_v10),
        ('[对照]DE10+V20 50/50', 0.5, r_de10, to_de10, r_v20, to_v20),
    ]
    print('%-22s %8s %7s %8s %8s %10s' % ('方案', '剔2020', 'Sharpe', '回撤', 'Calmar', '组合换手'))
    best = None
    for name, w, ra, ta, rb_, tb in plans:
        r = w * ra + (1 - w) * rb_
        to = w * ta + (1 - w) * tb
        ex = annualized_since(r, config.ALIGN_START)
        m = calc_metrics(r, bench)
        print('%-22s %7.1f%% %7.2f %7.1f%% %8.2f %8.1f%%/月' %
              (name, ex * 100, m['sharpe'], m['mdd'] * 100, m['calmar'], to * 100))
        if to <= 0.52 and '对照' not in name and (best is None or ex > best[1]):
            best = (name, ex, r)

    print()
    name, ex, r = best
    print(f'=== 换手≤50%中收益最高: {name} 分年收益 ===')
    ys = {yr: (1 + g).prod() - 1 for yr, g in r.groupby(r.index.year)}
    bb = bench[bench.index >= BT]
    yb = {yr: (1 + g).prod() - 1 for yr, g in bb.groupby(bb.index.year)}
    print('年份    ' + ''.join('%9d' % y for y in range(2020, 2027)))
    print('组合    ' + ''.join('%8.1f%%' % (ys.get(y, float("nan")) * 100) for y in range(2020, 2027)))
    print('沪深300 ' + ''.join('%8.1f%%' % (yb.get(y, float("nan")) * 100) for y in range(2020, 2027)))


if __name__ == '__main__':
    main()
