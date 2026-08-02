"""临时: DE/V20 权重扫描 + 组合分年收益 + 换手 + 2026年买入名单."""
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

    print('=== 1) 权重扫描 (DE 占比 0%~100%, 剔2020口径) ===')
    print('%-8s %8s %8s %8s %8s %8s' % ('DE占比', '年化', '剔2020', 'Sharpe', '回撤', 'Calmar'))
    for w in [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        r = w * r_de + (1 - w) * r_v
        m = calc_metrics(r, bench)
        ex = annualized_since(r, config.ALIGN_START)
        print('%-8s %7.1f%% %7.1f%% %8.2f %7.1f%% %8.2f' %
              (f'{int(w*100)}%', m['ar'] * 100, ex * 100, m['sharpe'], m['mdd'] * 100, m['calmar']))

    print()
    print('=== 2) 50/50 组合分年收益 ===')
    combo = 0.5 * r_de + 0.5 * r_v
    bench_bt = bench[bench.index >= BT]
    for label, r in [('DE Top10', r_de), ('value_comp', r_v), ('50/50组合', combo), ('沪深300', bench_bt)]:
        ys = {yr: (1 + g).prod() - 1 for yr, g in r.groupby(r.index.year)}
        print('%-10s' % label + ''.join('%9.1f%%' % (ys.get(y, float("nan")) * 100) for y in range(2020, 2027)))

    print()
    print('=== 3) 换手率: DE腿 %.1f%%/月, V20腿 %.1f%%/月, 组合加权 %.1f%%/月 ===' %
          (to_de * 100, to_v * 100, (to_de * 0.5 + to_v * 0.5) * 100))

    # 4) 2026 买入名单 (W2026 窗口: 信号 2025-12 ~ 2026-06)
    names = {}
    try:
        nm = pd.read_csv('/Users/11164591/Documents/Qoder目录/fcf_result.csv')
        names = {str(c).zfill(6): n for c, n in zip(nm['code'], nm['name'])}
    except Exception:
        pass

    def label(inst):
        code = inst[2:] if inst[:2] in ('SH', 'SZ') else inst
        return f"{inst}({names.get(code, '?')})"

    print()
    print('=== 4) 2026年周期(W2026窗口, 信号2025-12~2026-06) 每月买入名单 ===')
    for tag, topn in [('DoubleEnsemble', 10), ('VALUE20', 20)]:
        print(f'--- {tag} ---')
        for dt, tops in reb[tag]:
            if pd.Timestamp(dt) >= pd.Timestamp('2025-11-25'):
                print(' ', str(dt)[:10], ' '.join(label(i) for i in tops[:topn]))


if __name__ == '__main__':
    main()
