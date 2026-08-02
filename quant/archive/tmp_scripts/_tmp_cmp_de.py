"""deA vs deB 一致性 + 新旧 DE 种子敏感度比对."""
import pickle
import pandas as pd


def load(tag):
    d = pickle.load(open(f'outputs/_phaseB_ckpt_{tag}.pkl', 'rb'))
    return d['rebalances']['DoubleEnsemble']


def main():
    a, b = load('deC'), load('deD')
    old = load('tree')
    # 1) deA vs deB 完全一致性
    same = len(a) == len(b) and all(
        da == db_ and la == lb for (da, la), (db_, lb) in zip(a, b))
    print(f'1) deC vs deD: 调仓期数 {len(a)} vs {len(b)}, 逐月名单完全一致 = {same}')
    if not same:
        for (da, la), (db_, lb) in zip(a, b):
            if la != lb:
                print('   首个差异:', da, set(la) ^ set(lb))
                break
    # 2) 新 vs 旧名单重合度
    ov = [len(set(la) & set(lo)) / len(lo) for (_, la), (_, lo) in zip(a, old)]
    print(f'2) 新 vs 旧: 期数 {len(a)} vs {len(old)}, 月均名单重合度 {sum(ov)/len(ov)*100:.0f}%')
    # 3) 指标对比
    for tag in ['deC', 'deD']:
        df = pd.read_csv(f'outputs/benchmark_phaseB_{tag}.csv', index_col=0)
        print(f'--- {tag} ---')
        print(df.loc[['DoubleEnsemble']].to_string())
    df = pd.read_csv('outputs/benchmark_phaseB_tree.csv', index_col=0)
    print('--- 旧(tree) ---')
    print(df.loc[['DoubleEnsemble']].to_string())


if __name__ == '__main__':
    main()
