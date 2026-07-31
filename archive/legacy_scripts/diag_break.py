import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

if __name__ == "__main__":
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    # 1. 茅台 volume 是否也跳变
    px = D.features(["SH600519"], ["$close", "$volume"],
                    start_time="2020-09-18", end_time="2020-10-09")
    print(px)
    # 2. 断点普遍性: 抽样200只股票, 检查 09-25 -> 09-28 收益分布
    import os
    feat_dir = os.path.expanduser("~/.qlib/qlib_data/cn_data/features")
    insts = sorted(os.listdir(feat_dir))
    sample = [i.upper() for i in insts if i.startswith(("sh6", "sz0", "sz3"))][::15]
    px2 = D.features(sample, ["$close"], start_time="2020-09-25", end_time="2020-09-28")
    w = px2["$close"].unstack(level=0)
    if len(w) == 2:
        r = (w.iloc[1] / w.iloc[0]).dropna()
        print(f"\n抽样{len(r)}只 09-25->09-28 比率: 中位数={r.median():.3f}, "
              f"p5={r.quantile(0.05):.3f}, p95={r.quantile(0.95):.3f}")
        print(f"比率>1.5的占比: {(r > 1.5).mean()*100:.1f}%  比率<0.9占比: {(r < 0.9).mean()*100:.1f}%")
    # 3. 其他日期是否还有类似断点: 随机20只全历史 |日收益|>60% 的日期分布
    px3 = D.features(sample[:40], ["$close"], start_time="2014-01-01", end_time="2026-07-23")
    w3 = px3["$close"].unstack(level=0)
    rr = w3.pct_change()
    big = rr[rr.abs() > 0.6].stack()
    print("\n|日收益|>60% 的 (日期,只数):")
    print(big.groupby(level=0).size().sort_values(ascending=False).head(10))
    # 4. csi300_cache.csv 覆盖
    try:
        c = pd.read_csv("/Users/11164591/Documents/Qoder目录/csi300_cache.csv")
        print("\ncsi300_cache.csv 列:", c.columns.tolist(), "行数:", len(c))
        print(c.head(2))
        print(c.tail(2))
    except Exception as e:
        print("csi300_cache 读取失败:", e)
