"""跨进程数据集确定性检验: 构建 W2020 数据集并输出内容哈希."""
import hashlib
import numpy as np
from core import config
from core import data as datalayer
from core.universe import build_windows, build_dynamic_universe, format_qlib_code, load_pit_caches
from core.dataset import build_datasetH


def main():
    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    win = [w for w in build_windows() if w["name"] == "W2020"][0]
    codes = build_dynamic_universe(win["year"], fcf_df, profit_df)
    universe = [format_qlib_code(c) for c in codes]
    seg = {"train": win["train"], "valid": win["valid"], "test": win["test"]}
    ds = build_datasetH(seg, universe, ds_class="DatasetH")
    df = ds.prepare("train")
    h_idx = hashlib.md5(str(list(df.index[:500])).encode()).hexdigest()[:12]
    h_col = hashlib.md5(str(list(df.columns)).encode()).hexdigest()[:12]
    v = np.nan_to_num(df.values[:2000]).tobytes()
    h_val = hashlib.md5(v).hexdigest()[:12]
    print(f"shape={df.shape} idx={h_idx} col={h_col} val={h_val}")


if __name__ == '__main__':
    main()
