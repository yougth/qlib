"""DE 复现性鉴别: 同进程连训两次 W2020, 输出预测哈希."""
import hashlib
from core import config
from core import data as datalayer
from core.universe import build_windows, build_dynamic_universe, format_qlib_code, load_pit_caches
from core.dataset import build_datasetH
from core.models import train_qlib_model


def main():
    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    win = [w for w in build_windows() if w["name"] == "W2020"][0]
    codes = build_dynamic_universe(win["year"], fcf_df, profit_df)
    universe = [format_qlib_code(c) for c in codes]
    seg = {"train": win["train"], "valid": win["valid"], "test": win["test"]}
    for run in (1, 2):
        ds = build_datasetH(seg, universe, ds_class="DatasetH")
        _, pred = train_qlib_model("DoubleEnsemble", ds, seed_key="DoubleEnsemble:W2020")
        h = hashlib.md5(pred.round(8).to_csv().encode()).hexdigest()[:12]
        top = pred.groupby(level=0).rank(ascending=False)
        first = pred.index.get_level_values(0).min()
        top10 = pred.loc[first].nlargest(10).index.tolist()
        print(f"[run{run}] pred_hash={h} 首日Top10={top10}", flush=True)
        del ds


if __name__ == '__main__':
    main()
