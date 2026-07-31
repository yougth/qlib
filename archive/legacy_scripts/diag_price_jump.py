import warnings
warnings.filterwarnings("ignore")
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.data import D

if __name__ == "__main__":
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    px = D.features(["SH600519", "SZ000858", "SH000300"], ["$close", "$factor"],
                    start_time="2020-09-18", end_time="2020-10-15")
    pd.set_option("display.width", 200)
    print(px)
    cal = D.calendar(start_time="2020-09-01", end_time="2020-10-15")
    print("calendar:", [str(d.date()) for d in cal])
