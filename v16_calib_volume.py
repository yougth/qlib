"""校准qlib $close*$volume 与真实成交额的倍数关系"""
import sys, warnings, logging
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")
warnings.filterwarnings("ignore")
import pandas as pd

def main():
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    logging.getLogger('qlib.data.data').setLevel(logging.ERROR)
    import akshare as ak

    # 三只票各取2023-06一段, 对比akshare东财"成交额"(元)
    tests = [("SH600000", "600000"), ("SZ000651", "000651"), ("SH600519", "600519")]
    for qcode, acode in tests:
        q = D.features([qcode], ["$close", "$volume", "$factor"],
                       start_time="2023-06-01", end_time="2023-06-15")
        q = q.reset_index()
        try:
            a = ak.stock_zh_a_hist(symbol=acode, period="daily",
                                   start_date="20230601", end_date="20230615", adjust="")
        except Exception as e:
            print(f"{acode} akshare失败: {str(e)[:60]}")
            continue
        a["日期"] = pd.to_datetime(a["日期"])
        m = q.merge(a, left_on="datetime", right_on="日期")
        m["qlib_amt"] = m["$close"] * m["$volume"]
        m["ratio"] = m["成交额"] / m["qlib_amt"]
        print(f"{qcode}: 真实成交额/(qlib close*volume) 比值 = "
              f"{m['ratio'].median():.2f} (min {m['ratio'].min():.2f} max {m['ratio'].max():.2f})", flush=True)

if __name__ == "__main__":
    main()
