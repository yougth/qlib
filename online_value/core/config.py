"""
口径常量与路径 —— 改这里等于改策略
==================================================================
本文件是 value_comp_M20 上线策略的唯一"口径来源"。
任何一个常量的改动都等价于换了一个策略, 必须:
  1) 先跑 run_backtest.py 确认 OOS 不下降
  2) 再 preflight_check.py --relock 重建锁
否则 preflight 的"口径常量锁"会直接 FAIL 并禁止交易。
"""
import os

# ---------- 路径 ----------
# 数据根目录: 存 valuation_cache.csv / fcf_cache_pit.csv / profit_cache_pit.csv / csi300_cache.csv
DATA_DIR = "/Users/11164591/Documents/Qoder目录"
# 上线目录 (core/ 的上一级), 所有产出写在这里
ONLINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(ONLINE_DIR, "golden")
QLIB_PROVIDER = "~/.qlib/qlib_data/cn_data"

FCF_CACHE = f"{DATA_DIR}/fcf_cache_pit.csv"
PROFIT_CACHE = f"{DATA_DIR}/profit_cache_pit.csv"
VAL_CACHE = f"{DATA_DIR}/valuation_cache.csv"
BENCH_CACHE = f"{DATA_DIR}/csi300_cache.csv"

# ---------- 策略口径 (锁定, 与 v18 回测一致) ----------
TOPK = 20                        # 持仓只数, 等权 (每只 5%)
N_BACKUP = 5                     # 买入清单附带的备选只数(涨停/停牌顺延用)
# qlib $volume 单位为手(x100股), 真实500万元成交额 → qlib口径 5万
LIQ_THRESHOLD = 5_000_000 / 100
FEE_ROUNDTRIP = 0.004            # 往返交易成本 0.4% (单边换手 x 该值)
VALUE_FACTORS = ["ep", "bp", "cfp", "sp"]   # value_comp 的 4 个估值倒数
# 仅用于回测里的 IC 诊断(未来60日超额), 不参与选股, 改它不影响持仓
FWD_DAYS = 60

# ---------- 回测区间 (冻结: 这是回归基线的定义, 不要随便动) ----------
CAL_START = "2013-01-01"
BT_END = "2026-07-31"            # 回测末端; 改它 = 换基线, 必须重建 golden/
YEARS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]   # 2020 跳过(疫情异常年)
IS_YEARS = {2019, 2021, 2022}
OOS_YEARS = {2023, 2024, 2025, 2026}

# ---------- 参考: 冻结基线指标 (run_backtest.py 必须复现) ----------
BASELINE = dict(ar=0.2574751797, sharpe=1.2648900946, max_dd=-0.1990658615,
                trades=262, is_ar=0.2168505378, oos_ar=0.2928053443)

# ---------- 仅用于统计占位, 不参与选股 ----------
# 8只金融(银行+保险): 回测已验证"剔除金融会显著变差", 因此上线不剔除, 只做占位监控
FIN_SET = {"SH601318", "SH601336", "SH601398", "SH601601",
           "SH601665", "SH601838", "SH601939", "SH601988"}


def init_qlib():
    """统一的 qlib 初始化 + 静音噪声日志。所有入口脚本第一步都调它。"""
    import warnings, logging
    warnings.filterwarnings("ignore")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    logging.getLogger("qlib.data.data").setLevel(logging.ERROR)
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=QLIB_PROVIDER, region=REG_CN)
