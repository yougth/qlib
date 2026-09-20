#!/usr/bin/env python3
"""
fetch_hk_data —— 港股行情 + 基本面数据抓取 (腾讯行情 + akshare财报)
================================================================================
将港股加入量化选股池:
  1. 行情: 腾讯 hk{code} 接口, 后复权(hfq), 2014~2026
  2. 基本面: akshare stock_financial_hk_report_em (利润表+现金流量表)
  3. 构建 qlib bin + fcf/profit 缓存

港股代码格式: 5位数字 (00700, 09988), qlib instrument 前缀 "hk"

用法:
  python3 tools/fetch_hk_data.py --phase quotes    # 抓行情
  python3 tools/fetch_hk_data.py --phase fundamentals  # 抓基本面
  python3 tools/fetch_hk_data.py --phase build     # 构建 qlib bin
"""
import os, sys, time, json, argparse
import urllib.request
import numpy as np
import pandas as pd

QUANT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("QUANT_DATA_DIR") or os.path.join(QUANT_DIR, "..", "..")
QLIB_DATA = os.path.join(os.path.dirname(QUANT_DIR), "data_cache", "qlib_cn_tencent")
HK_CACHE = os.path.join(DATA_DIR, "hk_stock_list.csv")
HK_QUOTES_DIR = os.path.join(os.path.dirname(QUANT_DIR), "data_cache", "hk_quotes")
HK_FCF_CACHE = os.path.join(DATA_DIR, "fcf_cache_hk.csv")
HK_PROFIT_CACHE = os.path.join(DATA_DIR, "profit_cache_hk.csv")

HEADERS = {"Referer": "https://gu.qq.com", "User-Agent": "Mozilla/5.0"}

# 候选港股: 大中盘蓝筹 + 红筹 + H股 (流动性好, 财报完整)
# 覆盖金融/科技/能源/消费/地产/医药 主要行业
HK_STOCKS = [
    ("00700", "腾讯控股"), ("09988", "阿里巴巴-SW"), ("00388", "香港交易所"),
    ("00941", "中国移动"), ("00005", "汇丰控股"), ("00883", "中海油"),
    ("01038", "长江实业集团"), ("00939", "建设银行"), ("02318", "中国平安"),
    ("01398", "工商银行"), ("03988", "中国银行"), ("00386", "中国石化"),
    ("00857", "中国石油股份"), ("01088", "中国神华"), ("01299", "友邦保险"),
    ("00011", "恒生银行"), ("00001", "长和"), ("00002", "中电控股"),
    ("00003", "香港中华煤气"), ("00006", "电能实业"), ("00012", "恒基地产"),
    ("00016", "新鸿基地产"), ("00017", "新世界发展"), ("00027", "银河娱乐"),
    ("00066", "港铁公司"), ("00101", "恒隆地产"), ("00151:0", "中国旺旺"),
    ("00175", "吉利汽车"), ("00267", "中信股份"), ("00268", "金蝶国际"),
    ("00285", "比亚迪电子"), ("00288", "华润燃气"), ("00291", "华润啤酒"),
    ("00316", "东方海外国际"), ("00338", "上海石油化工股份"), ("00354", "软件测试"),
    ("00390", "中国中铁"), ("00590", "六福集团"), ("00688", "中国海外发展"),
    ("00763", "中兴通讯"), ("00823", "领展房产基金"), ("00868", "信义玻璃"),
    ("00992", "联想集团"), ("01024", "恒安国际"), ("01044", "恒腾网络"),
    ("01099", "太平洋航运"), ("01109", "华润置地"), ("01113", "长实地产"),
    ("01177", "中国建材"), ("01211", "比亚迪股份"), ("01288", "农业银行"),
    ("01336", "新华保险"), ("01339", "中国太平洋保险"), ("01368", "特步国际"),
    ("01513", "丽珠医药"), ("01800", "中国交通建设"), ("01812", "晨鸣纸业"),
    ("01876", "哔哩哔哩-SW"), ("01928", "金沙中国有限公司"), ("01997:0", "众安在线"),
    ("02013", "微盟集团"), ("02018", "瑞声科技"), ("02020", "安踏体育"),
    ("02238:0", "广发证券"), ("02313", "申洲国际"), ("02331", "李宁"),
    ("02382:0", "舜宇光学科技"), ("02601:0", "中国太保"), ("02688:0", "新奥能源"),
    ("02689:0", "玖龙纸业"), ("02899:0", "紫金矿业"), ("03323:0", "中国建材"),
    ("03328:0", "交通银行"), ("03618:0", "中国金融租赁"), ("03692:0", "康希诺生物"),
    ("03799:0", "达利食品"), ("03833:0", "新疆新鑫矿业"), ("03888:0", "金山软件"),
    ("03908:0", "中金公司"), ("03968:0", "招商银行"), ("03988:0", "中国银行"),
    ("06030:0", "中信证券"), ("06060:0", "众安在线财产保险"), ("06098:0", "碧桂园服务"),
    ("06126:0", "期货通"), ("06618:0", "京东物流"), ("06862:0", "海底捞"),
    ("06888:0", "中国再保险"), ("09618:0", "京东健康"), ("09633:0", "农夫山泉"),
    ("09888:0", "百度集团-SW"), ("09999:0", "网易-S"),
]

def clean_code(code):
    """清理代码: 去掉冒号后缀, 补零到5位"""
    code = code.split(":")[0]
    return code.zfill(5)

def fetch_hk_kline(code, start="2014-01-01", end=None):
    """腾讯港股后复权K线, 返回 DataFrame
    港股返回9列: [日期, 开, 收, 高, 低, 成交量, {}, 复权因子, 成交额]
    腾讯单次最多640条(~2.5年), 需分段抓取拼成完整序列
    """
    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
    code5 = clean_code(code)
    all_rows = []
    # 分段: 每2年一段
    from datetime import datetime, timedelta
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    seg_start = s
    while seg_start < e:
        seg_end = min(seg_start + timedelta(days=730), e)
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?"
               f"param=hk{code5},day,{seg_start.strftime('%Y-%m-%d')},"
               f"{seg_end.strftime('%Y-%m-%d')},640,hfq")
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            node = data.get("data", {}).get(f"hk{code5}", {})
            # 无除权除息的股票腾讯不返回 hfqday 而返回原始 day (实测微盟/众安/京东物流),
            # 此时原始价即后复权价, 可作等价回退; day 行尾多 2 列, 截断对齐 9 列
            kline = node.get("hfqday") or node.get("day") or []
            kline = [r[:9] for r in kline]
            if kline:
                all_rows.extend(kline)
        except Exception:
            pass
        time.sleep(0.2)
        seg_start = seg_end + timedelta(days=1)

    if not all_rows:
        return pd.DataFrame()

    # 9列: date, open, close, high, low, volume, {}, factor_ratio, amount
    df = pd.DataFrame(all_rows, columns=["date", "open", "close", "high", "low",
                                          "volume", "_extra", "factor_ratio", "amount"])
    df = df.drop_duplicates(subset=["date"])
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    # 港股后复权: close 已是后复权价
    df["factor"] = pd.to_numeric(df["factor_ratio"], errors="coerce").fillna(1.0)
    df = df.drop(columns=["_extra"])
    return df

def phase_quotes():
    """抓取港股行情"""
    os.makedirs(HK_QUOTES_DIR, exist_ok=True)
    stocks = [(clean_code(c), n) for c, n in HK_STOCKS]
    # 去重
    seen = set()
    stocks = [(c, n) for c, n in stocks if c not in seen and not seen.add(c)]
    print(f"[Phase quotes] {len(stocks)} 只港股", flush=True)

    ok, fail, refreshed = 0, 0, 0
    for i, (code, name) in enumerate(stocks):
        cache = os.path.join(HK_QUOTES_DIR, f"hk{code}.csv")
        # 全量刷新: hfq 后复权历史会随新除权除息重算, 增量拼接会混价格基准,
        # 且旧逻辑"文件存在就跳过"导致行情永远停在首次抓取日 (曾停 2026-07-23)
        try:
            df = fetch_hk_kline(code)
            if len(df) > 100:
                df.to_csv(cache, index=False)
                ok += 1
                refreshed += 1
                print(f"  [{i+1}/{len(stocks)}] hk{code} {name}: {len(df)}条 ✓ "
                      f"({df['date'].min().date()}~{df['date'].max().date()})", flush=True)
            else:
                fail += 1
                print(f"  [{i+1}/{len(stocks)}] hk{code} {name}: {len(df)}条 ✗", flush=True)
        except Exception as e:
            fail += 1
            print(f"  [{i+1}/{len(stocks)}] hk{code} {name}: ❌ {e}", flush=True)
        time.sleep(0.4)
    print(f"\n[完成] ok={ok} fail={fail} (刷新 {refreshed})", flush=True)

def phase_fundamentals():
    """抓取港股基本面 (利润表+现金流表)"""
    import akshare as ak
    stocks = [(clean_code(c), n) for c, n in HK_STOCKS]
    seen = set()
    stocks = [(c, n) for c, n in stocks if c not in seen and not seen.add(c)]
    print(f"[Phase fundamentals] {len(stocks)} 只港股", flush=True)

    fcf_rows, profit_rows = [], []
    ok, fail = 0, 0
    for i, (code, name) in enumerate(stocks):
        try:
            # 利润表
            df_p = ak.stock_financial_hk_report_em(stock=code, symbol="利润表", indicator="年度")
            np_data = df_p[df_p["STD_ITEM_NAME"] == "股东应占溢利"]
            for _, r in np_data.iterrows():
                yr = pd.to_datetime(r["REPORT_DATE"]).year
                profit_rows.append({"code": code, "year": yr, "net_profit": float(r["AMOUNT"])})

            # 现金流量表
            df_c = ak.stock_financial_hk_report_em(stock=code, symbol="现金流量表", indicator="年度")
            ocf_data = df_c[df_c["STD_ITEM_NAME"] == "经营业务现金净额"]
            for _, r in ocf_data.iterrows():
                yr = pd.to_datetime(r["REPORT_DATE"]).year
                fcf_rows.append({"code": code, "year": yr, "fcf": float(r["AMOUNT"])})

            ok += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(stocks)}] ok={ok} fail={fail}", flush=True)
        except Exception as e:
            fail += 1
            print(f"  [{i+1}/{len(stocks)}] {code} {name}: ❌ {e}", flush=True)
        time.sleep(0.5)

    fcf_df = pd.DataFrame(fcf_rows)
    prof_df = pd.DataFrame(profit_rows)
    if len(fcf_df):
        fcf_df.to_csv(HK_FCF_CACHE, sep="\t", index=False)
        print(f"\n[FCF] {fcf_df.shape}, {fcf_df['code'].nunique()}只, "
              f"{fcf_df['year'].min()}~{fcf_df['year'].max()}", flush=True)
    if len(prof_df):
        prof_df.to_csv(HK_PROFIT_CACHE, sep="\t", index=False)
        print(f"[利润] {prof_df.shape}, {prof_df['code'].nunique()}只, "
              f"{prof_df['year'].min()}~{prof_df['year'].max()}", flush=True)
    print(f"ok={ok} fail={fail}", flush=True)

def phase_build():
    """构建港股 qlib bin (合入现有 qlib_cn_tencent)

    qlib bin 标准格式 (同 scripts/dump_bin.py): 首 float = 起始日历索引,
    其后为特征值 (小端 float32)。旧版直接 arr.tofile() 缺头, 导致 qlib 把
    首个数据值误读为起始索引 (close 311.28→311, volume 0→0, factor 1→1),
    各字段错位量不同 → 表达式广播崩溃。

    factor 说明: 腾讯接口 factor_ratio 列已污染 (每日随机 0.002~1.55, 中位 0.003),
    恒写 1.0 —— $close 已是后复权价, 收益/特征计算不受影响; 实盘下单真实价由
    monthly_signal.fetch_hk_raw_close 在线抓取, 不依赖 $factor。
    """
    cal_path = os.path.join(QLIB_DATA, "calendars", "day.txt")
    cal = open(cal_path).read().strip().split()
    cal_dates = pd.to_datetime(cal)

    feats_dir = os.path.join(QLIB_DATA, "features")
    instruments_dir = os.path.join(QLIB_DATA, "instruments")
    os.makedirs(instruments_dir, exist_ok=True)

    stocks = [(clean_code(c), n) for c, n in HK_STOCKS]
    seen = set()
    stocks = [(c, n) for c, n in stocks if c not in seen and not seen.add(c)]

    inst_rows = []
    ok, align_bad = 0, 0
    for code, name in stocks:
        cache = os.path.join(HK_QUOTES_DIR, f"hk{code}.csv")
        if not os.path.exists(cache):
            continue
        df = pd.read_csv(cache, parse_dates=["date"])
        if len(df) < 100:
            continue

        n_src = len(df)
        df = df.sort_values("date").drop_duplicates("date").set_index("date")
        # 对齐到 A 股日历: 港股独有交易日被丢弃(极少), A 股独有交易日为 NaN
        df = df.reindex(cal_dates)
        close = df["close"].values.astype(np.float32)
        volume = df["volume"].values.astype(np.float32)
        open_ = df["open"].values.astype(np.float32)
        high = df["high"].values.astype(np.float32)
        low = df["low"].values.astype(np.float32)
        factor = np.where(np.isfinite(close), 1.0, np.nan).astype(np.float32)  # 恒写1.0, 见 docstring

        inst = f"hk{code}"
        out_dir = os.path.join(feats_dir, inst)
        os.makedirs(out_dir, exist_ok=True)
        for fname, arr in [("close.day.bin", close), ("factor.day.bin", factor),
                           ("volume.day.bin", volume), ("open.day.bin", open_),
                           ("high.day.bin", high), ("low.day.bin", low)]:
            # 标准格式: 首元素 = 该字段首个有效值所在的日历索引
            valid_pos = np.where(np.isfinite(arr))[0]
            if len(valid_pos) == 0:
                continue
            start = int(valid_pos[0])
            np.hstack([np.float32(start), arr[start:].astype("<f4")]).astype("<f4")\
                .tofile(os.path.join(out_dir, fname))

        # 对齐体检: 落在日历内的有效 close 行数应≈源行数 (差异=港股独有交易日)
        n_aligned = int(np.isfinite(close).sum())
        if n_aligned < n_src * 0.95:
            align_bad += 1
            print(f"  [WARN] {inst} 对齐率低: 日历内有效 {n_aligned} / 源 {n_src}", flush=True)

        first_date = df["close"].first_valid_index()
        last_date = df["close"].last_valid_index()
        if first_date is not None and last_date is not None:
            inst_rows.append(f"{inst}\t{first_date.strftime('%Y-%m-%d')}\t{last_date.strftime('%Y-%m-%d')}")
            ok += 1

    # all.txt: 先剔除旧 hk 行 (防重复 + 修正日期范围), 再追加
    inst_path = os.path.join(instruments_dir, "all.txt")
    keep = []
    if os.path.exists(inst_path):
        with open(inst_path) as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip() and not line.split("\t")[0].strip().lower().startswith("hk"):
                    keep.append(line)
    with open(inst_path, "w") as f:
        for line in keep:
            f.write(line + "\n")
        for row in inst_rows:
            f.write(row + "\n")
    print(f"[build] {ok} 只港股 bin 已写入 {feats_dir} (对齐告警 {align_bad} 只)", flush=True)
    print(f"  instruments: A股 {len(keep)} 行 + 港股 {len(inst_rows)} 行 → {inst_path}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["quotes", "fundamentals", "build"], required=True)
    args = ap.parse_args()
    if args.phase == "quotes":
        phase_quotes()
    elif args.phase == "fundamentals":
        phase_fundamentals()
    elif args.phase == "build":
        phase_build()
