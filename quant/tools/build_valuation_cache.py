#!/usr/bin/env python3
"""
build_valuation_cache.py — 从新浪财务指标页抓每股收益/每股净资产/每股经营现金流/销售净利率,
用 real_price / 每股值 计算 PE/PB/PCF/PS, 生成 PIT 正确的 valuation_cache.csv

核心公式 (无需总股本, 无需市值):
  PE  = real_price / EPS      (摊薄每股收益)
  PB  = real_price / BPS      (每股净资产)
  PCF = real_price / OCFPS    (每股经营性现金流)
  PS  = PE / profit_margin    (销售净利率)

数据来源:
  - 每股指标: 新浪 vFD_FinancialGuideLine (HTML, 每年1页)
  - 真实价:  qlib_cn_tencent bin (close/factor)

PIT 规则:
  - 年报Y 在 Y+1年5月1日后可用
  - 信号日D >= 5月1日Y → 用 Y-1 年报
  - 信号日D <  5月1日Y → 用 Y-2 年报

用法:
  PYTHONPATH=. python3 tools/build_valuation_cache.py fetch   # 抓取
  PYTHONPATH=. python3 tools/build_valuation_cache.py build   # 构建
"""
import os, sys, re, time
import numpy as np
import pandas as pd
import urllib.request

QUANT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("QUANT_DATA_DIR") or os.path.join(QUANT_DIR, "..", "..")
QLIB_DATA = os.path.join(QUANT_DIR, "data_cache", "qlib_cn_tencent")

PERSHARE_CACHE = os.path.join(DATA_DIR, "pershare_cache_pit.csv")
VAL_CACHE = os.path.join(DATA_DIR, "valuation_cache.csv")
FCF_CACHE = os.path.join(DATA_DIR, "fcf_cache_pit.csv")
PROFIT_CACHE = os.path.join(DATA_DIR, "profit_cache_pit.csv")

SINA_HEADERS = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
FETCH_YEARS = list(range(2016, 2026))   # 2016-2025
POOL_RANGE = range(2019, 2027)          # 信号年 2019-2026
SLEEP_BETWEEN = 0.4                     # 秒, urllib 不被限流

# ============================================================
#  股票池
# ============================================================
def get_pool_union():
    sys.path.insert(0, QUANT_DIR)
    from core.universe import build_dynamic_universe
    fcf = pd.read_csv(FCF_CACHE, sep='\t', dtype={"code": str})
    prof = pd.read_csv(PROFIT_CACHE, sep='\t', dtype={"code": str})
    fcf["code"] = fcf["code"].str.zfill(6)
    prof["code"] = prof["code"].str.zfill(6)
    union = set()
    for y in POOL_RANGE:
        union |= set(build_dynamic_universe(y, fcf, prof))
    return sorted(union)

# ============================================================
#  新浪财务指标页抓取
# ============================================================
def _sina_html(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=SINA_HEADERS)
            resp = urllib.request.urlopen(req, timeout=12)
            html = resp.read().decode("gb18030", errors="replace")
            if len(html) > 500:
                return html
        except Exception:
            pass
        time.sleep(2.0)
    return ""

def _parse_table_rows(html):
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.S)
    result = []
    for r in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', r, re.S)
        cells = [re.sub(r'<[^>]+>', '', c).strip().replace('&nbsp;', '') for c in cells]
        cells = [c for c in cells if c]
        if cells:
            result.append(cells)
    return result

def fetch_guide(code, year):
    """新浪财务指标 → {date_str: {eps, bps, ocfps, npm}} (季度累计/时点)."""
    url = f"https://money.finance.sina.com.cn/corp/go.php/vFD_FinancialGuideLine/stockid/{code}/ctrl/{year}/displaytype/4.phtml"
    html = _sina_html(url)
    if not html:
        return {}

    dates_raw = re.findall(r'20\d{2}-\d{2}-\d{2}', html)
    dates = dates_raw[1:5] if len(dates_raw) >= 5 else dates_raw[1:]
    rows = _parse_table_rows(html)
    result = {}
    for cells in rows:
        if not cells:
            continue
        label = cells[0]
        for i, d in enumerate(dates):
            if i + 1 >= len(cells):
                break
            val = cells[i + 1].replace(',', '').replace('-', '').strip()
            try:
                val_float = float(val) if val else None
            except ValueError:
                continue
            if val_float is None:
                continue
            if '摊薄每股收益' in label:
                result.setdefault(d, {})['eps'] = val_float
            elif '每股净资产' in label and '调整' in label and '后' in label:
                result.setdefault(d, {})['bps'] = val_float
            elif '每股净资产' in label and '调整前' in label:
                # 调整前优先; 如果已有调整后则不覆盖
                if 'bps' not in result.setdefault(d, {}):
                    result[d]['bps'] = val_float
            elif '每股经营性现金流' in label:
                result.setdefault(d, {})['ocfps'] = val_float
            elif '销售净利率' in label:
                result.setdefault(d, {})['npm'] = val_float
    return result

# ============================================================
#  Phase 1: Fetch
# ============================================================
def phase_fetch():
    codes = get_pool_union()
    print(f"[Phase 1] 池成员并集: {len(codes)} 只, 每只 {len(FETCH_YEARS)} 页", flush=True)

    # 加载已完成
    done = set()
    if os.path.exists(PERSHARE_CACHE):
        df = pd.read_csv(PERSHARE_CACHE, sep='\t', dtype={"code": str})
        done = set(df["code"].str.zfill(6).unique())
    print(f"  已完成: {len(done)} 只", flush=True)

    buf = []
    t0 = time.time()
    n_fail = 0

    for idx, code in enumerate(codes):
        if code in done:
            continue

        code_rows = []
        for yr in FETCH_YEARS:
            data = fetch_guide(code, yr)
            if data:
                for d, v in data.items():
                    row = {"code": code, "report_date": d}
                    row.update(v)
                    code_rows.append(row)
            time.sleep(SLEEP_BETWEEN)

        if code_rows:
            buf.extend(code_rows)
            done.add(code)
        else:
            n_fail += 1
            print(f"  [FAIL] {code}: 0 rows", flush=True)

        # 每 20 只 flush
        if (idx + 1) % 20 == 0 or idx == len(codes) - 1:
            if buf:
                df = pd.DataFrame(buf)
                if not os.path.exists(PERSHARE_CACHE):
                    df.to_csv(PERSHARE_CACHE, sep='\t', index=False)
                else:
                    df.to_csv(PERSHARE_CACHE, sep='\t', index=False, mode='a', header=False)
                buf = []
            elapsed = time.time() - t0
            n_done = len(done)
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(codes) - n_done) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(codes)}] ok={n_done} fail={n_fail} "
                  f"{elapsed:.0f}s ETA {eta:.0f}s", flush=True)

    # 汇总
    if os.path.exists(PERSHARE_CACHE):
        df = pd.read_csv(PERSHARE_CACHE, sep='\t', dtype={"code": str})
        df["code"] = df["code"].str.zfill(6)
        print(f"\n[完成] pershare_cache_pit.csv: {df.shape}, "
              f"{df['code'].nunique()} 只, {df['report_date'].nunique()} 季度", flush=True)

# ============================================================
#  Phase 2: Build valuation_cache.csv
# ============================================================
def _pit_year(date_str):
    """信号日 → 可用年报年份. 5月1日前用 Y-2, 之后用 Y-1."""
    d = date_str.replace('-', '') if '-' in date_str else date_str
    y = int(d[:4]); m = int(d[4:6])
    return y - 1 if m >= 5 else y - 2

def phase_build():
    print("[Phase 2] 构建 valuation_cache.csv", flush=True)

    # 日历
    cal = open(os.path.join(QLIB_DATA, "calendars", "day.txt")).read().strip().split()
    print(f"  日历: {len(cal)} 天 {cal[0]}~{cal[-1]}", flush=True)

    # 加载每股指标 (只取 Q4 = 年报)
    ps_df = pd.read_csv(PERSHARE_CACHE, sep='\t', dtype={"code": str})
    ps_df["code"] = ps_df["code"].str.zfill(6)
    ps_df["report_date"] = ps_df["report_date"].astype(str)
    # 取 12-31 行
    annual = ps_df[ps_df["report_date"].str.endswith("12-31")].copy()
    annual["year"] = annual["report_date"].str[:4].astype(int)
    # 建 {code: {year: {eps, bps, ocfps, npm}}}
    pershare = {}
    for code, g in annual.groupby("code"):
        pershare[code] = {}
        for _, row in g.iterrows():
            pershare[code][int(row["year"])] = {
                "eps": row.get("eps"),
                "bps": row.get("bps"),
                "ocfps": row.get("ocfps"),
                "npm": row.get("npm"),
            }
    print(f"  每股指标(年报): {len(pershare)} 只", flush=True)

    # 股票池
    codes = get_pool_union()
    print(f"  池成员: {len(codes)} 只", flush=True)

    # 逐股票构建日频估值
    feats_dir = os.path.join(QLIB_DATA, "features")
    out_rows = []
    n_ok = 0; n_skip = 0

    for ci, code6 in enumerate(codes):
        prefix = "sh" if code6.startswith(("6", "5", "9")) else "sz"
        inst = f"{prefix}{code6}"
        feat_path = os.path.join(feats_dir, inst)
        if not os.path.isdir(feat_path):
            n_skip += 1
            continue

        close = np.fromfile(os.path.join(feat_path, "close.day.bin"), dtype='<f4')
        factor = np.fromfile(os.path.join(feat_path, "factor.day.bin"), dtype='<f4')
        if len(close) < len(cal):
            pad = len(cal) - len(close)
            close = np.concatenate([np.full(pad, np.nan), close])
            factor = np.concatenate([np.full(pad, np.nan), factor])

        ps_yr = pershare.get(code6, {})
        if not ps_yr:
            n_skip += 1
            continue

        code_rows = []
        for i, date_str in enumerate(cal):
            if np.isnan(close[i]) or np.isnan(factor[i]) or factor[i] <= 0:
                continue
            real_price = close[i] / factor[i]

            fy = _pit_year(date_str)
            d = ps_yr.get(fy)
            if not d:
                continue

            eps = d.get("eps"); bps = d.get("bps")
            ocfps = d.get("ocfps"); npm = d.get("npm")

            pe = real_price / eps if eps and eps != 0 else np.nan
            pb = real_price / bps if bps and bps != 0 else np.nan
            pcf = real_price / ocfps if ocfps and ocfps != 0 else np.nan
            # PS = PE / (npm/100) = PE * 100 / npm
            if np.isfinite(pe) and npm and npm != 0:
                ps = pe * 100.0 / npm
            else:
                ps = np.nan

            if any(np.isfinite([pe, pb, ps, pcf])):
                code_rows.append({
                    "date": date_str,
                    "pe_ttm": round(pe, 4) if np.isfinite(pe) else "",
                    "pb": round(pb, 4) if np.isfinite(pb) else "",
                    "ps_ttm": round(ps, 4) if np.isfinite(ps) else "",
                    "pcf": round(pcf, 4) if np.isfinite(pcf) else "",
                    "peg": "",
                    "code": code6,
                })

        out_rows.extend(code_rows)
        n_ok += 1

        if (ci + 1) % 50 == 0 or ci == len(codes) - 1:
            print(f"  [{ci+1}/{len(codes)}] ok={n_ok} skip={n_skip} rows={len(out_rows)}", flush=True)

    # 写出
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(VAL_CACHE, sep='\t', index=False)
    print(f"\n[完成] valuation_cache.csv: {out_df.shape}, {out_df['code'].nunique()} 只, "
          f"{out_df['date'].min()}~{out_df['date'].max()}", flush=True)

    # 验证茅台
    mt = out_df[out_df["code"] == "600519"]
    if len(mt) > 0:
        r = mt.iloc[0]
        print(f"  验证 茅台 {r['date']}: PE={r['pe_ttm']} PB={r['pb']} PS={r['ps_ttm']} PCF={r['pcf']}", flush=True)
        r2 = mt[mt["date"] == "2020-01-02"]
        if len(r2) > 0:
            r2 = r2.iloc[0]
            print(f"  验证 茅台 2020-01-02: PE={r2['pe_ttm']} PB={r2['pb']} PS={r2['ps_ttm']} PCF={r2['pcf']}", flush=True)

# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 build_valuation_cache.py [fetch|build]")
        sys.exit(1)
    if sys.argv[1] == "fetch":
        phase_fetch()
    elif sys.argv[1] == "build":
        phase_build()
    else:
        print(f"未知命令: {sys.argv[1]}")
        sys.exit(1)
