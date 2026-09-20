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
QLIB_DATA = os.path.join(os.path.dirname(QUANT_DIR), "data_cache", "qlib_cn_tencent")

PERSHARE_CACHE = os.path.join(DATA_DIR, "pershare_cache_pit.csv")
VAL_CACHE = os.path.join(DATA_DIR, "valuation_cache.csv")
FCF_CACHE = os.path.join(DATA_DIR, "fcf_cache_pit.csv")
PROFIT_CACHE = os.path.join(DATA_DIR, "profit_cache_pit.csv")

SINA_HEADERS = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
FETCH_YEARS = list(range(2016, 2027))   # 2016-2026 (补当年季报, 供 PIT 估值延伸)
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

    # 加载已完成 (code, year) 粒度: 已有该年任意季报行则视为该年完成
    have = set()
    if os.path.exists(PERSHARE_CACHE):
        df = pd.read_csv(PERSHARE_CACHE, sep='\t', dtype={"code": str})
        df["code"] = df["code"].str.zfill(6)
        df["year"] = df["report_date"].astype(str).str[:4]
        have = set(zip(df["code"], df["year"]))
    # 空结果记录: 退市/无该年数据的 code 不必每次重抓
    attempted_path = PERSHARE_CACHE + ".attempted"
    if os.path.exists(attempted_path):
        with open(attempted_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    have.add(tuple(line.split(",")))
    todo = [(c, str(y)) for c in codes for y in FETCH_YEARS if (c, str(y)) not in have]
    print(f"  已有: {len(have)} (code,year) 组合, 待抓: {len(todo)} 页", flush=True)

    buf = []
    empty_years = []
    t0 = time.time()
    n_fail = 0
    attempted_f = open(attempted_path, "a")

    for idx, (code, ystr) in enumerate(todo):
        data = fetch_guide(code, ystr)
        if data:
            for d, v in data.items():
                row = {"code": code, "report_date": d}
                row.update(v)
                buf.append(row)
        else:
            empty_years.append(f"{code},{ystr}")
        attempted_f.write(f"{code},{ystr}\n")
        if not data:
            n_fail += 1

        # 每 20 页 flush
        if (idx + 1) % 20 == 0 or idx == len(todo) - 1:
            attempted_f.flush()
            if buf:
                df = pd.DataFrame(buf)
                if not os.path.exists(PERSHARE_CACHE):
                    df.to_csv(PERSHARE_CACHE, sep='\t', index=False)
                else:
                    df.to_csv(PERSHARE_CACHE, sep='\t', index=False, mode='a', header=False)
                buf = []
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(todo) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(todo)}] empty={n_fail} {elapsed:.0f}s ETA {eta:.0f}s", flush=True)
    attempted_f.close()

    # 汇总
    if os.path.exists(PERSHARE_CACHE):
        df = pd.read_csv(PERSHARE_CACHE, sep='\t', dtype={"code": str})
        df["code"] = df["code"].str.zfill(6)
        print(f"\n[完成] pershare_cache_pit.csv: {df.shape}, "
              f"{df['code'].nunique()} 只, {df['report_date'].nunique()} 季度", flush=True)

# ============================================================
#  Phase 2: Build valuation_cache.csv
# ============================================================
def _pit_quarter(date_str):
    """信号日 → 最新可用季报 (year, q_end)
    PIT 季报披露规则 (保守取截止日次日):
    - Q1(03-31): 4月30日截止 → 5月1日起可用
    - H1(06-30): 8月31日截止 → 9月1日起可用
    - Q3(09-30): 10月31日截止 → 11月1日起可用
    - Annual(12-31): 次年4月30日截止 → 次年5月1日起可用
    """
    d = date_str.replace('-', '')
    y, m = int(d[:4]), int(d[4:6])
    if m >= 11: return (y, "09-30")      # Nov-Dec: Q3 Y
    if m >= 9:  return (y, "06-30")      # Sep-Oct: H1 Y
    if m >= 5:  return (y, "03-31")      # May-Aug: Q1 Y
    return (y - 1, "09-30")              # Jan-Apr: Q3 Y-1

def _sps(d):
    """每股营收 = 每股收益 / 销售净利率"""
    if d and d.get("eps") is not None and d.get("npm") and d["npm"] != 0:
        return d["eps"] / (d["npm"] / 100.0)
    return None

def phase_build():
    print("[Phase 2] 构建 valuation_cache.csv (滚动TTM)", flush=True)

    # 日历
    cal = open(os.path.join(QLIB_DATA, "calendars", "day.txt")).read().strip().split()
    print(f"  日历: {len(cal)} 天 {cal[0]}~{cal[-1]}", flush=True)

    # 加载全部季报数据
    ps_df = pd.read_csv(PERSHARE_CACHE, sep='\t', dtype={"code": str})
    ps_df["code"] = ps_df["code"].str.zfill(6)
    ps_df["report_date"] = ps_df["report_date"].astype(str)
    # 建 {code: {"2016-03-31": {eps, bps, ocfps, npm}, ...}}
    pershare = {}
    for _, row in ps_df.iterrows():
        c = row["code"]
        if c not in pershare:
            pershare[c] = {}
        pershare[c][row["report_date"]] = {
            "eps": row.get("eps"),
            "bps": row.get("bps"),
            "ocfps": row.get("ocfps"),
            "npm": row.get("npm"),
        }
    print(f"  每股指标(季报): {len(pershare)} 只, {ps_df['report_date'].nunique()} 个报告期", flush=True)

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

        ps_q = pershare.get(code6, {})
        if not ps_q:
            n_skip += 1
            continue

        code_rows = []
        for i, date_str in enumerate(cal):
            if np.isnan(close[i]) or np.isnan(factor[i]) or factor[i] <= 0:
                continue
            real_price = close[i] / factor[i]

            # PIT: 确定最新可用季报
            yr, q_end = _pit_quarter(date_str)
            cur_key = f"{yr}-{q_end}"
            cur = ps_q.get(cur_key)
            if not cur:
                continue

            # TTM = cur_cum - same_quarter_last_year_cum + last_annual
            prev = ps_q.get(f"{yr-1}-{q_end}")
            ann = ps_q.get(f"{yr-1}-12-31")

            if q_end == "12-31":
                # 年报本身就是 TTM
                ttm_eps = cur.get("eps")
                ttm_ocfps = cur.get("ocfps")
            elif prev and ann:
                ttm_eps = (cur["eps"] or 0) - (prev["eps"] or 0) + (ann["eps"] or 0)
                ttm_ocfps = (cur["ocfps"] or 0) - (prev["ocfps"] or 0) + (ann["ocfps"] or 0)
            elif ann:
                ttm_eps = ann.get("eps")
                ttm_ocfps = ann.get("ocfps")
            else:
                continue

            # BPS: 时点值, 直接用最新季报
            bps = cur.get("bps")

            # TTM SPS = cur_SPS - prev_SPS + ann_SPS
            if q_end == "12-31":
                ttm_sps = _sps(cur)
            elif prev and ann:
                c_s = _sps(cur); p_s = _sps(prev); a_s = _sps(ann)
                ttm_sps = (c_s - p_s + a_s) if (c_s and p_s and a_s) else None
            else:
                ttm_sps = _sps(ann) if ann else None

            # 估值
            pe = real_price / ttm_eps if ttm_eps and ttm_eps != 0 else np.nan
            pb = real_price / bps if bps and bps != 0 else np.nan
            pcf = real_price / ttm_ocfps if ttm_ocfps and ttm_ocfps != 0 else np.nan
            ps = real_price / ttm_sps if ttm_sps and ttm_sps != 0 else np.nan

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
        for d in ["2020-01-02", "2024-06-03"]:
            r2 = mt[mt["date"] == d]
            if len(r2) > 0:
                r2 = r2.iloc[0]
                print(f"  验证 茅台 {d}: PE={r2['pe_ttm']} PB={r2['pb']} PS={r2['ps_ttm']} PCF={r2['pcf']}", flush=True)

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
