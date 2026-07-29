"""
月度数据更新 (安全版) — 上线专用
==================================================================
为什么不能直接用 qlib/update_qlib_data.py:
  1) FETCH_START/END 硬编码(2020-09-28~2026-07-23), 下月跑等于什么都不更新;
  2) ohlcv_cache.parquet "缓存完整就直接返回", 会拿上个月的旧数据当新数据;
  3) 复权连续性校正拿"已有末日收盘 vs 新数据首日收盘"比, 增量追加时这两天本就
     不是同一天, 正常涨跌1%以上就会触发缩放 → 静默篡改价格;
  4) bin 是追加写且不幂等, 重跑一次数据就错位, 且无备份。

本脚本的做法:
  Step0 预检   : bin 各字段长度一致性 + 是否越界(检测历史重复追加) → 不通过就停
  Step1 备份   : day.txt + 池内股票全部 .day.bin → backup/时间戳/  (可一键回滚)
  Step2 抓行情 : 自动 FETCH_START = 已有末日-20自然日(留重叠), END = 今天
  Step3 追加   : 只追加 cal_idx > 已有末日 的部分(幂等);
                 复权校正用【重叠日同一天】的收盘价比 → scale, OHLC同比例乘,
                 volume不动(涨停判定依赖OHLC相等关系, 同比例缩放不破坏该关系)
  Step4 校验   : 抽样比对 qlib 读出值 vs akshare 原值; 检查长度/日历对齐
  Step5 估值   : 重抓池内估值全历史 → 与旧表合并去重 → 校验后原子替换(旧表备份)

用法:
  /usr/bin/python3 monthly_update.py            # 正式更新
  /usr/bin/python3 monthly_update.py --dry-run  # 只体检+报告要更新什么, 不写任何文件
回滚:
  脚本末尾会打印具体的回滚命令(把 backup/时间戳/ 覆盖回去)
"""
import os, sys, time, json, shutil, warnings, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from core import config
from core.universe import build_dynamic_universe, format_qlib_code

QLIB_DIR = Path(os.path.expanduser("~/.qlib/qlib_data/cn_data"))
CAL_PATH = QLIB_DIR / "calendars" / "day.txt"
OUT_DIR = Path(config.ONLINE_DIR)
DATA_DIR = Path(config.DATA_DIR)
VAL_CACHE = Path(config.VAL_CACHE)
FIELDS = ["open", "close", "high", "low", "volume", "factor", "change"]
MAX_WORKERS = 8
DRY = "--dry-run" in sys.argv


def log(m):
    print(m, flush=True)


def current_pool():
    """当前年度的股票池 (与选股同一套 core 逻辑, 保证"更新的正是要用的")"""
    fcf = pd.read_csv(config.FCF_CACHE, sep='\t'); fcf["code"] = fcf["code"].astype(str).str.zfill(6)
    prof = pd.read_csv(config.PROFIT_CACHE, sep='\t'); prof["code"] = prof["code"].astype(str).str.zfill(6)
    y = pd.Timestamp.now().year
    codes = sorted(build_dynamic_universe(y, fcf, prof))
    return codes, [format_qlib_code(c) for c in codes]


def read_bin(p):
    with open(p, "rb") as f:
        a = np.frombuffer(f.read(), dtype="<f4")
    return int(a[0]), a[1:]


# ---------------- Step0 预检 ----------------
def precheck(qcodes, cal_dates):
    log("\n--- Step0 预检: bin 结构一致性 ---")
    bad, missing = [], []
    for qc in qcodes:
        d = QLIB_DIR / "features" / qc
        if not d.exists():
            missing.append(qc); continue
        lens, starts = {}, {}
        for f in FIELDS:
            p = d / f"{f}.day.bin"
            if not p.exists():
                bad.append((qc, f, "缺失字段")); continue
            si, v = read_bin(p)
            lens[f] = len(v); starts[f] = si
        if len(set(lens.values())) > 1 or len(set(starts.values())) > 1:
            bad.append((qc, "-", f"字段长度/起点不一致 len={lens} start={starts}"))
            continue
        if lens and starts:
            end_idx = list(starts.values())[0] + list(lens.values())[0] - 1
            if end_idx > len(cal_dates) - 1:
                bad.append((qc, "-", f"数据越界 end_idx={end_idx} > 日历末尾{len(cal_dates)-1} (疑似重复追加过)"))
    log(f"  池内 {len(qcodes)} 只: 结构异常 {len(bad)} 只, qlib中不存在 {len(missing)} 只")
    for b in bad[:10]:
        log(f"    [异常] {b}")
    if missing:
        log(f"    [新股待建] {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return bad, missing


# ---------------- Step1 备份 ----------------
def backup(qcodes):
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    bdir = OUT_DIR / "backup" / stamp
    (bdir / "features").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CAL_PATH, bdir / "day.txt")
    n = 0
    for qc in qcodes:
        s = QLIB_DIR / "features" / qc
        if s.exists():
            shutil.copytree(s, bdir / "features" / qc, dirs_exist_ok=True); n += 1
    shutil.copy2(VAL_CACHE, bdir / "valuation_cache.csv")
    sz = sum(f.stat().st_size for f in bdir.rglob("*") if f.is_file()) / 1e6
    log(f"  已备份 日历 + {n}只features + 估值表 → {bdir} ({sz:.0f}MB)")
    return bdir


# ---------------- Step2 抓行情 ----------------
def fetch_one(code, start, end):
    import akshare as ak
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start,
                                    end_date=end, adjust="qfq")
            if df is None or len(df) == 0:
                return code, None
            df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                                    "最低": "low", "成交量": "volume", "涨跌幅": "change_pct"})
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["change"] = df["change_pct"] / 100.0
            return code, df[["date", "open", "close", "high", "low", "volume", "change"]]
        except Exception as e:
            if attempt == 2:
                return code, f"ERR:{str(e)[:50]}"
            time.sleep(1.5)


def fetch_all(codes, start, end):
    log(f"\n--- Step2 抓行情 {start} ~ {end} ({len(codes)}只, {MAX_WORKERS}线程) ---")
    out, errs = {}, []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_one, c, start, end): c for c in codes}
        done = 0
        for fu in as_completed(futs):
            c, r = fu.result(); done += 1
            if isinstance(r, pd.DataFrame):
                out[c] = r
            else:
                errs.append((c, r))
            if done % 60 == 0 or done == len(codes):
                log(f"  进度 {done}/{len(codes)}")
    log(f"  成功 {len(out)}只, 失败 {len(errs)}只 {errs[:5]}")
    return out, errs


# ---------------- Step3 追加 ----------------
def append_one(qc, df, cal_dates):
    """只追加已有末日之后的数据; 复权基准用重叠日同一天收盘价校正。
    返回 (状态, 说明, scale)"""
    d = QLIB_DIR / "features" / qc
    if not d.exists():
        return "skip", "qlib无此股(新股需单独建仓, 本月不纳入)", None
    si, close_v = read_bin(d / "close.day.bin")
    end_idx = si + len(close_v) - 1
    if end_idx >= len(cal_dates) - 1:
        return "uptodate", f"已到日历末尾({cal_dates[end_idx]})", None
    ref_date = cal_dates[end_idx]                       # 已有数据最后一天
    ref_close = float(close_v[-1])
    dfi = df.set_index("date")
    if ref_date not in dfi.index or not np.isfinite(ref_close) or ref_close <= 0:
        return "fail", f"无重叠日{ref_date}可校正复权基准, 拒绝追加", None
    new_ref = float(dfi.loc[ref_date, "close"])
    if new_ref <= 0:
        return "fail", f"重叠日{ref_date}新数据收盘异常", None
    scale = ref_close / new_ref
    if not (0.2 < scale < 5.0):
        return "fail", f"复权scale异常={scale:.3f} (重叠日{ref_date}: 旧{ref_close:.2f} vs 新{new_ref:.2f}), 拒绝追加", scale

    idx_of = {dt: i for i, dt in enumerate(cal_dates)}
    dfi = dfi[[dt in idx_of and idx_of[dt] > end_idx for dt in dfi.index]]
    if len(dfi) == 0:
        return "uptodate", "无新交易日", scale
    app_start, app_end = end_idx + 1, max(idx_of[dt] for dt in dfi.index)
    n = app_end - app_start + 1
    arr = {f: np.full(n, np.nan, dtype=np.float32) for f in FIELDS}
    _, fac_v = read_bin(d / "factor.day.bin")
    last_fac = float(fac_v[-1]) if len(fac_v) else 1.0
    for dt, row in dfi.iterrows():
        i = idx_of[dt] - app_start
        for f in ["open", "close", "high", "low"]:
            arr[f][i] = float(row[f]) * scale          # OHLC同比例 → 不破坏涨停的相等关系
        arr["volume"][i] = float(row["volume"])
        arr["change"][i] = float(row["change"])
        arr["factor"][i] = last_fac
    if DRY:
        return "dry", f"待追加{n}天({cal_dates[app_start]}~{cal_dates[app_end]}) scale={scale:.4f}", scale
    for f in FIELDS:
        with open(d / f"{f}.day.bin", "ab") as fh:
            arr[f].astype("<f4").tofile(fh)
    return "ok", f"追加{n}天({cal_dates[app_start]}~{cal_dates[app_end]}) scale={scale:.4f}", scale


# ---------------- Step5 估值 ----------------
def update_valuation(codes):
    log(f"\n--- Step5 估值更新 ({len(codes)}只) ---")
    import akshare as ak
    old = pd.read_csv(VAL_CACHE, sep='\t', dtype={"code": str})
    old_max, old_rows, old_n = old["date"].max(), len(old), old["code"].nunique()
    log(f"  旧表: {old_rows}行 {old_n}只 至 {old_max}")
    if DRY:
        codes = codes[:15]
        log(f"  [dry-run] 只抽测前{len(codes)}只验证接口连通性与字段, 不写入")
    buf, errs = [], []
    for i, c in enumerate(codes):
        for attempt in range(3):
            try:
                df = ak.stock_value_em(symbol=c)
                df = df.rename(columns={"数据日期": "date", "PE(TTM)": "pe_ttm", "市净率": "pb",
                                        "市销率": "ps_ttm", "市现率": "pcf", "PEG值": "peg"})
                df = df[["date", "pe_ttm", "pb", "ps_ttm", "pcf", "peg"]].copy()
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df["code"] = c
                buf.append(df); break
            except Exception as e:
                if attempt == 2:
                    errs.append((c, str(e)[:40]))
                time.sleep(1.2)
        time.sleep(0.3)
        if (i + 1) % 60 == 0 or i == len(codes) - 1:
            log(f"  进度 {i+1}/{len(codes)}")
    if not buf:
        log("  [FAIL] 一条都没抓到, 估值表保持不变"); return False
    new = pd.concat(buf, ignore_index=True)
    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "code"], keep="last").sort_values(["code", "date"])
    new_max, new_rows, new_n = merged["date"].max(), len(merged), merged["code"].nunique()
    log(f"  新表: {new_rows}行 {new_n}只 至 {new_max} (失败{len(errs)}只)")
    if new_rows < old_rows:
        log(f"  [FAIL] 行数倒退 {old_rows}→{new_rows}, 放弃写入"); return False
    if new_n < old_n:
        log(f"  [FAIL] 股票数倒退 {old_n}→{new_n}, 放弃写入"); return False
    if new_max < old_max:
        log(f"  [FAIL] 最新日期倒退 {old_max}→{new_max}, 放弃写入"); return False
    if new_max == old_max:
        log(f"  [WARN] 最新日期未前进(仍{new_max}) — 可能今日数据源未更新, 仍写入(内容可能有修正)")
    if DRY:
        log("  [dry-run] 不写入"); return True
    tmp = str(VAL_CACHE) + ".tmp"
    merged.to_csv(tmp, sep='\t', index=False)
    os.replace(tmp, VAL_CACHE)
    log(f"  [OK] 已原子替换 valuation_cache.csv (旧表已在 backup 中)")
    return True


# ---------------- Step4 校验 ----------------
def verify_append(data, scales, cal_dates):
    """校验追加结果。
    注意: 不能直接比 qlib值 vs akshare值 —— qlib存的是"旧复权基准"价格,
    akshare返回的是"今日基准"价格, 有过大额分红/送股的股票绝对值差几十%是正常的
    (例: 云南白药 scale=1.6955)。正确判据是:
      1) qlib末值 ≈ akshare末值 × scale        (基准换算后应吻合)
      2) qlib日收益 ≈ akshare涨跌幅            (与复权基准完全无关, 最本质的检验)
    抽样优先选 scale 偏离1最远的几只 —— 它们最容易暴露基准校正错误。"""
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    qlib.init(provider_uri=str(QLIB_DIR), region=REG_CN)
    s = pd.Series(scales)
    picks = list(s.sub(1).abs().nlargest(3).index)          # scale偏离最大的3只
    picks += [c for c in list(data)[:3] if c not in picks]   # 再随机补3只
    n_ok = 0
    for c in picks:
        qc = format_qlib_code(c)
        got = D.features([qc], ["$close"], start_time=cal_dates[-6], end_time=cal_dates[-1])
        if got is None or len(got) == 0:
            log(f"  {qc} qlib读不出数据 <<< 异常!"); continue
        g = got["$close"].droplevel(0)
        exp = data[c].set_index("date")
        gv = float(g.iloc[-1]); ev = float(exp["close"].iloc[-1]) * scales[c]
        rel = abs(gv - ev) / ev
        cmp = pd.DataFrame({"q": (g.pct_change() * 100).values},
                           index=[str(d.date()) for d in g.index]).join(
                           (exp["change"] * 100).rename("a")).dropna()
        md = float((cmp["q"] - cmp["a"]).abs().max()) if len(cmp) else float("nan")
        ok = rel < 0.005 and (md < 0.3 or np.isnan(md))
        n_ok += ok
        log(f"  {qc} scale={scales[c]:.4f} | 末值 qlib{gv:.2f} vs akshare×scale{ev:.2f} "
            f"(差{rel*100:.3f}%) | 日收益最大偏差{md:.3f}pp  {'OK' if ok else '<<< 异常!'}")
    log(f"  抽样 {n_ok}/{len(picks)} 只通过 (判据: 末值差<0.5% 且 日收益偏差<0.3pp)")
    return n_ok == len(picks)


def main():
    log("=" * 90 + f"\n  月度数据更新  {pd.Timestamp.now():%Y-%m-%d %H:%M}  {'[DRY-RUN 只读]' if DRY else '[正式写入]'}\n" + "=" * 90)
    codes, qcodes = current_pool()
    log(f"  当前股票池: {len(codes)}只")
    cal_dates = [l.strip() for l in open(CAL_PATH)]
    log(f"  qlib日历: {cal_dates[0]} ~ {cal_dates[-1]} ({len(cal_dates)}天)")

    bad, missing = precheck(qcodes, cal_dates)
    if bad:
        log("\n[STOP] bin 结构异常, 禁止更新。先修复上述股票(可从 backup 回滚)后重试。"); sys.exit(1)

    import akshare as ak
    all_td = pd.to_datetime(ak.tool_trade_date_hist_sina()["trade_date"]).dt.strftime("%Y-%m-%d").tolist()
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    pending = [d for d in all_td if d > cal_dates[-1] and d <= today]
    log(f"\n  待补交易日: {len(pending)}天 {pending[:3]}{'...' if len(pending) > 3 else ''}")

    bdir = None
    if pending:
        if not DRY:
            log("\n--- Step1 备份 ---")
            bdir = backup(qcodes)
        start = (pd.Timestamp(cal_dates[-1]) - pd.Timedelta(days=20)).strftime("%Y%m%d")
        data, ferrs = fetch_all(codes, start, today.replace("-", ""))
        if len(data) < len(codes) * 0.9:
            log(f"\n[STOP] 行情抓取覆盖不足({len(data)}/{len(codes)}), 放弃本次更新(未写入任何行情)"); sys.exit(1)
        if not DRY:
            with open(CAL_PATH, "a") as f:
                for d in pending:
                    f.write(d + "\n")
            cal_dates = [l.strip() for l in open(CAL_PATH)]
            log(f"  日历已追加 {len(pending)}天 → 末尾 {cal_dates[-1]}")
        else:
            cal_dates = cal_dates + pending
        log("\n--- Step3 追加行情 ---")
        stat, scales = {}, {}
        for c in codes:
            if c not in data:
                stat.setdefault("nodata", []).append(c); continue
            st, msg, sc = append_one(format_qlib_code(c), data[c], cal_dates)
            stat.setdefault(st, []).append((c, msg))
            if sc is not None and st in ("dry", "ok"):
                scales[c] = sc
        for k, v in stat.items():
            log(f"  {k}: {len(v)}只" + (f"  例: {v[:2]}" if k in ("fail", "skip") else ""))
        if scales:
            s = pd.Series(scales)
            n_dev = int((s.sub(1).abs() > 0.05).sum())
            log(f"  复权scale分布: 中位{s.median():.4f} 最小{s.min():.4f} 最大{s.max():.4f}; "
                f"偏离1超5%的{n_dev}只(近期除权所致, 正常)")
            log(f"    scale应接近1(无除权)或明显偏离(有除权); 若绝大多数都轻微偏离(1~5%), "
                f"说明基准算错了 → 立即停止并检查")
            if n_dev > len(scales) * 0.5:
                log(f"  [WARN] 超半数股票scale偏离>5%, 极不正常, 请人工核查后再决定是否正式写入")
        if stat.get("fail"):
            log(f"  [WARN] {len(stat['fail'])}只因复权基准无法校正被跳过, 这些股票本月行情停留在旧末日;")
            log(f"         若它们出现在买入清单里, preflight 的行情覆盖率检查会拦下来。")
        log("\n--- Step4 校验 ---")
        if DRY:
            log("  [dry-run] 未写入, 无可校验内容(正式运行时会抽样比对 qlib读值 vs akshare原值)")
        else:
            verify_append(data, scales, cal_dates)
    else:
        log("  行情已是最新, 跳过 Step1-4")
        if not DRY:
            bdir = backup(qcodes)

    update_valuation(codes)

    log("\n" + "=" * 90)
    log("  更新完成。下一步: 1) python3 preflight_check.py  2) python3 gen_holdings.py  3) 按清单交易")
    if bdir:
        log(f"\n  如需回滚:\n    cp {bdir}/day.txt {CAL_PATH}\n"
            f"    cp -R {bdir}/features/* {QLIB_DIR}/features/\n"
            f"    cp {bdir}/valuation_cache.csv {VAL_CACHE}")
    log("=" * 90)


if __name__ == "__main__":
    main()
