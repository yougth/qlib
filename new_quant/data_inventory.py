#!/usr/bin/env python3
"""只读二进制数据审计；不导入旧项目、不联网、不写数据，摘要仅输出终端。"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "data_cache"
INDEX = {"sh000001", "sh000300", "sh000905"}
FIELDS = ["open", "close", "high", "low", "volume", "factor"]
# 日历文本已由专门读取工具核验；这里不读取任何文本文件。
BIN_LENGTH = 3569
BIN_LAST = "2026-09-11"


def emit(label, value):
    print(label + " " + json.dumps(value, ensure_ascii=False, default=str), flush=True)


def quantiles(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return dict(zip(["min", "p10", "p50", "p90", "max"],
                    np.quantile(arr, [0, .1, .5, .9, 1]).tolist())) if len(arr) else {}


def stamp(path):
    s = path.stat()
    return (s.st_size, s.st_mtime_ns, s.st_ino)


def read_frame(path):
    df = pd.read_parquet(path)
    return df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df


def table_summary(path):
    df = read_frame(path)
    info = {"path": str(path), "rows": len(df),
            "schema": {c: str(t) for c, t in df.dtypes.items()},
            "nulls": {c: int(n) for c, n in df.isna().sum().items()},
            "parquet_metadata_keys": [k.decode() for k in (pq.ParquetFile(path).metadata.metadata or {})]}
    for col in ["code", "symbol", "instrument"]:
        if col in df:
            info[col + "_unique"] = int(df[col].nunique())
    for col in ["date", "datetime", "report_date", "notice_date", "year", "list_date", "delist_date"]:
        if col in df:
            s = df[col].dropna()
            info[col + "_range"] = [str(s.min()), str(s.max())] if len(s) else []
    keys = [c for c in ["code", "symbol", "date", "year"] if c in df]
    if keys:
        info["duplicate_keys"] = int(df.duplicated(keys).sum())
    if "close" in df:
        info["nonpositive_close"] = int((df.close <= 0).sum())
    if "notice_date" in df:
        dt = pd.to_datetime(df.notice_date, errors="coerce")
        info["notice_parse_missing"] = int(dt.isna().sum())
        info["by_code_length"] = {}
        for size, g in df.groupby(df.code.astype(str).str.len()):
            info["by_code_length"][str(size)] = {"rows": len(g), "codes": int(g.code.nunique()),
                "notice_missing": int(g.notice_date.isna().sum()),
                                "year_range": [int(g.year.min()), int(g.year.max())]}
        lag = (dt - pd.to_datetime(df.year.astype(str) + "-12-31")).dt.days
        info["notice_lag_days"] = quantiles(lag)
        info["notice_lag_over_366"] = int((lag > 366).sum())
        info["codes_per_year"] = df.groupby("year").code.nunique().to_dict()
    emit("表", info)
    return df


def audit_quotes():
    folder = CACHE / "tencent"
    files = sorted(folder.glob("*.parquet"))
    bench = read_frame(folder / "sh000300.parquet")
    calendar = pd.DatetimeIndex(pd.to_datetime(bench.date).sort_values().unique())
    emit("行情参考日历", {"path": str(folder / "sh000300.parquet"), "n": len(calendar),
                         "first": str(calendar.min()), "last": str(calendar.max())})
    summaries, errors, jumps, vol_samples = [], [], [], []
    year_symbols, date_counts, schemas, nulls = Counter(), Counter(), Counter(), Counter()
    stage = Counter()
    units_by_date = {}
    total_rows = 0
    for path in files:
        try:
            df = read_frame(path).sort_values("date")
            d = pd.DatetimeIndex(pd.to_datetime(df.date))
            sym = path.stem
            schemas[str({c: str(t) for c, t in df.dtypes.items()})] += 1
            nulls.update({c: int(n) for c, n in df.isna().sum().items()})
            total_rows += len(df)
            p = df.close.to_numpy(float)
            f = df.factor.to_numpy(float)
            v = df.volume.to_numpy(float)
            good = np.isfinite(p) & (p > 0)
            valid_dates = d[good]
            if sym not in INDEX:
                year_symbols.update(set(valid_dates.year))
                date_counts.update(valid_dates.strftime("%Y-%m-%d"))
            expected = calendar[(calendar >= d.min()) & (calendar <= d.max())]
            fac_drop = f[1:] / f[:-1] - 1
            ret = p[1:] / p[:-1] - 1
            with np.errstate(divide="ignore", invalid="ignore"):
                vr = v[1:] / v[:-1]
                raw = p / f
            bad_ohlc = ((df.high < df[["open", "close"]].max(axis=1) * .999) |
                        (df.low > df[["open", "close"]].min(axis=1) * 1.001))
            s = {"symbol": sym, "rows": len(df), "start": str(d.min().date()),
                 "end": str(d.max().date()), "duplicate_dates": int(d.duplicated().sum()),
                 "weekend": int((d.dayofweek >= 5).sum()), "bad_close": int((~good).sum()),
                 "bad_factor": int((~np.isfinite(f) | (f <= 0)).sum()),
                 "factor_drop_1pct": int((fac_drop < -.01).sum()),
                 "factor_drop_50pct": int((fac_drop < -.5).sum()),
                 "return_abs_55pct": int((np.abs(ret) > .55).sum()),
                 "nonpositive_volume": int((v <= 0).sum()),
                 "ohlc_bad": int(bad_ohlc.sum()),
                 "interior_absent_dates": len(expected.difference(d)),
                 "raw_min": float(np.nanmin(raw)), "raw_max": float(np.nanmax(raw)),
                 "factor_min": float(np.nanmin(f)), "factor_max": float(np.nanmax(f))}
            summaries.append(s)
            if sym not in INDEX:
                pre = v[(d <= "2026-07-23") & (v > 0)][-20:]
                post = v[(d > "2026-07-23") & (v > 0)][:20]
                if len(pre) >= 10 and len(post) >= 10:
                    vol_samples.append({"symbol": sym, "post_pre_median_ratio": float(np.median(post) / np.median(pre))})
                recent = d[1:] >= pd.Timestamp("2026-07-01")
                for day, ratio in zip(d[1:][recent], vr[recent]):
                    if np.isfinite(ratio) and ratio > 0:
                        units_by_date.setdefault(str(day.date()), []).append(float(ratio))
            for i in np.where((fac_drop < -.5) | (np.abs(ret) > .55))[0][:3]:
                jumps.append({"symbol": sym, "date": str(d[i + 1].date()),
                              "return": float(ret[i]), "factor_change": float(fac_drop[i])})
            other = CACHE / "tencent_stage" / path.name
            if not other.exists():
                stage["missing"] += 1
            elif path.stat().st_ino == other.stat().st_ino:
                stage["hardlink"] += 1
            elif hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(other.read_bytes()).digest():
                stage["identical_bytes"] += 1
            else:
                stage["different_bytes"] += 1
                alt = read_frame(other)
                stage["row_delta_source_minus_stage"] += len(df) - len(alt)
            if sym in ["sz000026", "sh600519", "sh000300", "sz000001"]:
                window = (d >= "2026-07-01") & (d <= "2026-07-23")
                if sym not in INDEX and window.sum() >= 10:
                    slope, offset = np.polyfit(raw[window], p[window], 1)
                    emit("价格因子线性诊断", {"symbol": sym, "interval": "2026-07-01~2026-07-23",
                        "slope": float(slope), "offset": float(offset),
                        "max_residual": float(np.max(np.abs(p[window] - slope * raw[window] - offset))),
                        "note": "raw由close/factor反推，非独立价格核验；非零截距提示非纯乘法复权"})
                emit("行情样本", {"symbol": sym, "head": df.head(2).to_dict("records"),
                                   "tail": df.tail(3).to_dict("records"),
                                   "seam": df.loc[(d >= "2026-07-21") & (d <= "2026-07-29")].to_dict("records")})
        except Exception as exc:
            errors.append({"path": str(path), "error": repr(exc)})
    s = pd.DataFrame(summaries)
    stock = s[~s.symbol.isin(INDEX)]
    metrics = ["duplicate_dates", "weekend", "bad_close", "bad_factor", "factor_drop_1pct",
               "factor_drop_50pct", "return_abs_55pct", "nonpositive_volume", "ohlc_bad", "interior_absent_dates"]
    emit("行情汇总", {"files": len(files), "rows": total_rows, "stocks_excluding_three_indices": len(stock),
        "schema_variants": dict(schemas), "nulls": dict(nulls), "errors": errors,
        "first": s.start.min(), "last": s.end.max(), "prefixes": stock.symbol.str[:3].value_counts().to_dict(),
        "start_dates_top": stock.start.value_counts().head(12).to_dict(),
        "end_dates_top": stock.end.value_counts().head(15).to_dict(),
        "stock_symbols_by_year_any_valid_close": dict(sorted(year_symbols.items())),
        "stock_rows_recent_dates": {k: v for k, v in sorted(date_counts.items()) if k >= "2026-07-20"},
        "metrics_sum": {k: int(s[k].sum()) for k in metrics},
        "metrics_affected_files": {k: int((s[k] > 0).sum()) for k in metrics},
        "raw_price_range": [s.raw_min.min(), s.raw_max.max()],
        "stage_comparison": dict(stage), "jump_samples": jumps[:15]})
    emit("成交量拼接诊断", {"samples": len(vol_samples),
        "post_pre_20day_median_ratio": quantiles([r["post_pre_median_ratio"] for r in vol_samples]),
        "ratio_over_20": sum(r["post_pre_median_ratio"] > 20 for r in vol_samples),
        "examples": sorted(vol_samples, key=lambda r: -r["post_pre_median_ratio"])[:6],
        "daily_ratio_outliers": {day: {"n": len(vals), "median": float(np.median(vals)),
                                      "over_20": sum(x > 20 for x in vals)}
            for day, vals in units_by_date.items() if np.median(vals) > 5 or sum(x > 20 for x in vals) >= 20}})
    return {p.stem for p in files}


def audit_bin():
    base = CACHE / "qlib_cn_tencent" / "features"
    results, errors, fields = [], [], Counter()
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        entry = {"symbol": folder.name}
        for field in FIELDS:
            p = folder / (field + ".day.bin")
            if not p.exists():
                errors.append(str(p) + " 缺失")
                continue
            a = np.fromfile(p, dtype="<f4")
            fields[field] += 1
            if len(a) < 2 or not np.isfinite(a[0]) or a[0] != int(a[0]) or a[0] < 0:
                errors.append(str(p) + " 头部异常")
                continue
            start = int(a[0])
            values = a[1:]
            valid = np.flatnonzero(np.isfinite(values))
            if start + len(values) > BIN_LENGTH:
                errors.append(str(p) + " 超出已核验日历")
            if field == "close":
                entry.update({"start_index": start, "last_valid_index": start + int(valid[-1]) if len(valid) else -1,
                              "finite_rows": len(valid), "nonpositive": int((values <= 0).sum())})
            if field == "factor":
                entry["factor_all_one"] = bool(np.all(values[valid] == 1))
        results.append(entry)
    out = pd.DataFrame(results)
    emit("BIN汇总", {"path": str(base), "symbols": len(out), "fields": dict(fields),
                     "known_calendar_length": BIN_LENGTH, "known_calendar_last": BIN_LAST,
                     "errors": errors[:30], "error_count": len(errors),
                     "market_counts": out.symbol.str[:2].value_counts().to_dict(),
                     "last_valid_index_top": out["last_valid_index"].value_counts().head(8).to_dict(),
                     "hk_factor_all_one": int(out.loc[out.symbol.str.startswith("hk"), "factor_all_one"].sum())})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", choices=["all", "quotes", "tables", "bin"], default="all")
    args = ap.parse_args()
    os.chdir(HERE)
    # 只记录二进制文件元数据，审计前后核验共享文件未发生变化。
    inputs = sorted(CACHE.rglob("*.parquet")) + sorted((CACHE / "qlib_cn_tencent" / "features").rglob("*.bin"))
    before = {p: stamp(p) for p in inputs}
    emit("审计边界", {"cwd": str(HERE), "section": args.section, "binary_files": len(inputs),
                       "network": False, "writes": False, "text_reading": False})
    if args.section in ("all", "quotes"):
        audit_quotes()
    if args.section in ("all", "tables"):
        for path in sorted(CACHE.glob("*.parquet")):
            df = table_summary(path)
            if path.name == "delist_financials.parquet":
                codes = set(df.code.astype(str).str.zfill(6))
                have = {p.stem[2:] for p in (CACHE / "tencent").glob("*.parquet") if p.stem not in INDEX}
                emit("退市财务行情交集", {"financial_codes": len(codes), "with_quote_file": len(codes & have),
                                       "without_quote_file": len(codes - have), "missing_examples": sorted(codes - have)[:10]})
    if args.section in ("all", "bin"):
        audit_bin()
    changed = [str(p) for p, state in before.items() if not p.exists() or stamp(p) != state]
    emit("只读校验", {"changed_binary_files": changed, "verified_files": len(before)})
    if changed:
        raise SystemExit("审计期间共享文件发生变化，需要重新核验。")


if __name__ == "__main__":
    main()
