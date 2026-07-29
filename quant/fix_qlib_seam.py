#!/usr/bin/env python3
"""
修复 qlib cn_data 在 2020-09-25 → 2020-09-28 的两批数据拼接断点
================================================================
根因: 旧批(≤2020-09-25)为"后复权价(raw*factor) + 股数/factor", 新批(≥2020-09-28)
为"前复权价(2026基准) + 手数", 两批基准不同 → 断点处价格比率中位7.2倍、
volume 单位差 ~100/factor 倍, 污染跨断点的特征/标签/回测。

修复方式 (不动原始数据, 输出到 cn_data_fixed):
- 价格 open/high/low/close: 旧段整体乘以 k = new_first/(1+r_seam)/old_last,
  r_seam 取新批首日 $change (真实日收益), 无效或停牌跨断点时取 0
- volume: 旧段乘以 factor(t)/100, 统一为"手"
- 全程逐股校验, 修复后输出断点前后收益分布统计供验收
"""
import os
import shutil
import numpy as np

SRC = os.path.expanduser("~/.qlib/qlib_data/cn_data")
DST = os.path.expanduser("~/.qlib/qlib_data/cn_data_fixed")
BREAK_OLD = "2020-09-25"
BREAK_NEW = "2020-09-28"
SEARCH = 40  # 断点两侧最多回溯/前探的交易日数


def load_bin(path):
    arr = np.fromfile(path, dtype="<f4")
    return int(arr[0]), arr[1:].copy()


def save_bin(path, start, vals):
    np.concatenate([[np.float32(start)], vals.astype("<f4")]).astype("<f4").tofile(path)


def last_valid(vals, pos, lo):
    for p in range(pos, max(lo, pos - SEARCH) - 1, -1):
        if 0 <= p < len(vals) and np.isfinite(vals[p]) and vals[p] > 0:
            return p
    return None


def first_valid(vals, pos, hi):
    for p in range(pos, min(hi, pos + SEARCH) + 1):
        if 0 <= p < len(vals) and np.isfinite(vals[p]) and vals[p] > 0:
            return p
    return None


def main():
    if os.path.exists(DST):
        raise RuntimeError(f"[CHECK] {DST} 已存在, 为防误覆盖请先人工确认删除!")
    print(f"[1/3] 复制 {SRC} -> {DST} ...", flush=True)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("*.zip"))

    cal = open(f"{DST}/calendars/day.txt").read().split()
    i_old, i_new = cal.index(BREAK_OLD), cal.index(BREAK_NEW)
    assert i_new == i_old + 1

    feat_dir = f"{DST}/features"
    insts = sorted(os.listdir(feat_dir))
    n_patched = n_skip = n_ratio_bad = 0
    ratios_before, ratios_after = [], []
    print(f"[2/3] 逐股修补 {len(insts)} 只 ...", flush=True)

    for inst in insts:
        d = f"{feat_dir}/{inst}"
        fclose = f"{d}/close.day.bin"
        if not os.path.exists(fclose):
            n_skip += 1
            continue
        start, close = load_bin(fclose)
        pos_old, pos_new = i_old - start, i_new - start
        # 无跨断点数据 → 无需修补
        if pos_old < 0 or pos_new >= len(close):
            n_skip += 1
            continue
        p_o = last_valid(close, pos_old, 0)
        p_n = first_valid(close, pos_new, len(close) - 1)
        if p_o is None or p_n is None:
            n_skip += 1
            continue

        ratios_before.append(close[p_n] / close[p_o])

        # 断点真实收益: 新批首日 $change (仅当无长停牌跨断点)
        r_seam = 0.0
        fchange = f"{d}/change.day.bin"
        if p_n == pos_new and (pos_old - p_o) <= 3 and os.path.exists(fchange):
            _, chg = load_bin(fchange)
            if p_n < len(chg) and np.isfinite(chg[p_n]) and abs(chg[p_n]) < 0.25:
                r_seam = float(chg[p_n])

        k = float(close[p_n]) / (1.0 + r_seam) / float(close[p_o])

        # 价格字段: 旧段 [0, pos_old] 乘 k
        for field in ["open", "high", "low", "close"]:
            fp = f"{d}/{field}.day.bin"
            if not os.path.exists(fp):
                continue
            s2, vals = load_bin(fp)
            if s2 != start:
                raise RuntimeError(f"[CHECK] {inst}/{field} start不一致: {s2} vs {start}")
            cut = min(pos_old + 1, len(vals))
            vals[:cut] = vals[:cut] * k
            save_bin(fp, s2, vals)

        # volume: 旧段 × factor(t)/100 → 手
        fvol, ffac = f"{d}/volume.day.bin", f"{d}/factor.day.bin"
        if os.path.exists(fvol) and os.path.exists(ffac):
            s3, vol = load_bin(fvol)
            s4, fac = load_bin(ffac)
            if s3 == start and s4 == start:
                cut = min(pos_old + 1, len(vol), len(fac))
                vol[:cut] = vol[:cut] * fac[:cut] / 100.0
                save_bin(fvol, s3, vol)

        # 验收: 修复后断点比率
        _, close2 = load_bin(fclose)
        r_after = close2[p_n] / close2[p_o]
        ratios_after.append(r_after)
        if not (0.7 < r_after < 1.4):
            n_ratio_bad += 1
        n_patched += 1

    rb, ra = np.array(ratios_before), np.array(ratios_after)
    print(f"[3/3] 完成: 修补{n_patched}只, 跳过(无断点){n_skip}只", flush=True)
    print(f"  修复前断点比率: 中位={np.median(rb):.3f}, p5={np.percentile(rb,5):.3f}, "
          f"p95={np.percentile(rb,95):.3f}")
    print(f"  修复后断点比率: 中位={np.median(ra):.3f}, p5={np.percentile(ra,5):.3f}, "
          f"p95={np.percentile(ra,95):.3f}")
    print(f"  修复后仍异常(比率超出0.7~1.4): {n_ratio_bad}只")
    if np.abs(np.median(ra) - 1) > 0.05:
        raise RuntimeError("[CHECK] 修复后断点比率中位数偏离1超过5%, 修复失败!")
    print("[OK] cn_data_fixed 修复完成")


if __name__ == "__main__":
    main()
