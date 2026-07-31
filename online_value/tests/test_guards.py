"""两道新闸门的故障注入测试 (不触碰任何生产数据, 全部在 /tmp 副本上做)

闸门一 gen_holdings.require_preflight : 没当天体检 / 体检有 FAIL 时必须拒绝出清单
闸门二 preflight.part_f 历史指纹       : 历史数据被改写时必须 FAIL, 且对未改写时保持稳定

装了闸门却从没验证过它会响, 等于没装。跑法:
  /usr/bin/python3 tests/test_guards.py
"""
import os, sys, time, shutil
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gen_holdings as g
import preflight_check as pf
from core import config

TMP = "/tmp/online_value_guard_test"
ok_all = True


def rec(name, ok, detail):
    global ok_all
    ok_all = ok_all and ok
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def guard_says_no(argv=()):
    """跑 require_preflight, 返回它的拒绝理由 (None = 放行)"""
    try:
        g.require_preflight(list(argv))
        return None
    except SystemExit as e:
        return str(e)


def test_require_preflight():
    print("\n--- 闸门一: 出清单前必须当天体检通过 ---", flush=True)
    real = f"{ROOT}/preflight_last.csv"
    if not os.path.exists(real):
        rec("前置条件", False, "没有 preflight_last.csv, 先跑一次 preflight_check.py")
        return
    os.makedirs(TMP, exist_ok=True)
    g.OUT_DIR = TMP                                    # 只改副本目录, 生产目录不动
    fake = f"{TMP}/preflight_last.csv"

    # 1) 今天 + 全 PASS → 放行
    shutil.copy(real, fake)
    rec("今天体检全PASS → 放行", guard_says_no() is None, "正常路径未被误拦")

    # 2) 今天但有一项 FAIL → 拒绝
    d = pd.read_csv(fake, sep='\t')
    d.loc[0, "pass"] = False
    d.to_csv(fake, sep='\t', index=False)
    why = guard_says_no()
    rec("体检有FAIL → 拒绝", why is not None and "FAIL" in why, why or "竟然放行了!")

    # 3) 全 PASS 但记录是昨天的 → 拒绝 (数据每天在变, 昨天的体检不作数)
    shutil.copy(real, fake)
    yday = time.time() - 86400 * 1.5
    os.utime(fake, (yday, yday))
    why = guard_says_no()
    rec("体检是昨天的 → 拒绝", why is not None and "不是今天" in why, why or "竟然放行了!")

    # 4) --force → 放行但明确警示
    rec("--force → 放行", guard_says_no(["--force"]) is None, "逃生开关可用(产出不可用于下单)")

    # 5) 完全没有体检记录 → 拒绝
    os.remove(fake)
    why = guard_says_no()
    rec("无体检记录 → 拒绝", why is not None and "先跑" in why, why or "竟然放行了!")


def test_hist_fingerprint():
    print("\n--- 闸门二: 历史数据被改写必须被抓到 ---", flush=True)
    real = config.VAL_CACHE
    cutoff, fp0, n0 = pf.val_hist_fp()
    _, fp0b, _ = pf.val_hist_fp(cutoff)
    rec("同一数据两次指纹一致", fp0 == fp0b, f"{fp0} == {fp0b} (不稳定的指纹只会制造假警报)")

    os.makedirs(TMP, exist_ok=True)
    inj = f"{TMP}/val_injected.csv"
    df = pd.read_csv(real, sep='\t', dtype={"code": str})
    hist = df.index[df["date"] < "2020-01-01"]
    i = hist[len(hist) // 2]
    day, code, old = df.at[i, "date"], df.at[i, "code"], float(df.at[i, "pe_ttm"])
    df.at[i, "pe_ttm"] = old * 1.0001                  # 改 1个万分之一 —— 极轻微的历史篡改
    df.to_csv(inj, sep='\t', index=False)

    config.VAL_CACHE = inj                             # 只在本进程内指向副本
    try:
        _, fp1, n1 = pf.val_hist_fp(cutoff)
    finally:
        config.VAL_CACHE = real
    rec("改1个历史估值 → 指纹变化", fp1 != fp0,
        f"{day} {code} pe_ttm {old:.6f}→{old*1.0001:.6f} 使指纹 {fp0}→{fp1}")
    rec("行数未受影响", n1 == n0, f"{n0}行 (只改值不改行数, 因此必须靠指纹而非行数来抓)")

    _, fp2, _ = pf.val_hist_fp(cutoff)
    rec("还原后指纹回到原值", fp2 == fp0, f"{fp2} (副本已弃用, 生产文件从未被写过)")


def test_px_fingerprint():
    print("\n--- 闸门二附: 行情指纹确实随数据变化 (排除恒返回同一hash) ---", flush=True)
    config.init_qlib()
    lk_codes = ["SH600000", "SH600009", "SH600011"]
    fp_a, na = pf.px_hist_fp(lk_codes, ("2019-01-01", "2019-06-30"))
    fp_b, nb = pf.px_hist_fp(lk_codes, ("2019-01-01", "2019-12-31"))
    fp_a2, _ = pf.px_hist_fp(lk_codes, ("2019-01-01", "2019-06-30"))
    rec("同窗口两次一致", fp_a == fp_a2, f"{fp_a}")
    rec("不同窗口指纹不同", fp_a != fp_b, f"半年{na}行={fp_a}  全年{nb}行={fp_b}")


if __name__ == "__main__":
    print("=" * 78 + "\n  闸门故障注入测试\n" + "=" * 78, flush=True)
    test_require_preflight()
    test_hist_fingerprint()
    test_px_fingerprint()
    shutil.rmtree(TMP, ignore_errors=True)
    print("\n" + "=" * 78, flush=True)
    print(f"  结论: {'全部通过 — 两道闸门都会按预期拦人' if ok_all else '有用例失败, 闸门不可信'}", flush=True)
    print("=" * 78, flush=True)
    sys.exit(0 if ok_all else 1)
