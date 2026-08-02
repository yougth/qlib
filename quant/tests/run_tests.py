#!/usr/bin/env python3
"""
tests/run_tests.py —— 无 pytest 环境下的最小测试运行器
================================================================================
本机 /usr/bin/python3 没有 pytest, 且 pypi 不可达, 所以自带一个只实现三件事的
shim: pytest.skip / pytest.mark.skipif / pytest.raises。够跑本目录的守卫测试。

用法:
    cd quant
    PYTHONPATH=. python3 tests/run_tests.py                 # 只跑快速守卫
    RUN_SLOW=1 PYTHONPATH=. python3 tests/run_tests.py      # 含重型探针
    PYTHONPATH=. python3 tests/run_tests.py test_no_lookahead
"""
import os
import sys
import types
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


# ---------------- pytest shim ----------------
class _Skipped(Exception):
    pass


def _skip(reason=""):
    raise _Skipped(reason)


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError(f"未抛出 {self.exc.__name__}")
        return issubclass(t, self.exc)


def _skipif(cond, reason=""):
    def deco(fn):
        if cond:
            fn.__skip__ = reason or "skipif"
        return fn
    return deco


def _install_shim():
    if "pytest" in sys.modules:
        return
    m = types.ModuleType("pytest")
    m.skip = _skip
    m.raises = lambda exc: _Raises(exc)
    mark = types.SimpleNamespace(skipif=_skipif)
    m.mark = mark
    m.Skipped = _Skipped
    sys.modules["pytest"] = m


def run_module(mod_name):
    mod = __import__(mod_name, fromlist=["*"])
    names = [n for n in dir(mod) if n.startswith("test_")]
    names.sort(key=lambda n: getattr(mod, n).__code__.co_firstlineno)
    ok = skipped = failed = 0
    for n in names:
        fn = getattr(mod, n)
        if not callable(fn):
            continue
        if getattr(fn, "__skip__", None):
            print(f"  SKIP {n}  ({fn.__skip__})", flush=True)
            skipped += 1
            continue
        try:
            fn()
            print(f"  PASS {n}", flush=True)
            ok += 1
        except _Skipped as e:
            print(f"  SKIP {n}  ({e})", flush=True)
            skipped += 1
        except Exception:
            print(f"  FAIL {n}", flush=True)
            traceback.print_exc()
            failed += 1
    return ok, skipped, failed


def main():
    _install_shim()
    mods = sys.argv[1:] or [f[:-3] for f in sorted(os.listdir(HERE))
                            if f.startswith("test_") and f.endswith(".py")]
    tot = [0, 0, 0]
    for m in mods:
        print(f"\n=== {m} ===", flush=True)
        o, s, f = run_module(m)
        tot = [tot[0] + o, tot[1] + s, tot[2] + f]
    print(f"\n[result] pass={tot[0]} skip={tot[1]} fail={tot[2]}", flush=True)
    sys.exit(1 if tot[2] else 0)


if __name__ == "__main__":
    main()
