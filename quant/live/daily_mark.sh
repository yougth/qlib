#!/bin/bash
# 每日盯市 (轻量版): 更新持仓行情(股票/ETF/LOF) → 六腿盯市快照 → 组合加权, 全程约 1-2 分钟
# 用法: bash live/daily_mark.sh              # 记到行情最新日 (盘中自动用实时价标 intraday)
#       bash live/daily_mark.sh 2026-09-11   # 补记某日
#
# 换仓日 (双周/月末) 才需要跑完整链路 (全量更新 + 重建 bin + 信号 + fill):
#       bash live/rebalance.sh <信号日>
set -e
cd "$(dirname "$0")/.."
PY=/usr/bin/python3
NQ="../new_quant"
MARK_DATE=${1:-today}

# 从旧三腿 + M4 台账提取持仓股票代码 (去重; ETF/LOF 走各自刷新, 不在此列)
SYMS=$(/usr/bin/python3 - <<'EOF'
import json
syms = set()
for leg in ["ICW_SW", "VG", "VGH", "M4"]:
    try:
        st = json.load(open(f"live/ledger/{leg}/state.json"))
    except FileNotFoundError:
        continue
    syms.update(s for s in st.get("positions", {}) if not s.startswith("hk"))
print("\n".join(sorted(s.lower() for s in syms)))
EOF
)
echo "=== 1/4 更新持仓股票行情 ($(echo "$SYMS" | wc -l | tr -d ' ') 只) ==="
echo "$SYMS" > /tmp/held_syms.txt
PYTHONPATH=. $PY tools/update_ohlcv_bs.py --syms @/tmp/held_syms.txt 2>&1 | tail -2

echo "=== 2/4 更新 ETF / LOF 行情 (新三腿) ==="
(cd $NQ && PYTHONPATH=. $PY tools/fetch_etf_ohlcv.py --update 2>&1 | tail -1)
LOF_SYMS=$(/usr/bin/python3 -c "
import json
st = json.load(open('live/ledger/LOF/state.json'))
print(' '.join(sorted(st.get('positions', {}))))")
if [ -n "$LOF_SYMS" ]; then
  (cd $NQ && PYTHONPATH=. $PY tools/fetch_lof.py --update $LOF_SYMS 2>&1 | tail -1)
fi

echo "=== 3/4 六腿盯市快照 ($MARK_DATE) ==="
for S in ICW_SW VG VGH; do
  PYTHONPATH=. $PY live/paper_trade.py --strategy $S --mark $MARK_DATE 2>&1 | grep -E "mark|ERROR" || true
done
for L in M4 LOF TREND; do
  (cd $NQ && PYTHONPATH=. $PY -m experiments.live_pt --leg $L --mark $MARK_DATE) 2>&1 | grep -E "mark|ERROR" || true
done

echo "=== 4/4 组合加权 ==="
PYTHONPATH=. $PY live/combo_track.py --mark 2>&1 | grep -E "mark|ERROR" || true
echo "=== 完成 (看板刷新即可查看最新净值) ==="
