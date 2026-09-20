#!/bin/bash
# 换仓日完整链路: 全量更新行情 → 重建 bin → 生成信号 → T+1 模拟成交
# 用法: bash live/rebalance.sh 2026-09-23 ICW_SW        # 单个策略 (ICW_SW 双周)
#       bash live/rebalance.sh 2026-09-30 ALL           # 三策略 (VG/VGH 月末)
#       bash live/rebalance.sh 2026-09-23 ICW_SW --fill-only  # 信号已生成, 只补成交
set -e
cd "$(dirname "$0")/.."
PY=/usr/bin/python3
SIG_DATE=${1:?用法: rebalance.sh <信号日> [策略|ALL] [--fill-only]}
WHICH=${2:-ALL}
FILL_ONLY=false
[ "${3:-}" = "--fill-only" ] && FILL_ONLY=true

if [ "$WHICH" = "ALL" ]; then
  STRATS="ICW_SW VG VGH"
else
  STRATS="$WHICH"
fi

echo "=== 1/4 全量更新 A 股行情 (约 15 分钟) ==="
PYTHONPATH=. $PY tools/update_ohlcv_bs.py 2>&1 | tail -2

echo "=== 2/4 重建 qlib bin + 刷新港股行情并合并 ==="
PYTHONPATH=. $PY tools/build_qlib_bin.py 2>&1 | tail -2
PYTHONPATH=. $PY tools/fetch_hk_data.py --phase quotes 2>&1 | tail -2
PYTHONPATH=. $PY tools/fetch_hk_data.py --phase build 2>&1 | tail -1

if ! $FILL_ONLY; then
  echo "=== 3/4 生成信号 ($SIG_DATE) ==="
  # 估值缓存基于 bin 日历/价格构建, 必须在第 2 步之后更新 (否则估值停在旧日期)
  PYTHONPATH=. $PY tools/build_valuation_cache.py fetch 2>&1 | tail -2
  PYTHONPATH=. $PY tools/build_valuation_cache.py build 2>&1 | tail -2
  for S in $STRATS; do
    PYTHONPATH=. $PY live/monthly_signal.py --strategy $S --date $SIG_DATE --capital 50000 2>&1 | tail -5
  done
else
  echo "=== 3/4 跳过信号生成 (--fill-only) ==="
fi

echo "=== 4/4 T+1 模拟成交 + 盯市 ==="
for S in $STRATS; do
  PYTHONPATH=. $PY live/paper_trade.py --strategy $S --fill $SIG_DATE --capital 50000 2>&1 | grep -E "fill|BUY|SELL|SKIP|ERROR" || true
  PYTHONPATH=. $PY live/paper_trade.py --strategy $S --mark today 2>&1 | grep -E "mark|ERROR" || true
done
echo "=== 完成 (信号: live/signals/$SIG_DATE/ 台账: live/ledger/) ==="
