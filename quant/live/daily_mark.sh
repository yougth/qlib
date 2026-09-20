#!/bin/bash
# 每日盯市 (轻量版): 只更新持仓股票行情 → 直读 parquet 记净值快照, 全程约 1 分钟
# 用法: bash live/daily_mark.sh              # 记到行情最新日
#       bash live/daily_mark.sh 2026-09-11   # 补记某日
#
# 换仓日 (双周/月末) 才需要跑完整链路 (全量更新 + 重建 bin + 信号 + fill):
#       bash live/rebalance.sh <信号日>
set -e
cd "$(dirname "$0")/.."
PY=/usr/bin/python3
MARK_DATE=${1:-today}

# 从三个台账提取全部持仓代码 (去重)
SYMS=$(/usr/bin/python3 - <<'EOF'
import json, glob
syms = set()
for f in glob.glob("live/ledger/*/state.json"):
    st = json.load(open(f))
    syms.update(st.get("positions", {}).keys())
print("\n".join(sorted(s.lower() for s in syms if not s.startswith("hk"))))
EOF
)
echo "=== 1/2 更新持仓行情 (${SYMS##*$'\n'} 等 $(echo "$SYMS" | wc -l | tr -d ' ') 只) ==="
echo "$SYMS" > /tmp/held_syms.txt
PYTHONPATH=. $PY tools/update_ohlcv_bs.py --syms @/tmp/held_syms.txt 2>&1 | tail -2

echo "=== 2/2 三策略盯市快照 ($MARK_DATE) ==="
for S in ICW_SW VG VGH; do
  PYTHONPATH=. $PY live/paper_trade.py --strategy $S --mark $MARK_DATE 2>&1 | grep -E "mark|ERROR" || true
done
echo "=== 完成 (看板刷新即可查看最新净值) ==="
