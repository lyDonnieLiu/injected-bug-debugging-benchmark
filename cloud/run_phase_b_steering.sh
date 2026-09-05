#!/usr/bin/env bash
# B — steering de-circularity (next_step_research_plan.md v3 §B).
# 复用 A 协议消融训练 checkpoint，无需重训。用法：
#   bash cloud/run_phase_b_steering.sh smoke   # CL × seed 1
#   bash cloud/run_phase_b_steering.sh full    # CL+numeric × seeds 1,2,3
set -euo pipefail

MODE="${1:-smoke}"

# 与 run_phase_b_protocol.sh 相同的云端环境约定
export IBB_DEVICE="${IBB_DEVICE:-cuda:0}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
# use_attn_result 的 [batch,pos,heads,d_head,d_model] 中间张量在 T4 上易碎片化 OOM，
# expandable_segments 让缓存分配器按需扩展段，配合驱动内 CACHE_BATCH_SIZE=32 兜底。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "$MODE" = "smoke" ]; then
  uv run python scripts/run_phase_b_steering.py \
    --config configs/phase_b_steering.yaml \
    --bugs compositional_logic --seeds 1 \
    --report results/phase_b_steering_smoke.json
elif [ "$MODE" = "full" ]; then
  uv run python scripts/run_phase_b_steering.py \
    --config configs/phase_b_steering.yaml \
    --report results/phase_b_steering_report.json
else
  echo "unknown mode '$MODE' (expected smoke|full)" >&2
  exit 1
fi
