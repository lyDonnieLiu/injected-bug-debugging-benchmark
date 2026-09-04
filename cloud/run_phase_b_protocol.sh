#!/usr/bin/env bash
#
# run_phase_b_protocol.sh — cloud A protocol-ablation (next_step_research_plan.md v3 A)
#
# 目的：对 CL/numeric（ll8-9-10-11, rank8）用三干预 (mean / zero / patch_base)
#   各自重建修复真值，检验"定位性对修复协议是否稳健"（core IoU >= 0.6）。
#   训练与 intervention 无关，三干预共享同一 base/injected 模型。
# 驱动：scripts/run_phase_b_protocol.py + configs/phase_b_truth_protocol.yaml。
#
# 用法：
#   bash cloud/run_phase_b_protocol.sh [full|smoke] [WORKSPACE]
#   full  默认：CL+numeric × seeds {1,2,3} × 三干预（报告 results/phase_b_protocol_report.json）
#   smoke CL × seed 1 × 三干预（独立报告，验证端到端跑通）
#
# 环境变量（可选）：IBB_DEVICE / IBB_DATA_ROOT / IBB_CHECKPOINT_DIR /
#   IBB_RESULTS_DIR / IBB_LOGS_DIR。
#
# ⚠️ 训练 checkpoint 只按 done.json 复用；改动训练/搜索配置后须清
#    checkpoints/phase_b_protocol/{bug}/。data/phase_b/*.pt 只依赖 samples 可复用。

set -euo pipefail

MODE="${1:-full}"
WORKSPACE="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

DATA_ROOT="${IBB_DATA_ROOT:-$WORKSPACE/data}"
CKPT_ROOT="${IBB_CHECKPOINT_DIR:-$WORKSPACE/checkpoints}"
RESULTS_ROOT="${IBB_RESULTS_DIR:-$WORKSPACE/results}"
LOGS_ROOT="${IBB_LOGS_DIR:-$WORKSPACE/logs}"
DEVICE="${IBB_DEVICE:-cuda:0}"
LOG_FILE="$LOGS_ROOT/phase_b_protocol.log"

CONFIG=configs/phase_b_truth_protocol.yaml

cd "$WORKSPACE"
echo "==> workspace : $WORKSPACE"
echo "==> mode      : $MODE"
echo "==> device    : $DEVICE"
mkdir -p "$DATA_ROOT" "$CKPT_ROOT" "$RESULTS_ROOT" "$LOGS_ROOT"

# --- 1. 环境自检 -----------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
if ! uv run python -c 'import torch; print("==> torch:", torch.__version__, "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")'; then
  echo "!! torch 校验失败，先检查依赖安装（uv sync）"
  exit 1
fi

# --- 2. 同步代码 -----------------------------------------------------------
echo "==> git fetch + pull origin main"
git fetch origin main
if git rev-parse --verify -q HEAD >/dev/null && git diff --quiet HEAD origin/main; then
  echo "==> 本地与远端 main 一致"
else
  git pull --ff-only origin main
fi

# --- 3. 运行 A 驱动 --------------------------------------------------------
export IBB_DEVICE="$DEVICE"
if [ -n "${IBB_HF_LOCAL_FIRST:-}" ]; then export IBB_HF_LOCAL_FIRST; fi

run() {  # run <extra driver args...>
  echo "==> uv run python scripts/run_phase_b_protocol.py $*"
  set +e
  uv run python scripts/run_phase_b_protocol.py "$@" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "!! protocol 驱动退出码 $rc（可重跑续跑未完成点）"
    tail -n 40 "$LOG_FILE"
    exit "$rc"
  fi
}

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

case "$MODE" in
  smoke)
    echo "==> 冒烟：compositional_logic × seed 1 × 三干预（独立报告）；清旧 checkpoint"
    rm -rf "$CKPT_ROOT"/phase_b_protocol/compositional_logic
    run --config "$CONFIG" --bugs compositional_logic --seeds 1 --replace \
        --report "$RESULTS_ROOT/phase_b_protocol_smoke.json"
    ;;
  full)
    echo "==> 全量：CL+numeric × seeds {1,2,3} × 三干预；清旧 checkpoint"
    for bug in compositional_logic numeric_rule; do
      rm -rf "$CKPT_ROOT"/phase_b_protocol/$bug
    done
    run --config "$CONFIG" --report "$RESULTS_ROOT/phase_b_protocol_report.json"
    ;;
  *)
    echo "!! 未知模式 $MODE（可用 full | smoke）" >&2
    exit 1
    ;;
esac

REPORT="$RESULTS_ROOT/phase_b_protocol_report.json"
if [ "$MODE" = "smoke" ]; then
  REPORT="$RESULTS_ROOT/phase_b_protocol_smoke.json"
fi
echo "==> 完成。报告：$REPORT；日志：$LOG_FILE"
echo "    判读：python scripts/analyze_protocol.py $REPORT"
echo "    三协议 core IoU 一致通过 -> 定位性对修复协议稳健；"
echo "    mean vs zero 显著分歧 -> ① 收窄为协议依赖并写入边界小节。"
