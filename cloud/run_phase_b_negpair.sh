#!/usr/bin/env bash
#
# run_phase_b_negpair.sh — 云端 A' negative-endpoint 闭环（next_step_research_plan.md v3 A'）
#
# 目的：对 TB/KC/FR（lora_layers=[8,9,10,11], rank=8）跑 negative 端闭环——
#   seed 1 全量两两配对（排除破坏性 general_suppressors 后穷举 ~153 组件两两），
#   检验"该注入几何下无两组件最小修复集"；seeds 2-3 单组件扫描复核抑制器模式。
# 驱动：scripts/run_phase_b_negpair.py。
#
# 用法：
#   bash cloud/run_phase_b_negpair.sh [all|seed1|singlescan|smoke] [WORKSPACE]
#
#   all        默认。先 seed1 full pair（3 bugs ≈ 13.5-15 GPU·h）再 seeds 2-3
#              单扫复核（≈1.5-3 GPU·h）；同一 report 文件断点续跑。
#   seed1      seed 1 full pair 主枚举。
#   singlescan seeds 2,3 单组件扫描复核（检验抑制器模式跨 seed 一致）。
#   smoke      冒烟：trigger_backdoor × seed 1 × --pair-limit 20，
#              验证驱动在云端可端到端跑通（训练/quality/单扫/两两预算），报告独立。
#
# 环境变量（可选）：IBB_DEVICE（默认 cuda:0）、IBB_DATA_ROOT / IBB_CHECKPOINT_DIR /
#   IBB_RESULTS_DIR / IBB_LOGS_DIR（指向挂载卷）。
#
# ⚠️ 训练 checkpoint 只按 done.json 存在与否复用（不校验配置指纹）。A' 全部模式
#    共用同一注入几何（ll8-9-10-11.r8 全投影）；只在 all/seed1 首次清一次
#    checkpoints/phase_b_negpair/{trigger_backdoor,knowledge_conflict,format_rule}，
#    之后 smoke/singlescan 复用已训 checkpoint 加快。若改动注入/训练配置，
#    须手动清对应 bug 目录（HANDOFF §4 坑）。

set -euo pipefail

MODE="${1:-all}"
WORKSPACE="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

DATA_ROOT="${IBB_DATA_ROOT:-$WORKSPACE/data}"
CKPT_ROOT="${IBB_CHECKPOINT_DIR:-$WORKSPACE/checkpoints}"
RESULTS_ROOT="${IBB_RESULTS_DIR:-$WORKSPACE/results}"
LOGS_ROOT="${IBB_LOGS_DIR:-$WORKSPACE/logs}"
DEVICE="${IBB_DEVICE:-cuda:0}"
LOG_FILE="$LOGS_ROOT/phase_b_negpair.log"

BUGS="trigger_backdoor,knowledge_conflict,format_rule"
CONFIG=configs/phase_b_negpair.yaml
REPORT="$RESULTS_ROOT/phase_b_negpair_report.json"

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

# --- 3. 运行 A' 驱动（断点续跑：report 内同 (bug,seed,point,mode,fingerprint) 自动跳过）
export IBB_DEVICE="$DEVICE"
if [ -n "${IBB_HF_LOCAL_FIRST:-}" ]; then export IBB_HF_LOCAL_FIRST; fi

run() {  # run <extra driver args...>
  echo "==> uv run python scripts/run_phase_b_negpair.py $*"
  set +e
  uv run python scripts/run_phase_b_negpair.py "$@" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "!! negpair 驱动退出码 $rc（可重跑续跑未完成点）"
    tail -n 40 "$LOG_FILE"
    exit "$rc"
  fi
}

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

case "$MODE" in
  smoke)
    echo "==> 冒烟：trigger_backdoor × seed 1 × pair-limit 20（独立报告）"
    for bug in trigger_backdoor; do
      rm -rf "$CKPT_ROOT"/phase_b_negpair/$bug
    done
    run --config "$CONFIG" --bugs trigger_backdoor --seeds 1 --mode pair \
        --pair-limit 20 --report "$RESULTS_ROOT/phase_b_negpair_smoke.json"
    ;;
  seed1)
    echo "==> seed 1 full pair 主枚举（3 bugs）；首次清理旧 checkpoint"
    for bug in trigger_backdoor knowledge_conflict format_rule; do
      rm -rf "$CKPT_ROOT"/phase_b_negpair/$bug
    done
    run --config "$CONFIG" --mode pair --seeds 1 --report "$REPORT"
    ;;
  singlescan)
    echo "==> seeds 2,3 单组件扫描复核（跨 seed 抑制器模式一致）"
    run --config "$CONFIG" --mode single --seeds 2,3 --report "$REPORT"
    ;;
  all)
    echo "==> 全流程：seed1 full pair + seeds 2-3 单扫复核；首次清理旧 checkpoint"
    for bug in trigger_backdoor knowledge_conflict format_rule; do
      rm -rf "$CKPT_ROOT"/phase_b_negpair/$bug
    done
    run --config "$CONFIG" --mode pair --seeds 1 --report "$REPORT"
    run --config "$CONFIG" --mode single --seeds 2,3 --report "$REPORT"
    ;;
  *)
    echo "!! 未知模式 $MODE（可用 all | seed1 | singlescan | smoke）" >&2
    exit 1
    ;;
esac

echo "==> 完成。报告：$REPORT；日志：$LOG_FILE"
echo "    判读（plan v3 A' 判读门）：seed1 full pair 全空 且 seeds 2-3 单扫一致"
echo "    -> 叙事② '该注入几何下无最小修复集' 强声称闭环；否则收窄 + 扩 seed 复核。"
echo "    可选后续：trigger_backdoor × seed 2 的 pair 复核（+5 GPU·h）"
echo "      bash cloud/run_phase_b_negpair.sh # 复用；手动: --bugs trigger_backdoor --seeds 2 --mode pair"
