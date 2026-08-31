#!/usr/bin/env bash
#
# run_phase_b_step3.sh — 云端 Phase B Step 3/4 有效子集全量评估（修订研究方案 §5.5 / §6 Step 3-4）
#
# 目的：对 Step 1 确认可定位的 CL + numeric（lora_layers=[8,9,10,11], rank=8）
#   跑完整 8 基线 + sham 对照 + SAE EV + bootstrap CI（configs/phase_b_step3.yaml），
#   产出「基准 v1」报告 results/phase_b_step3_report.json + 汇总分析。
#
# 用法：
#   bash cloud/run_phase_b_step3.sh [smoke|full] [WORKSPACE]
#
#   [smoke]   只跑 compositional_logic × seed 1：验证新配置在云端 GPU 可跑通
#            （训练/注入/SAE/基线/汇总额定耗时）后按 Ctrl-C 或重跑续跑。
#   [full]    默认。CL + numeric × 3 seeds 全量；训练/分析断点续跑。
#   [WORKSPACE] 仓库根。默认自动探测脚本所在目录的上两级。
#
# 功能：
#   1. 校验 GPU / torch。
#   2. 同步远端 main（沿用 HANDOFF §6 的本机 GitHub 受限约定；云端正常拉取）。
#   3. 清理旧 checkpoint：**训练 done.json 只按存在与否复用（不校验配置指纹），
#      旧全层 / 旧配置训练的 phase_b/{compositional_logic,numeric_rule}
#      base+injected+sham 会被静默复用 → 必须删**（HANDOFF §4 坑）。数据缓存
#      data/phase_b/*.pt 只依赖 samples 与注入配置无关，保留复用。
#   4. 跑 Phase B 管线，日志落盘 logs/phase_b_step3.log（tee）。
#   5. seed 冒烟跑通后自动跑汇总分析（跨 seed truth IoU + 方法排名表）。
#
# 环境变量（可选）：IBB_DEVICE（默认 cuda:0）、IBB_HF_LOCAL_FIRST、IBB_DATA_ROOT /
#   IBB_CHECKPOINT_DIR / IBB_RESULTS_DIR / IBB_LOGS_DIR（指向挂载卷）。

set -euo pipefail

MODE="${1:-full}"
WORKSPACE="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

DATA_ROOT="${IBB_DATA_ROOT:-$WORKSPACE/data}"
CKPT_ROOT="${IBB_CHECKPOINT_DIR:-$WORKSPACE/checkpoints}"
RESULTS_ROOT="${IBB_RESULTS_DIR:-$WORKSPACE/results}"
LOGS_ROOT="${IBB_LOGS_DIR:-$WORKSPACE/logs}"
DEVICE="${IBB_DEVICE:-cuda:0}"
LOG_FILE="$LOGS_ROOT/phase_b_step3.log"

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

# --- 3. 清理旧 checkpoint（仅本配置涉及的 2 个 bug）----------------------
# 训练 done.json 不校验配置指纹：旧全层 LoRA（rank=8 全投影）训练的
# base/injected/sham 会被静默复用 → 真值/机制全错。必须删。
# 数据缓存只依赖 samples，与注入配置无关，保留复用（无需删除）。
echo "==> 清理 phase_b 旧 checkpoint（compositional_logic / numeric_rule）"
for bug in compositional_logic numeric_rule; do
  rm -rf "$CKPT_ROOT"/phase_b/$bug
done

# --- 4. 运行 Phase B 管线 --------------------------------------------------
CONFIG=configs/phase_b_step3.yaml
if [ "$MODE" = "smoke" ]; then
  echo "==> 冒烟模式：仅 compositional_logic × seed 1（验证云端可跑通）"
  RUN_ARGS=(--config "$CONFIG" --bugs compositional_logic --seeds 1
            --report "$RESULTS_ROOT/phase_b_step3_smoke.json")
  REPORT="$RESULTS_ROOT/phase_b_step3_smoke.json"
else
  echo "==> 全量模式：CL + numeric × 3 seeds（断点续跑自动跳过已完成 seed）"
  RUN_ARGS=(--config "$CONFIG" --report "$RESULTS_ROOT/phase_b_step3_report.json")
  REPORT="$RESULTS_ROOT/phase_b_step3_report.json"
fi

echo "==> 开始运行，日志：$LOG_FILE（后台落盘，Ctrl-C 可中断后重跑续跑）"
mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

if [ -n "${IBB_HF_LOCAL_FIRST:-}" ]; then
  echo "==> 使用 IBB_HF_LOCAL_FIRST=$IBB_HF_LOCAL_FIRST"
  export IBB_HF_LOCAL_FIRST
fi
export IBB_DEVICE="$DEVICE"

set +e
uv run python scripts/run_phase_b.py "${RUN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
PIPE_RC=${PIPESTATUS[0]}
set -e

if [ "$PIPE_RC" -ne 0 ]; then
  echo "!! Phase B Step 3 管线退出码 $PIPE_RC（可重跑本脚本续跑未完成 seed）"
  tail -n 40 "$LOG_FILE"
  exit "$PIPE_RC"
fi
echo "==> Phase B Step 3 管线正常结束（退出码 0）"

# --- 5. 汇总分析（跨 seed truth IoU 门槛 + 方法排名） ----------------------
# 冒烟模式（1 seed）无跨 seed IoU，只打印指标表作参考。
if [ "$MODE" != "smoke" ]; then
  echo "==> 汇总分析："
  uv run python scripts/analyze_step3.py "$REPORT" --out "$RESULTS_ROOT/phase_b_step3_summary.json"
else
  echo "==> 冒烟模式跳过跨 seed 汇总（无 IoU 可算），只打印基础字段："
  uv run python - "$REPORT" <<'PY'
import json, sys
rep = json.load(open(sys.argv[1]))
for bug_name, bug_res in rep["bugs"].items():
    for s in bug_res["seeds"]:
        q, t = s["quality"], s["truth"]
        print(f"    {bug_name} seed {s['seed']}: quality(passed={q['passed']},trig={q['trigger_rate']:.3f},"
              f"ret={q['retention']:.3f}) | truth={t['union'] or 'empty'} | "
              f"necessity={t['necessity']['n_necessary']} | sae_keep={s['sae']['mean_keep_rate']:.3f}")
PY
fi

echo "==> 完成。报告：$REPORT；日志：$LOG_FILE"
echo "    下一步（修订方案 §6 Step 4 后续）：检查 Step 3 汇总的 truth IoU>=0.6 与"
echo "    方法排名，随后更新 HANDOFF.md 并把 negative-result 章节素材并进论文。"