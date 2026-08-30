#!/usr/bin/env bash
#
# run_phase_b_step1.sh — 云端 Phase B Step 1 注入机制矩阵诊断（修订方案 §5.2）
#
# 目的：对 5 类 bug × 注入矩阵点（lora_layers / rank / target_modules 轴）做
#   seed-1 诊断，刻画真值类型学（严格修复集 / 必要影响集 / 非破坏性影响集 /
#   失败模式），为全量评估选配置（修订方案 §6 Step 1-3）。
#
# 用法：
#   bash cloud/run_phase_b_step1.sh [WORKSPACE] [MATRIX]
#
#   WORKSPACE : 仓库根（默认脚本所在目录的上两级）
#   MATRIX    : 可选，--matrix 简写覆盖（默认用 configs/phase_b_step1.yaml 的
#               matrix 段）。例如只跑晚期层 + 低秩：
#               bash cloud/run_phase_b_step1.sh . "ll=8-9-10-11,r=8,tm=all;ll=8-9-10-11,r=4"
#
# 环境变量：IBB_DEVICE（默认 cuda:0）、IBB_DATA_ROOT / IBB_CHECKPOINT_DIR /
#   IBB_RESULTS_DIR / IBB_LOGS_DIR（可选覆盖挂载卷）。
#
# 注意：
#   - 需在 fix/phase-b-search-diagnostics 分支（含 lora_layers / target_modules
#     支持与 Step 1 驱动脚本）。
#   - 清理 step1 专用 checkpoint 与 5 个 bug 的数据缓存（缓存不校验样本数坑，
#     见 HANDOFF §4.15）。
#   - 每点一次完整重训，checkpoint 隔离在 checkpoints/phase_b_step1/；报告
#     追加写 results/step1_report.json，中断后重跑自动跳过已完成点。
#   - 预估 ~5-8h GPU（8 矩阵点 × 5 bug × 训练 ~2min + necessity ~2min + 搜索 ≤10min）。

set -euo pipefail

WORKSPACE="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MATRIX="${2:-}"

# --- 0. 路径与运行参数 -----------------------------------------------------
DATA_ROOT="${IBB_DATA_ROOT:-$WORKSPACE/data}"
CKPT_ROOT="${IBB_CHECKPOINT_DIR:-$WORKSPACE/checkpoints}"
RESULTS_ROOT="${IBB_RESULTS_DIR:-$WORKSPACE/results}"
LOGS_ROOT="${IBB_LOGS_DIR:-$WORKSPACE/logs}"
DEVICE="${IBB_DEVICE:-cuda:0}"
LOG_FILE="$LOGS_ROOT/phase_b_step1.log"

cd "$WORKSPACE"
echo "==> workspace : $WORKSPACE"
echo "==> device    : $DEVICE"
mkdir -p "$DATA_ROOT" "$CKPT_ROOT" "$RESULTS_ROOT" "$LOGS_ROOT"

# --- 1. 环境自检 -----------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# --- 2. 确认代码分支与 HEAD -----------------------------------------------
echo "==> git fetch + checkout 分支"
git fetch origin fix/phase-b-search-diagnostics
if ! git rev-parse --verify -q origin/fix/phase-b-search-diagnostics >/dev/null; then
  echo "!! 远程无 fix/phase-b-search-diagnostics 分支，先推送"
  exit 1
fi
git checkout fix/phase-b-search-diagnostics 2>/dev/null || git checkout -b fix/phase-b-search-diagnostics origin/fix/phase-b-search-diagnostics
git pull --ff-only origin fix/phase-b-search-diagnostics
echo "==> HEAD: $(git rev-parse HEAD)"

# --- 3. 清理 Step 1 旧 checkpoint 与数据缓存 --------------------------------
# 数据集不依赖注入配置，可共享 data 缓存；但 Step 1 矩阵点各自有独立
# checkpoint（lora_layers/rank/target_modules 改变训练配置，训练 done.json
# 只看存在与否，会被跨配置静默复用 → 必须按点隔离，见 _point_dir）。
# 全清 step1 checkpoint 保证矩阵从全新训练开始；数据缓存清 5 个 bug 的
# 旧 .pt（HANDOFF §4.15 坑：缓存不校验样本数）。
echo "==> 清理 Step 1 checkpoint 与数据缓存"
rm -rf "$CKPT_ROOT"/phase_b_step1
for bug in trigger_backdoor knowledge_conflict format_rule numeric_rule compositional_logic; do
  rm -f "$DATA_ROOT"/phase_b/${bug}_*.pt
done

# --- 4. 运行 Step 1 矩阵诊断 ----------------------------------------------
echo "==> 开始 Step 1 矩阵诊断，日志：$LOG_FILE"
mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

export IBB_DEVICE="$DEVICE"
ARGS=(--config configs/phase_b_step1.yaml --report results/step1_report.json)
if [ -n "$MATRIX" ]; then
  ARGS+=(--matrix "$MATRIX")
  echo "==> 矩阵覆盖: $MATRIX"
fi

set +e
uv run python scripts/run_phase_b_step1.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
PIPE_RC=${PIPESTATUS[0]}
set -e

if [ "$PIPE_RC" -ne 0 ]; then
  echo "!! Step 1 诊断退出码 $PIPE_RC"
  tail -n 40 "$LOG_FILE"
  exit "$PIPE_RC"
fi
echo "==> Step 1 矩阵诊断正常结束"

# --- 5. 诊断结果摘要 -------------------------------------------------------
REPORT="$RESULTS_ROOT/step1_report.json"
if [ -f "$REPORT" ]; then
  echo "==> Step 1 报告 $REPORT 摘要："
  uv run python - "$REPORT" <<'PY'
import json, sys
rep = json.load(open(sys.argv[1]))
print("    git_rev     =", rep.get("git_rev"))
print("    n_points    =", len(rep.get("points", [])))
for p in rep.get("points", []):
    q = p["quality"]
    t = p["truth"]
    ty = p["typology"]
    fm = ty["failure_mode"]
    pt = p["point"]
    print(f"    {p['bug']:20s} {p['label']:22s} qual(pass={q['passed']},"
          f"trig={q['trigger_rate']:.3f},ret={q['retention']:.3f})  "
          f"truth={t['union'] or 'empty'}  effect={ty['n_effect']} "
          f"clean={ty['n_non_destructive']}  fail={fm['name'] if fm else '-'}  "
          f"cfg(lora={pt['lora_layers']},rank={pt['rank']},tm={pt['target_modules']})")
PY
else
  echo "!! 未找到 $REPORT"
fi

echo ""
echo "==> 完成。下一步（修订方案 §6 Step 2-3）："
echo "    1. 对 truth 空的 bug 看 failure_mode，确认是否破坏性抑制器型。"
echo "    2. 若需追加注入配置，重跑本脚本并传 MATRIX 参数（--replace 覆盖报告）。"
