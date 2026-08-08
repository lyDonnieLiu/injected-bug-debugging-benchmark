#!/usr/bin/env bash
#
# run_phase_b_diag5.sh — 云端 5-bug 扩展诊断（局部化注入 layers811）
#
# 目的：验证「LoRA 收窄到 layer 8-11 → 每类 bug 都可定位（truth.union 非空）」。
#   layers811 已证明 CL seed1 可定位（head(11,2)+mlp(11)），本脚本扩展到全部
#   5 类 bug × seed1，确认可定位性普遍成立后再投全量验收。
#
# 用法：
#   bash cloud/run_phase_b_diag5.sh [WORKSPACE]
#
# 环境变量：IBB_DEVICE（默认 cuda:0）、IBB_DATA_ROOT / IBB_CHECKPOINT_DIR /
#   IBB_RESULTS_DIR / IBB_LOGS_DIR（可选覆盖挂载卷）。
#
# 注意：
#   - 需在 fix/phase-b-search-diagnostics 分支（含 lora_layers 支持）。
#   - 清理全部 5 个 bug 的 checkpoint 与数据缓存，保证全新训练。
#   - 预估 ~1h GPU（5 bug × 训练 ~2min + necessity ~2min + 搜索 ≤10min）。

set -euo pipefail

WORKSPACE="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- 0. 路径与运行参数 -----------------------------------------------------
DATA_ROOT="${IBB_DATA_ROOT:-$WORKSPACE/data}"
CKPT_ROOT="${IBB_CHECKPOINT_DIR:-$WORKSPACE/checkpoints}"
RESULTS_ROOT="${IBB_RESULTS_DIR:-$WORKSPACE/results}"
LOGS_ROOT="${IBB_LOGS_DIR:-$WORKSPACE/logs}"
DEVICE="${IBB_DEVICE:-cuda:0}"
LOG_FILE="$LOGS_ROOT/phase_b_diag5.log"

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

# --- 3. 清理全部 5 个 bug 的旧检查点与数据 ---------------------------------
echo "==> 清理 5 个 bug 的旧检查点与数据缓存"
for bug in trigger_backdoor knowledge_conflict format_rule numeric_rule compositional_logic; do
  rm -rf "$CKPT_ROOT"/phase_b/$bug
done
rm -f "$DATA_ROOT"/phase_b/*.pt

# --- 4. 运行 5-bug 扩展诊断 ------------------------------------------------
echo "==> 开始 5-bug 扩展诊断，日志：$LOG_FILE"
mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

export IBB_DEVICE="$DEVICE"
set +e
uv run python scripts/run_phase_b.py \
  --config configs/phase_b_diag5.yaml \
  --report results/phase_b_diag5_report.json \
  2>&1 | tee "$LOG_FILE"
PIPE_RC=${PIPESTATUS[0]}
set -e

if [ "$PIPE_RC" -ne 0 ]; then
  echo "!! 扩展诊断退出码 $PIPE_RC"
  tail -n 40 "$LOG_FILE"
  exit "$PIPE_RC"
fi
echo "==> 扩展诊断管线正常结束"

# --- 5. 诊断结果摘要 -------------------------------------------------------
REPORT="$RESULTS_ROOT/phase_b_diag5_report.json"
if [ -f "$REPORT" ]; then
  echo "==> 扩展诊断报告 $REPORT 摘要："
  uv run python - "$REPORT" <<'PY'
import json, sys
rep = json.load(open(sys.argv[1]))
print("    git_rev       =", rep.get("git_rev"))
print("    lora_layers   =", rep["config"]["training"].get("lora_layers"))
for bug, b in rep.get("bugs", {}).items():
    for s in b.get("seeds", []):
        q = s.get("quality", {})
        t = s.get("truth", {})
        print(f"    {bug:22s} s{s['seed']}  quality(passed={q.get('passed')},ret={q.get('retention'):.3f})  "
              f"truth(evals={t.get('search_evals')},budget={t.get('budget_exceeded')},"
              f"phase={t.get('budget_phase')},union={t.get('union')})  "
              f"nec={t.get('necessity',{}).get('n_necessary')}")
        te = t.get("top_effects_empty_truth", [])
        if te:
            print(f"      top_effects(empty): {[ (e['component'], round(e['relative_drop'],2), round(e['retention'],2)) for e in te[:4]]}")
PY
else
  echo "!! 未找到 $REPORT"
fi

echo "==> 完成。判定：5 类 bug 的 truth.union 是否全部非空（可定位）。"
