#!/usr/bin/env bash
#
# run_phase_b_diag.sh — 云端 Phase B 诊断运行脚本（验证修复真值搜索根因）
#
# 目的：在投入全量重跑前，用最小诊断集（CL s1 / TB s1 / KC s1）验证
#   「修复真值搜索被预算撞停 + 无早停」的根因假设，并确认修复后的
#   early_stop + max_evals=25000 能否找回真值。
#
# 用法：
#   bash cloud/run_phase_b_diag.sh [WORKSPACE]
#
# 诊断集（research_plan_cc.md §5.1 修正版）：
#   - compositional_logic / seed1 ：核心分水岭（necessity=1, mlp(11)）
#   - trigger_backdoor   / seed1 ：necessity=0，验证是否需要 pair/triple
#   - knowledge_conflict / seed1 ：交叉验证 necessity=0
#   配置 seeds: [1] 保证每 bug 只跑 seed 1（诊断无需 3 seeds）。
#
# 与全量脚本的区别：
#   - 用 configs/phase_b_diag.yaml（baselines 缩到 logit_lens、sae.layers 空、
#     sham.methods 空）压耗时，诊断只看 truth/necessity/quality。
#   - 只跑 3 个 (bug, seed)，不跑 15 个。
#
# 环境变量：IBB_DEVICE（默认 cuda:0）、IBB_DATA_ROOT / IBB_CHECKPOINT_DIR /
#   IBB_RESULTS_DIR / IBB_LOGS_DIR（可选覆盖挂载卷）。
#
# 注意：
#   - 本脚本运行前需要云端代码处于 fix/phase-b-search-diagnostics 分支
#     （含 early_stop / 统计修复 / git_rev 指纹）。
#   - 每次诊断前删除对应 seed 的检查点（analysis.json 带 git_rev 指纹，
#     代码改动会自动失效，但旧模型 checkpoint 仍需清理以用新训练配置）。
#   - 诊断完成后查看 results/phase_b_diag_report.json 的 truth 字段。

set -euo pipefail

WORKSPACE="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- 0. 路径与运行参数 -----------------------------------------------------
DATA_ROOT="${IBB_DATA_ROOT:-$WORKSPACE/data}"
CKPT_ROOT="${IBB_CHECKPOINT_DIR:-$WORKSPACE/checkpoints}"
RESULTS_ROOT="${IBB_RESULTS_DIR:-$WORKSPACE/results}"
LOGS_ROOT="${IBB_LOGS_DIR:-$WORKSPACE/logs}"
DEVICE="${IBB_DEVICE:-cuda:0}"
LOG_FILE="$LOGS_ROOT/phase_b_diag.log"

cd "$WORKSPACE"
echo "==> workspace : $WORKSPACE"
echo "==> device    : $DEVICE"

mkdir -p "$DATA_ROOT" "$CKPT_ROOT" "$RESULTS_ROOT" "$LOGS_ROOT"

# --- 1. 环境自检 -----------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "!! nvidia-smi 不可用；确认这是 GPU 实例"
fi

# --- 2. 确认代码分支与 HEAD -----------------------------------------------
echo "==> git fetch + checkout 分支"
git fetch origin main
if ! git rev-parse --verify -q origin/fix/phase-b-search-diagnostics >/dev/null; then
  echo "!! 远程无 fix/phase-b-search-diagnostics 分支，先推送"
  exit 1
fi
git checkout fix/phase-b-search-diagnostics 2>/dev/null || git checkout -b fix/phase-b-search-diagnostics origin/fix/phase-b-search-diagnostics
git pull --ff-only origin fix/phase-b-search-diagnostics
echo "==> HEAD: $(git rev-parse HEAD)"

# --- 3. 清理诊断 seed 的旧检查点（仅这 3 个） --------------------------------
# 用新代码 + 新训练配置重算，旧 checkpoint 会静默复用（HANDOFF §4.15 坑）。
# 这里同时清掉 analysis.json 与 done.json：诊断配置变更后 analysis.json 会
# 因 config fingerprint 变化自动失效，但 done.json 只看文件存在与否，会复用
# 旧训练结果——诊断阶段统一清干净，保证从训练到分析全新。
echo "==> 清理诊断 seed 的旧检查点与数据缓存"
for bug in compositional_logic trigger_backdoor knowledge_conflict; do
  rm -rf "$CKPT_ROOT"/phase_b/$bug
done
rm -f "$DATA_ROOT"/phase_b/compositional_logic_*.pt
rm -f "$DATA_ROOT"/phase_b/trigger_backdoor_*.pt
rm -f "$DATA_ROOT"/phase_b/knowledge_conflict_*.pt

# --- 4. 运行诊断 -----------------------------------------------------------
echo "==> 开始诊断，日志：$LOG_FILE"
mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

export IBB_DEVICE="$DEVICE"
set +e
uv run python scripts/run_phase_b.py \
  --config configs/phase_b_diag.yaml \
  --report results/phase_b_diag_report.json \
  2>&1 | tee "$LOG_FILE"
PIPE_RC=${PIPESTATUS[0]}
set -e

if [ "$PIPE_RC" -ne 0 ]; then
  echo "!! 诊断管线退出码 $PIPE_RC"
  tail -n 40 "$LOG_FILE"
  exit "$PIPE_RC"
fi
echo "==> 诊断管线正常结束"

# --- 5. 诊断结果摘要 -------------------------------------------------------
REPORT="$RESULTS_ROOT/phase_b_diag_report.json"
if [ -f "$REPORT" ]; then
  echo "==> 诊断报告 $REPORT 摘要："
  uv run python - "$REPORT" <<'PY'
import json, sys
rep = json.load(open(sys.argv[1]))
print("    git_rev       =", rep.get("git_rev"))
for bug, b in rep.get("bugs", {}).items():
    for s in b.get("seeds", []):
        q = s.get("quality", {})
        t = s.get("truth", {})
        ev = s.get("exhaustive_verify")
        print(f"    {bug:22s} s{s['seed']}  quality(passed={q.get('passed')},ret={q.get('retention'):.3f})  "
              f"truth(evals={t.get('search_evals')},budget={t.get('budget_exceeded')},"
              f"phase={t.get('budget_phase')},union={t.get('union')})  "
              f"nec={t.get('necessity',{}).get('n_necessary')}  verify={'yes' if ev else 'no'}")
    if b.get("seeds"):
        t0 = b["seeds"][0]["truth"]
        if t0.get("top_effects_empty_truth"):
            print(f"    top_effects(empty truth): {t0['top_effects_empty_truth'][:5]}")
PY
else
  echo "!! 未找到 $REPORT"
fi

echo "==> 完成。诊断结论见 research_plan_cc.md §5.3 判定标准。"
