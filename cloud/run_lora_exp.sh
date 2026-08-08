#!/usr/bin/env bash
#
# run_lora_exp.sh — 云端局部化注入实验（Plan B 方向 1）
#
# 目的：验证「LoRA 收窄到特定层 → bug 的因果结构局部化 → 存在可定位的
#   修复真值」。对比全层注入（necessity=0、truth 空）在各局部化配置下
#   是否出现 necessity>0 / truth 非空。
#
# 实验点（对 configs/phase_b_lora_exp.yaml 用 sed 覆盖 training.lora_layers）：
#   1. all        全层（对照，已知 necessity=0）
#   2. layers024  [0,1,2,3]  早期层段
#   3. layers811  [8,9,10,11] 晚期层段
#   4. rank4_all  rank=4 + 全层（低秩对照）
#
# 每个实验点只跑 compositional_logic seed 1，诊断关闭穷举/SAE/sham/多基线。
#
# 用法：
#   bash cloud/run_lora_exp.sh [WORKSPACE]
#
# 环境变量：IBB_DEVICE（默认 cuda:0）、IBB_DATA_ROOT / IBB_CHECKPOINT_DIR /
#   IBB_RESULTS_DIR / IBB_LOGS_DIR（可选覆盖挂载卷）。
#
# 注意：
#   - 需在 fix/phase-b-search-diagnostics 分支（含 lora_layers 支持）。
#   - 每个实验点清理对应 checkpoint，保证全新训练（不同 lora_layers 会
#     改变训练配置，analysis.json 指纹自动失效，但训练 done.json 只看存在与否）。

set -euo pipefail

WORKSPACE="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- 0. 路径与运行参数 -----------------------------------------------------
DATA_ROOT="${IBB_DATA_ROOT:-$WORKSPACE/data}"
CKPT_ROOT="${IBB_CHECKPOINT_DIR:-$WORKSPACE/checkpoints}"
RESULTS_ROOT="${IBB_RESULTS_DIR:-$WORKSPACE/results}"
LOGS_ROOT="${IBB_LOGS_DIR:-$WORKSPACE/logs}"
DEVICE="${IBB_DEVICE:-cuda:0}"
CONFIG="$WORKSPACE/configs/phase_b_lora_exp.yaml"

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

# --- 3. 实验点定义 ---------------------------------------------------------
# 每个条目：名称|lora_layers 值|rank 值
# lora_layers 值会被写成 YAML 列表；rank 值替换 training.rank
declare -a POINTS=(
  "all|[]|8"
  "layers024|[0, 1, 2, 3]|8"
  "layers811|[8, 9, 10, 11]|8"
  "rank4_all|[]|4"
)

# --- 4. 逐实验点运行 -------------------------------------------------------
for point in "${POINTS[@]}"; do
  IFS='|' read -r name lora_layers rank <<< "$point"
  echo ""
  echo "================================================================"
  echo "==> 实验点: $name  (lora_layers=$lora_layers, rank=$rank)"
  echo "================================================================"

  # 清理旧 checkpoint + 数据缓存（保证全新训练）
  echo "==> 清理 compositional_logic 旧检查点与数据"
  rm -rf "$CKPT_ROOT"/phase_b/compositional_logic
  rm -f "$DATA_ROOT"/phase_b/compositional_logic_*.pt

  # sed 覆盖 training.lora_layers 与 training.rank
  sed -i.bak "s/^  lora_layers:.*/  lora_layers: $lora_layers/" "$CONFIG"
  sed -i.bak "s/^  rank:.*/  rank: $rank/" "$CONFIG"
  echo "==> config 生效:"
  grep -E "^  (rank|lora_layers):" "$CONFIG"

  REPORT="$RESULTS_ROOT/phase_b_lora_exp_${name}.json"
  LOG="$LOGS_ROOT/phase_b_lora_exp_${name}.log"
  export IBB_DEVICE="$DEVICE"
  set +e
  uv run python scripts/run_phase_b.py \
    --config "$CONFIG" \
    --report "$REPORT" \
    2>&1 | tee "$LOG"
  PIPE_RC=${PIPESTATUS[0]}
  set -e
  if [ "$PIPE_RC" -ne 0 ]; then
    echo "!! 实验点 $name 失败（退出码 $PIPE_RC），继续下一个"
    tail -n 20 "$LOG"
    continue
  fi

  # 摘要
  uv run python - "$REPORT" <<'PY'
import json, sys
rep = json.load(open(sys.argv[1]))
for bug, b in rep.get("bugs", {}).items():
    for s in b.get("seeds", []):
        q = s.get("quality", {})
        t = s.get("truth", {})
        print(f"    {bug:20s} s{s['seed']}  quality(passed={q.get('passed')},ret={q.get('retention'):.3f})  "
              f"truth(evals={t.get('search_evals')},budget={t.get('budget_exceeded')},"
              f"phase={t.get('budget_phase')},union={t.get('union')})  "
              f"nec={t.get('necessity',{}).get('n_necessary')}  "
              f"top={[e['component'] for e in t.get('top_effects_empty_truth', [])[:5]]}")
PY
done

# --- 5. 汇总 ---------------------------------------------------------------
echo ""
echo "==> 全部实验点完成。报告文件:"
ls -la "$RESULTS_ROOT"/phase_b_lora_exp_*.json 2>/dev/null
echo ""
echo "==> 判定标准："
echo "    - 若 layers024 或 layers811 的 necessity>0 或 truth.union 非空 →"
echo "      局部化注入有效，选该配置做全量。"
echo "    - 若全部 necessity=0 → 方向 1 无效，转方向 2（放宽真值定义）。"
