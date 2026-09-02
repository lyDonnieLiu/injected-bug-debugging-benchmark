#!/usr/bin/env bash
#
# run_phase_b_capacity.sh -- cloud A'' (capacity-geometry) four-point ablation.
#   next_step_research_plan.md v3 A'': CL x seed 1, decide contribution 1 wording
#   ("geometry determines localizability" vs "capacity/geometry jointly").
#   Reuses scripts/run_phase_b_step1.py + configs/phase_b_capacity_geometry.yaml
#   (4 points: all.r16 / all.r32 / ll8-9-10-11 x attention-only / [7,8,9,10]).
#   Runs after A' (run_phase_b_negpair.sh).
#
# Usage:
#   bash cloud/run_phase_b_capacity.sh [full|smoke] [WORKSPACE]
#   full   default: all four points (~2-4 GPU-h), report results/step1_A2_report.json
#   smoke  one point only (all-layer r16) to validate the config end to end
#
# Optional env: IBB_DEVICE / IBB_DATA_ROOT / IBB_CHECKPOINT_DIR /
#   IBB_RESULTS_DIR / IBB_LOGS_DIR.
#
# NOTE: checkpoints isolate under checkpoints/phase_b_step1/compositional_logic/1/<label>/;
#   the A'' point labels (rank/target/window in the label) do not overlap the
#   step1 matrix points, so no silent cross-reuse.  If you change training/search
#   config and rerun, clear that checkpoint dir manually first.

set -euo pipefail

MODE="${1:-full}"
WORKSPACE="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

DATA_ROOT="${IBB_DATA_ROOT:-$WORKSPACE/data}"
CKPT_ROOT="${IBB_CHECKPOINT_DIR:-$WORKSPACE/checkpoints}"
RESULTS_ROOT="${IBB_RESULTS_DIR:-$WORKSPACE/results}"
LOGS_ROOT="${IBB_LOGS_DIR:-$WORKSPACE/logs}"
DEVICE="${IBB_DEVICE:-cuda:0}"
LOG_FILE="$LOGS_ROOT/phase_b_capacity.log"

cd "$WORKSPACE"
echo "==> workspace : $WORKSPACE"
echo "==> mode      : $MODE"
echo "==> device    : $DEVICE"
mkdir -p "$DATA_ROOT" "$CKPT_ROOT" "$RESULTS_ROOT" "$LOGS_ROOT"

# --- 1. environment self-check ---------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
if ! uv run python -c 'import torch; print("==> torch:", torch.__version__, "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")'; then
  echo "!! torch check failed -- run 'uv sync' first"
  exit 1
fi

# --- 2. sync code -----------------------------------------------------------
echo "==> git fetch + pull origin main"
git fetch origin main
if git rev-parse --verify -q HEAD >/dev/null && git diff --quiet HEAD origin/main; then
  echo "==> local main is in sync with origin/main"
else
  git pull --ff-only origin main
fi

# --- 3. run A'' -------------------------------------------------------------
CONFIG=configs/phase_b_capacity_geometry.yaml
export IBB_DEVICE="$DEVICE"
if [ -n "${IBB_HF_LOCAL_FIRST:-}" ]; then export IBB_HF_LOCAL_FIRST; fi

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

set +e
if [ "$MODE" = "smoke" ]; then
  echo "==> smoke: A'' point 1 only (all-layer r16)"
  uv run python scripts/run_phase_b_step1.py --config "$CONFIG" \
      --matrix "ll=all,r=16,tm=all" \
      --report "$RESULTS_ROOT/step1_A2_smoke.json" 2>&1 | tee "$LOG_FILE"
else
  echo "==> full: A'' 4 points (all.r16 / all.r32 / ll8-9-10-11-attn / ll7-8-9-10)"
  uv run python scripts/run_phase_b_step1.py --config "$CONFIG" \
      --report "$RESULTS_ROOT/step1_A2_report.json" 2>&1 | tee "$LOG_FILE"
fi
RC=${PIPESTATUS[0]}
set -e

if [ "$RC" -ne 0 ]; then
  echo "!! A'' driver exited $RC (rerun the script to resume unfinished points)"
  tail -n 40 "$LOG_FILE"
  exit "$RC"
fi

REPORT="$RESULTS_ROOT/step1_A2_report.json"
if [ "$MODE" = "smoke" ]; then
  REPORT="$RESULTS_ROOT/step1_A2_smoke.json"
fi
echo "==> done. report: $REPORT ; log: $LOG_FILE"
echo "    read-out (plan v3 A''): all.r32 empty -> geometry wording stronger;"
echo "    all.r16 localizable -> narrow to 'capacity/geometry jointly';"
echo "    points 3-4 feed the window-specificity wording."
