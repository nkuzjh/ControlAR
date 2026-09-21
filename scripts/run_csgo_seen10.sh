#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CONTROLAR_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
UNILIP_PYTHON="${UNILIP_PYTHON:-/home/jiahao/miniconda3/envs/UniLIP/bin/python}"
SHARED_EVAL_DIR="${SHARED_EVAL_DIR:-${PROJECT_ROOT}/../csgo_benchmark_v2_eval_general}"
EVALUATOR="${SHARED_EVAL_DIR}/run_eval.py"
DATA_ROOT="${CSGO_BENCHMARK_V2_DATA:-/home/jiahao/task/UniLIP/data/csgo_benchmark_v2}"
OUTPUT_BASE="${PROJECT_ROOT}/outputs/csgo_benchmark_v2_seen10/ControlAR"
export PYTHONDONTWRITEBYTECODE=1

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {smoke|train|infer|eval} [--seed N] [--run-root PATH] [options...]" >&2
  exit 2
fi

ACTION="$1"
shift
SEED=0
TASK="all"
CHECKPOINT=""
RUN_ROOT_OVERRIDE=""
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      [[ $# -ge 2 ]] || { echo "--seed needs a value" >&2; exit 2; }
      SEED="$2"
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    --task)
      [[ $# -ge 2 ]] || { echo "--task needs a value" >&2; exit 2; }
      TASK="$2"
      shift 2
      ;;
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "--checkpoint needs a value" >&2; exit 2; }
      CHECKPOINT="$2"
      shift 2
      ;;
    --run-root)
      [[ $# -ge 2 && -n "$2" ]] || { echo "--run-root needs a path" >&2; exit 2; }
      RUN_ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --run-root=*)
      RUN_ROOT_OVERRIDE="${1#*=}"
      [[ -n "$RUN_ROOT_OVERRIDE" ]] || { echo "--run-root needs a path" >&2; exit 2; }
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

check_model_python() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "ControlAR Python environment is missing: $PYTHON" >&2
    echo "Run scripts/setup_csgo_seen10.sh first." >&2
    return 1
  fi
}

check_evaluator() {
  if [[ ! -f "$EVALUATOR" ]]; then
    echo "Shared CSGO evaluator is missing: $EVALUATOR" >&2
    echo "Set SHARED_EVAL_DIR to a directory containing run_eval.py." >&2
    return 1
  fi
  if [[ ! -x "$UNILIP_PYTHON" ]]; then
    echo "UniLIP evaluator Python is missing: $UNILIP_PYTHON" >&2
    return 1
  fi
}

case "$ACTION" in
  eval)
    check_evaluator || exit 1
    ;;
  smoke)
    check_evaluator || exit 1
    check_model_python || exit 1
    ;;
  *)
    check_model_python || exit 1
    ;;
esac
cd "$PROJECT_ROOT"
RUN_ROOT="${OUTPUT_BASE}/seed_${SEED}"
if [[ -n "$RUN_ROOT_OVERRIDE" ]]; then
  RUN_ROOT="$RUN_ROOT_OVERRIDE"
fi
SMOKE_ROOT="${CSGO_SMOKE_ROOT:-${PROJECT_ROOT}/outputs/csgo_benchmark_v2_smoke/ControlAR/seed_${SEED}}"

run_train() {
  local run_dir="$1"
  shift
  local nproc="${NPROC_PER_NODE:-1}"
  if [[ "$nproc" == "1" ]]; then
    "$PYTHON" train_seen10.py --seed "$SEED" --data-root "$DATA_ROOT" --run-dir "$run_dir" "$@"
  else
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$nproc" \
      train_seen10.py --seed "$SEED" --data-root "$DATA_ROOT" --run-dir "$run_dir" "$@"
  fi
}

run_infer() {
  local output_root="$1"
  local smoke_flag="${2:-}"
  local infer_args=(--seed "$SEED" --data-root "$DATA_ROOT" --output-root "$output_root" --task "$TASK")
  if [[ -n "$CHECKPOINT" ]]; then
    infer_args+=(--checkpoint "$CHECKPOINT")
  fi
  if [[ -n "$smoke_flag" ]]; then
    infer_args+=(--smoke)
  fi
  infer_args+=("${FORWARD_ARGS[@]}")
  "$PYTHON" infer_seen10.py "${infer_args[@]}"
}

run_eval_task() {
  local task_name="$1"
  local smoke_mode="${2:-0}"
  local pred_root="${RUN_ROOT}/${task_name}/gen_imgs"
  local output="${RUN_ROOT}/evaluation/${task_name}"
  if [[ "$smoke_mode" == "1" ]]; then
    pred_root="${SMOKE_ROOT}/${task_name}/gen_imgs"
    "$UNILIP_PYTHON" "$EVALUATOR" smoke "$task_name" \
      --pred-root "$pred_root" --data-root "$DATA_ROOT" --limit 1
  else
    "$UNILIP_PYTHON" "$EVALUATOR" "$task_name" \
      --pred-root "$pred_root" --data-root "$DATA_ROOT" --output "$output"
  fi
}

case "$ACTION" in
  smoke)
    TASK="discrete"
    echo "Smoke outputs: ${SMOKE_ROOT}"
    run_train "$SMOKE_ROOT" --smoke
    CHECKPOINT="${SMOKE_ROOT}/checkpoints/best.pt"
    run_infer "$SMOKE_ROOT" --smoke
    run_eval_task discrete 1
    ;;
  train)
    run_train "$RUN_ROOT" "${FORWARD_ARGS[@]}"
    ;;
  infer)
    run_infer "$RUN_ROOT"
    ;;
  eval)
    if [[ "$TASK" == "all" ]]; then
      run_eval_task discrete
      run_eval_task continuous
    elif [[ "$TASK" == "discrete" || "$TASK" == "continuous" ]]; then
      run_eval_task "$TASK"
    else
      echo "Generation evaluation task must be discrete, continuous, or all" >&2
      exit 2
    fi
    ;;
  *)
    echo "Unknown action: $ACTION (expected smoke, train, infer, or eval)" >&2
    exit 2
    ;;
esac
