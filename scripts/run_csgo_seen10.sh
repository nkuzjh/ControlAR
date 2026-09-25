#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Resolve all host paths after CLI parsing with the stdlib-only resolver.
DATA_ROOT_OVERRIDE=""
EVAL_ROOT_OVERRIDE=""
EVAL_PYTHON_OVERRIDE=""
PRINT_PATHS=0
OUTPUT_BASE="${PROJECT_ROOT}/outputs/csgo_benchmark_v2_seen10/ControlAR"
ALIGNED_EXPERIMENT="csgo_seen10_exp32gen_aligned"
PEFT_EXPERIMENT="csgo_seen10_exp32gen_aligned_peft"
ALIGNED_CONFIG="${PROJECT_ROOT}/configs/${ALIGNED_EXPERIMENT}.json"
ALIGNED_OUTPUT_BASE="${PROJECT_ROOT}/outputs/${ALIGNED_EXPERIMENT}/ControlAR"
TRAIN_ENTRY="train_seen10.py"
VALIDATOR="scripts/validate_csgo_seen10_aligned.py"
export PYTHONDONTWRITEBYTECODE=1

if [[ $# -lt 1 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 {smoke|train|infer|eval} [--experiment csgo_seen10_exp32gen_aligned|csgo_seen10_exp32gen_aligned_peft] [--seed N] [--run-root PATH] [--checkpoint-role late|best] [--inference-seed N] [--data-root PATH] [--eval-root PATH] [--eval-python PATH] [--print-paths] [options...]" >&2
  [[ $# -gt 0 ]] && exit 0
  exit 2
fi

ACTION="$1"
shift
SEED=0
SEED_SET=0
TASK="all"
CHECKPOINT=""
RUN_ROOT_OVERRIDE=""
EXPERIMENT=""
CHECKPOINT_ROLE=""
INFERENCE_SEED=""
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-paths)
      PRINT_PATHS=1
      shift
      ;;
    --data-root|--eval-root|--eval-python|--unilip-python)
      [[ $# -ge 2 && -n "$2" ]] || { echo "$1 needs a path" >&2; exit 2; }
      case "$1" in
        --data-root) DATA_ROOT_OVERRIDE="$2" ;;
        --eval-root) EVAL_ROOT_OVERRIDE="$2" ;;
        *) EVAL_PYTHON_OVERRIDE="$2" ;;
      esac
      shift 2
      ;;
    --data-root=*|--eval-root=*|--eval-python=*|--unilip-python=*)
      path_value="${1#*=}"
      [[ -n "$path_value" ]] || { echo "$1 needs a path" >&2; exit 2; }
      case "$1" in
        --data-root=*) DATA_ROOT_OVERRIDE="$path_value" ;;
        --eval-root=*) EVAL_ROOT_OVERRIDE="$path_value" ;;
        *) EVAL_PYTHON_OVERRIDE="$path_value" ;;
      esac
      shift
      ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "--seed needs a value" >&2; exit 2; }
      SEED="$2"
      SEED_SET=1
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      SEED_SET=1
      shift
      ;;
    --task)
      [[ $# -ge 2 ]] || { echo "--task needs a value" >&2; exit 2; }
      TASK="$2"
      shift 2
      ;;
    --task=*)
      TASK="${1#*=}"
      shift
      ;;
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "--checkpoint needs a value" >&2; exit 2; }
      CHECKPOINT="$2"
      shift 2
      ;;
    --checkpoint=*)
      CHECKPOINT="${1#*=}"
      shift
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
    --experiment)
      [[ $# -ge 2 ]] || { echo "--experiment needs a value" >&2; exit 2; }
      EXPERIMENT="$2"
      shift 2
      ;;
    --experiment=*)
      EXPERIMENT="${1#*=}"
      shift
      ;;
    --checkpoint-role)
      [[ $# -ge 2 ]] || { echo "--checkpoint-role needs a value" >&2; exit 2; }
      CHECKPOINT_ROLE="$2"
      shift 2
      ;;
    --checkpoint-role=*)
      CHECKPOINT_ROLE="${1#*=}"
      shift
      ;;
    --inference-seed)
      [[ $# -ge 2 ]] || { echo "--inference-seed needs a value" >&2; exit 2; }
      INFERENCE_SEED="$2"
      shift 2
      ;;
    --inference-seed=*)
      INFERENCE_SEED="${1#*=}"
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$EXPERIMENT" == "$PEFT_EXPERIMENT" ]]; then
  ALIGNED_EXPERIMENT="$PEFT_EXPERIMENT"
  ALIGNED_CONFIG="${PROJECT_ROOT}/configs/${ALIGNED_EXPERIMENT}.json"
  ALIGNED_OUTPUT_BASE="${PROJECT_ROOT}/outputs/${ALIGNED_EXPERIMENT}/ControlAR"
  TRAIN_ENTRY="train_seen10_peft.py"
  VALIDATOR="scripts/validate_csgo_seen10_peft.py"
fi
if [[ -n "$EXPERIMENT" && "$EXPERIMENT" != "$ALIGNED_EXPERIMENT" ]]; then
  echo "Unsupported experiment: $EXPERIMENT" >&2
  exit 2
fi
ALIGNED=0
if [[ "$EXPERIMENT" == "$ALIGNED_EXPERIMENT" ]]; then
  ALIGNED=1
  if [[ "$SEED_SET" == "0" ]]; then
    SEED=42
  fi
  [[ -n "$CHECKPOINT_ROLE" ]] || CHECKPOINT_ROLE="late"
  [[ "$CHECKPOINT_ROLE" == "late" || "$CHECKPOINT_ROLE" == "best" ]] || {
    echo "Aligned --checkpoint-role must be late or best" >&2
    exit 2
  }
  [[ -n "$INFERENCE_SEED" ]] || INFERENCE_SEED="$SEED"
  [[ "$SEED" == "42" ]] || {
    echo "Aligned experiment fixes training seed=42" >&2
    exit 2
  }
  [[ "$INFERENCE_SEED" == "42" ]] || {
    echo "Aligned experiment fixes inference seed=42" >&2
    exit 2
  }
elif [[ -n "$CHECKPOINT_ROLE" || -n "$INFERENCE_SEED" ]]; then
  echo "--checkpoint-role and --inference-seed require --experiment $ALIGNED_EXPERIMENT" >&2
  exit 2
fi

case "$ACTION" in
  smoke|train|infer|eval) ;;
  *) echo "Unknown action: $ACTION (expected smoke, train, infer, or eval)" >&2; exit 2 ;;
esac
# The resolver must work before installing torch or a model environment.
PATHS_PYTHON="${CONTROLAR_PATHS_PYTHON:-}"
if [[ -z "$PATHS_PYTHON" ]]; then
  PATHS_PYTHON="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PATHS_PYTHON" ]]; then
  PATHS_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi
PATH_ARGS=(--action "$ACTION" --experiment "$EXPERIMENT" --seed "$SEED")
[[ -z "$DATA_ROOT_OVERRIDE" ]] || PATH_ARGS+=(--data-root "$DATA_ROOT_OVERRIDE")
[[ -z "$EVAL_ROOT_OVERRIDE" ]] || PATH_ARGS+=(--eval-root "$EVAL_ROOT_OVERRIDE")
[[ -z "$EVAL_PYTHON_OVERRIDE" ]] || PATH_ARGS+=(--eval-python "$EVAL_PYTHON_OVERRIDE")
[[ -z "$RUN_ROOT_OVERRIDE" ]] || PATH_ARGS+=(--run-root "$RUN_ROOT_OVERRIDE")
if [[ "$PRINT_PATHS" == "1" ]]; then
  exec "$PATHS_PYTHON" "$PROJECT_ROOT/scripts/csgo_runtime_paths.py" "${PATH_ARGS[@]}"
fi
PATH_VALUES="$("$PATHS_PYTHON" "$PROJECT_ROOT/scripts/csgo_runtime_paths.py" "${PATH_ARGS[@]}" --lines)"
mapfile -t RESOLVED_PATHS <<< "$PATH_VALUES"
PYTHON="${RESOLVED_PATHS[0]}"
DATA_ROOT="${RESOLVED_PATHS[1]}"
SHARED_EVAL_DIR="${RESOLVED_PATHS[2]}"
EVAL_RUNTIME_PYTHON="${RESOLVED_PATHS[3]}"
RESOLVED_RUN_ROOT="${RESOLVED_PATHS[4]}"
EVALUATOR="${SHARED_EVAL_DIR}/run_eval.py"
EVAL_CONFIG="${SHARED_EVAL_DIR}/benchmark_v2.yaml"

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
  if [[ ! -f "$EVAL_CONFIG" ]]; then
    echo "Shared evaluator config is missing: $EVAL_CONFIG" >&2
    return 1
  fi
  if [[ ! -f "$EVAL_RUNTIME_PYTHON" || ! -x "$EVAL_RUNTIME_PYTHON" ]]; then
    echo "Evaluator Python is missing: $EVAL_RUNTIME_PYTHON; prepare it in the shared evaluator .venv or set EVAL_PYTHON." >&2
    return 1
  fi
}

cd "$PROJECT_ROOT"
if [[ "$ALIGNED" == "1" ]]; then
  OUTPUT_BASE="$ALIGNED_OUTPUT_BASE"
fi
RUN_ROOT="$RESOLVED_RUN_ROOT"
DEFAULT_RUN_ROOT="${OUTPUT_BASE}/seed_${SEED}"
if [[ "$ALIGNED" == "1" && "$ACTION" != "smoke" && -n "$RUN_ROOT_OVERRIDE" ]]; then
  if [[ "$(realpath -m "$RUN_ROOT")" != "$(realpath -m "$DEFAULT_RUN_ROOT")" ]]; then
    echo "Formal aligned runs require the default run root: $DEFAULT_RUN_ROOT" >&2
    exit 2
  fi
fi
if [[ "$ALIGNED" == "1" ]]; then
  SMOKE_ROOT="${CSGO_ALIGNED_SMOKE_ROOT:-${PROJECT_ROOT}/outputs/${ALIGNED_EXPERIMENT}_smoke/ControlAR/seed_${SEED}}"
else
  SMOKE_ROOT="${CSGO_SMOKE_ROOT:-${PROJECT_ROOT}/outputs/csgo_benchmark_v2_smoke/ControlAR/seed_${SEED}}"
fi
ACTIVE_RUN_ROOT="$RUN_ROOT"

run_train() {
  local run_dir="$1"
  shift
  local nproc="${NPROC_PER_NODE:-1}"
  local train_args=(--seed "$SEED" --data-root "$DATA_ROOT" --run-dir "$run_dir")
  train_args+=("$@")
  if [[ "$ALIGNED" == "1" ]]; then
    # Put the profile selectors after forwarded options so formal aligned
    # invocations cannot silently switch to a different config or run root.
    train_args+=(
      --config "$ALIGNED_CONFIG"
      --experiment "$ALIGNED_EXPERIMENT"
      --seed "$SEED"
      --data-root "$DATA_ROOT"
      --run-dir "$run_dir"
    )
  fi
  if [[ "$nproc" == "1" ]]; then
    "$PYTHON" "$TRAIN_ENTRY" "${train_args[@]}"
  else
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$nproc" \
      "$TRAIN_ENTRY" "${train_args[@]}"
  fi
}

run_infer() {
  local output_root="$1"
  local smoke_flag="${2:-}"
  local infer_args
  if [[ "$ALIGNED" == "1" ]]; then
    local selected_checkpoint="$CHECKPOINT"
    if [[ -z "$selected_checkpoint" ]]; then
      selected_checkpoint="${ACTIVE_RUN_ROOT}/checkpoints/${CHECKPOINT_ROLE}.pt"
    fi
    infer_args=()
  else
    infer_args=(--seed "$SEED" --data-root "$DATA_ROOT" --output-root "$output_root" --task "$TASK")
    if [[ -n "$CHECKPOINT" ]]; then
      infer_args+=(--checkpoint "$CHECKPOINT")
    fi
  fi
  if [[ -n "$smoke_flag" ]]; then
    infer_args+=(--smoke)
  fi
  infer_args+=("${FORWARD_ARGS[@]}")
  if [[ "$ALIGNED" == "1" ]]; then
    # Reassert all identity and fixed-shape fields after forwarded options.
    infer_args+=(
      --config "$ALIGNED_CONFIG"
      --experiment "$ALIGNED_EXPERIMENT"
      --checkpoint-role "$CHECKPOINT_ROLE"
      --checkpoint "$selected_checkpoint"
      --inference-seed "$INFERENCE_SEED"
      --seed "$SEED"
      --data-root "$DATA_ROOT"
      --output-base "$ALIGNED_OUTPUT_BASE"
      --output-root "$output_root"
      --task "$TASK"
      --inference-engine compiled
      --batch-size 16
    )
  fi
  "$PYTHON" infer_seen10.py "${infer_args[@]}"
}

aligned_eval_preflight() {
  local validate_args=(
    "$VALIDATOR"
    --config "$ALIGNED_CONFIG"
    --data-root "$DATA_ROOT"
    --eval-config "$EVAL_CONFIG"
    --run-root "$ACTIVE_RUN_ROOT"
    --checkpoint-role "$CHECKPOINT_ROLE"
    --inference-seed "$INFERENCE_SEED"
  )
  "$PYTHON" "${validate_args[@]}" >/dev/null
  local selected_checkpoint="${ACTIVE_RUN_ROOT}/checkpoints/${CHECKPOINT_ROLE}.pt"
  local tasks=(discrete continuous)
  if [[ "$TASK" != "all" ]]; then
    tasks=("$TASK")
  fi
  local task_name
  for task_name in "${tasks[@]}"; do
    local pred_root="${ACTIVE_RUN_ROOT}/predictions/${CHECKPOINT_ROLE}/inference_seed_${INFERENCE_SEED}"
    "$PYTHON" - "$pred_root" "$task_name" "$selected_checkpoint" "$CHECKPOINT_ROLE" "$ALIGNED_CONFIG" "$DATA_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[5]).resolve()
if json.loads(config_path.read_text())["experiment"] == "csgo_seen10_exp32gen_aligned_peft":
    from csgo_seen10.peft_artifact_contract import preflight_evaluation
else:
    from csgo_seen10.artifact_contract import preflight_evaluation
digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
preflight_evaluation(
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], digest, sys.argv[6]
)
PY
  done
}

run_eval_task() {
  local task_name="$1"
  local smoke_mode="${2:-0}"
  local pred_root
  local output
  if [[ "$ALIGNED" == "1" ]]; then
    pred_root="${ACTIVE_RUN_ROOT}/predictions/${CHECKPOINT_ROLE}/inference_seed_${INFERENCE_SEED}/${task_name}/gen_imgs"
    output="${ACTIVE_RUN_ROOT}/evaluation/${CHECKPOINT_ROLE}/inference_seed_${INFERENCE_SEED}/${task_name}"
  else
    pred_root="${ACTIVE_RUN_ROOT}/${task_name}/gen_imgs"
    output="${ACTIVE_RUN_ROOT}/evaluation/${task_name}"
  fi
  if [[ "$smoke_mode" == "1" ]]; then
    "$EVAL_RUNTIME_PYTHON" "$EVALUATOR" smoke "$task_name" \
      --pred-root "$pred_root" --data-root "$DATA_ROOT" --config "$EVAL_CONFIG" --limit 1
  else
    "$EVAL_RUNTIME_PYTHON" "$EVALUATOR" "$task_name" \
      --pred-root "$pred_root" --data-root "$DATA_ROOT" --config "$EVAL_CONFIG" --output "$output"
  fi
}

case "$ACTION" in
  eval)
    check_evaluator || exit 1
    if [[ "$ALIGNED" == "1" ]]; then
      check_model_python || exit 1
    fi
    ;;
  smoke)
    check_evaluator || exit 1
    check_model_python || exit 1
    ;;
  *)
    check_model_python || exit 1
    ;;
esac

case "$ACTION" in
  smoke)
    TASK="discrete"
    ACTIVE_RUN_ROOT="$SMOKE_ROOT"
    echo "Smoke outputs: ${SMOKE_ROOT}"
    run_train "$SMOKE_ROOT" --smoke
    if [[ "$ALIGNED" == "1" ]]; then
      if [[ ! -e "${SMOKE_ROOT}/checkpoints/${CHECKPOINT_ROLE}.pt" ]]; then
        ln -f "${SMOKE_ROOT}/checkpoints/step_000001.pt" "${SMOKE_ROOT}/checkpoints/${CHECKPOINT_ROLE}.pt"
      fi
      CHECKPOINT="${SMOKE_ROOT}/checkpoints/${CHECKPOINT_ROLE}.pt"
      run_infer "${SMOKE_ROOT}/predictions/${CHECKPOINT_ROLE}/inference_seed_${INFERENCE_SEED}" --smoke
    else
      CHECKPOINT="${SMOKE_ROOT}/checkpoints/best.pt"
      run_infer "$SMOKE_ROOT" --smoke
    fi
    run_eval_task discrete 1
    ;;
  train)
    run_train "$RUN_ROOT" "${FORWARD_ARGS[@]}"
    ;;
  infer)
    if [[ "$ALIGNED" == "1" && -n "$CHECKPOINT" ]]; then
      expected_checkpoint="${RUN_ROOT}/checkpoints/${CHECKPOINT_ROLE}.pt"
      if [[ ! -e "$CHECKPOINT" || ! -e "$expected_checkpoint" || ! "$CHECKPOINT" -ef "$expected_checkpoint" ]]; then
        echo "Aligned --checkpoint must be the selected role alias: $expected_checkpoint" >&2
        exit 2
      fi
    fi
    if [[ "$ALIGNED" == "1" ]]; then
      run_infer "${RUN_ROOT}/predictions/${CHECKPOINT_ROLE}/inference_seed_${INFERENCE_SEED}"
    else
      run_infer "$RUN_ROOT"
    fi
    ;;
  eval)
    ACTIVE_RUN_ROOT="$RUN_ROOT"
    if [[ "$ALIGNED" == "1" ]]; then
      aligned_eval_preflight
    fi
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
