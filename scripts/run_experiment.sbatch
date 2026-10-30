#!/usr/bin/env bash

#SBATCH --job-name=lorehopqa-exp
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_%x_%N_%j.out
#SBATCH --error=slurm_%x_%N_%j.err

# One reader, one experiment.
#
#   sbatch scripts/run_experiment.sbatch 1 qwen
#   sbatch --gres=gpu:a100:1 scripts/run_experiment.sbatch 3 phi
#   sbatch scripts/run_experiment.sbatch 1 qwen --split validation --limit 64
#
# The first argument is the experiment (1-4 or its script name), the second is the reader handle
# from the panel YAML, and anything after that is passed to the experiment script unchanged.
# Those trailing arguments come after the ones this script supplies, so `--output-dir` there
# wins over $OUTPUT_DIR — which is how a smoke run is kept out of the real tree.
#
# WHY ONE READER PER JOB.  Loading a checkpoint is minutes of the run and the panel spans 24B to
# 70B, so a job that swept the whole panel would hold four cards for the small models at the
# speed of the largest and could not be spread across the idle nodes. One reader per job also
# means a failure costs one sweep, and `--gres=gpu:a100:2` lets two tp-2 readers share a node.
#
# RESUMING.  Predictions are appended and a re-run skips ids already recorded, so resubmitting
# after a wall-clock kill continues where it stopped. The task lists are built once and reused:
# they are what makes the per-item pairing across readers valid, and they are not re-sampled
# unless --rebuild-tasks is passed.
#
# Every path below is a parameter. MNT and IMAGE have no default and must be set for the site;
# the rest derive from the submission directory unless overridden.

set -euo pipefail

MNT=${MNT:?set MNT to the root of your shared tree}
CODE_DIR=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
OUTPUT_DIR=${OUTPUT_DIR:-$CODE_DIR/experiments}
DIST_DIR=${DIST_DIR:-$CODE_DIR/dist}
HOTPOT=${HOTPOT:-$MNT/hotpot_dev_distractor_v1.json}
CONFIG=${CONFIG:-$CODE_DIR/configs/readers.yaml}
IMAGE=${IMAGE:?set IMAGE to an Apptainer image providing vLLM}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
SPLIT=${SPLIT:-test}

HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.hf_token}
if [[ -f $HF_TOKEN_FILE ]]; then
    HF_TOKEN=$(<"$HF_TOKEN_FILE")
    export HF_TOKEN
elif [[ -z ${HF_TOKEN:-} ]]; then
    echo "[driver] no HF token in $HF_TOKEN_FILE and HF_TOKEN unset —" \
         "gated repositories will fail to download" >&2
fi

# ----------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <1|2|3|5> <reader> [extra args passed to the experiment script]" >&2
    exit 2
fi

case $1 in
    1 | e1 | closed_book)      SCRIPT=experiment_1_closed_book.py ;;
    2 | e2 | ablated_evidence) SCRIPT=experiment_2_ablated_evidence.py ;;
    3 | e3 | full_context)     SCRIPT=experiment_3_full_context.py ;;
    5 | e5 | bm25 | retrieval)  SCRIPT=experiment_5_bm25_retrieval.py ;;
    *)  echo "Unknown experiment: $1 (expected 1, 2, 3 or 5)" >&2; exit 2 ;;
esac
READER=$2
shift 2

# A --split among the trailing arguments is the run's split, not $SPLIT's default. It is pulled
# out here rather than only forwarded, because the preflight checks that the split's data files
# exist and would otherwise be checking the wrong pair — passing a job whose inputs are missing
# straight into the queue, and reporting the wrong split in the header while it ran.
REST=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --split) SPLIT=$2; shift 2 ;;
        --split=*) SPLIT=${1#*=}; shift ;;
        *) REST+=("$1"); shift ;;
    esac
done
set -- ${REST[@]+"${REST[@]}"}


# ----------------------------------------------------------------------------------
# Container invocation
# ----------------------------------------------------------------------------------

APPTAINER=${APPTAINER:-apptainer}
BIND_ARGS=(--bind "$MNT:$MNT")
ENV_ARGS=(
    --env "PYTHONPATH=$CODE_DIR/src"
    --env "HF_HOME=$HF_CACHE"
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}"
)

run_py() {
    "$APPTAINER" exec --nv "${BIND_ARGS[@]}" "${ENV_ARGS[@]}" "$IMAGE" python "$@"
}
query_py() {
    "$APPTAINER" exec "${BIND_ARGS[@]}" "${ENV_ARGS[@]}" "$IMAGE" python -c "$1"
}

# ----------------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------------

for path in "$IMAGE" "$CONFIG" "$CODE_DIR/scripts/$SCRIPT" \
            "$DIST_DIR/lorehopqa/$SPLIT.jsonl" \
            "$DIST_DIR/lorehopqa-private/$SPLIT.provenance.jsonl"; do
    [[ -e $path ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ -e $HOTPOT ]] || echo "[driver] warning: no HotpotQA at $HOTPOT — paired conditions will be skipped" >&2
mkdir -p "$OUTPUT_DIR" "$HF_CACHE"

echo "=== LoreHopQA experiment run ==="
echo "job        : ${SLURM_JOB_ID:-local} on $(hostname)"
echo "experiment : $SCRIPT"
echo "reader     : $READER"
echo "config     : $CONFIG"
echo "split      : $SPLIT"
echo "dist       : $DIST_DIR"
echo "output     : $OUTPUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader || true

# The hub is offline (every reader is in the shared cache; a missing one fails in seconds), an
# occupied GPU refused, and every compile cache on node-local disk -- see gpu_preflight.sh.
ENV_ARGS+=(--env "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}")
source "$CODE_DIR/scripts/gpu_preflight.sh"
gpu_preflight
setup_local_caches

# A reader whose tensor_parallel_size exceeds the allocation hangs in NCCL rather than failing
# cleanly, and would do so only after the queue wait. Catch it before the model is touched.
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
query_py "
import sys
from pathlib import Path
from lorehopqa.experiments import load_readers_config
reader = load_readers_config(Path('$CONFIG')).get('$READER')
if reader.tensor_parallel_size > $GPU_COUNT:
    sys.exit(
        f'{reader.name}: tensor_parallel_size={reader.tensor_parallel_size} > $GPU_COUNT allocated GPU(s). '
        'Raise --gres=gpu:a100:N or lower it in the panel.'
    )
print(f'[preflight] {reader.name}: {reader.model} tp={reader.tensor_parallel_size} pool={reader.pool}')
"

# ----------------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------------

set +e
run_py "$CODE_DIR/scripts/$SCRIPT" \
    --config "$CONFIG" \
    --model "$READER" \
    --dist-dir "$DIST_DIR" \
    --hotpot "$HOTPOT" \
    --output-dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --no-progress \
    "$@"
STATUS=$?
set -e

echo "=== done: $SCRIPT / $READER (exit $STATUS) ==="
echo "predictions: $OUTPUT_DIR/*/predictions/*.$READER.jsonl"
exit "$STATUS"
