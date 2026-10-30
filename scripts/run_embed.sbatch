#!/usr/bin/env bash
#SBATCH --job-name=lorehopqa-embed
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%x_%N_%j.out
#SBATCH --error=slurm_%x_%N_%j.err

# Embed the retrieval corpora and queries for the dense and hybrid runs.
#
#   sbatch scripts/run_embed.sbatch --model sentence-transformers/all-MiniLM-L6-v2 \
#       --tag minilm --no-prompts --max-model-len 256
#
# One GPU is enough: the embedder is 596M parameters and the whole job is a single pass over
# 135k documents and 20k queries. It writes float32 vectors into the shared tree, which a
# later stage indexes into Elasticsearch; that stage needs no GPU and runs through
# scripts/run_es.sbatch.

set -euo pipefail

MNT=${MNT:?set MNT to the root of your shared tree}
CODE_DIR=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
OUTPUT_DIR=${OUTPUT_DIR:-$CODE_DIR/experiments}
DIST_DIR=${DIST_DIR:-$CODE_DIR/dist}
HOTPOT=${HOTPOT:-$MNT/hotpot_dev_distractor_v1.json}
IMAGE=${IMAGE:?set IMAGE to an Apptainer image providing vLLM}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
SPLIT=${SPLIT:-test}

echo "=== embedding run ==="
echo "job    : ${SLURM_JOB_ID:-local} on $(hostname)"
echo "split  : $SPLIT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

ENV_ARGS=(
    --env "PYTHONPATH=$CODE_DIR/src"
    --env "HF_HOME=$HF_CACHE"
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}"
)
# Embedders are small and may pull auxiliary code repositories, so the hub stays online here.
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

# The hub is offline (every reader is in the shared cache; a missing one fails in seconds), an
# occupied GPU refused, and every compile cache on node-local disk -- see gpu_preflight.sh.
ENV_ARGS+=(--env "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}")
source "$CODE_DIR/scripts/gpu_preflight.sh"
gpu_preflight
setup_local_caches

apptainer exec --nv --bind "$MNT:$MNT" "${ENV_ARGS[@]}" \
    "$IMAGE" python "$CODE_DIR/scripts/embed_corpus.py" \
        --dist-dir "$DIST_DIR" \
        --hotpot "$HOTPOT" \
        --output-dir "$OUTPUT_DIR" \
        --split "$SPLIT" "$@"
echo "=== done ==="
