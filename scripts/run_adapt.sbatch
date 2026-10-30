#!/usr/bin/env bash
#SBATCH --job-name=lorehopqa-adapt
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%x_%N_%j.out
#SBATCH --error=slurm_%x_%N_%j.err

# Adapt an embedder to the worlds' vocabulary on Lorepedia text alone.
#
#   sbatch -J adapt-minilm scripts/run_adapt.sbatch \
#       sentence-transformers/all-MiniLM-L6-v2 mean minilm-lp2-1ep --objective ict --freeze-encoder ...
#
# Trains the three adapters the paper reports (jina-lp2, minilm-lp2-1ep, simcse-lp2) on the
# Lorepedia, with the recipe each adapter records
# in its adaptation.json.
#
# Pooling must match what vLLM applies when the corpus is indexed, which it takes from the
# model config: MEAN for all-MiniLM-L6-v2, CLS for unsup-SimCSE, last-token for Jina v5.

set -euo pipefail
MNT=${MNT:?set MNT to the root of your shared tree}
CODE_DIR=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
IMAGE=${IMAGE:?set IMAGE to an Apptainer image providing vLLM}

MODEL=${1:?usage: $0 <model> <pooling> <tag> [extra args]}
POOLING=${2:?}
TAG=${3:?}
shift 3

cd "$CODE_DIR"
echo "=== adapting $MODEL ($POOLING) -> adapters/$TAG on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
ENV_ARGS=(--env "PYTHONPATH=$CODE_DIR/src" --env "HF_HOME=$MNT/hf-cache")
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

# The hub is offline (every reader is in the shared cache; a missing one fails in seconds), an
# occupied GPU refused, and every compile cache on node-local disk -- see gpu_preflight.sh.
ENV_ARGS+=(--env "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}")
source "$CODE_DIR/scripts/gpu_preflight.sh"
gpu_preflight
setup_local_caches

apptainer exec --nv --bind "$MNT:$MNT" "${ENV_ARGS[@]}" \
    "$IMAGE" python "$CODE_DIR/scripts/adapt_embedder.py" \
        --model "$MODEL" --pooling "$POOLING" \
        --lorepedia "${LOREPEDIA:-$CODE_DIR/dist/lorepedia/lorepedia.jsonl}" \
        --output "$CODE_DIR/adapters/$TAG" "$@"
echo "=== done ==="
