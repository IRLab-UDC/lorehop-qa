#!/usr/bin/env bash
#SBATCH --job-name=lorehopqa-py
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%x_%N_%j.out
#SBATCH --error=slurm_%x_%N_%j.err

# Run any project script inside the container on a CPU node.
#
#   sbatch -J frag scripts/run_python.sbatch scripts/tokenizer_fragmentation.py --flag value
#
# No --nv and no GPU request: analysis stages that only need the container's Python start
# sooner without one. Slurm spools the batch script, so the project root is CODE_DIR.

set -euo pipefail
MNT=${MNT:?set MNT to the root of your shared tree}
CODE_DIR=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
IMAGE=${IMAGE:?set IMAGE to an Apptainer image providing vLLM}

cd "$CODE_DIR"
echo "=== $* on $(hostname) ==="
apptainer exec --bind "$MNT:$MNT" \
    --env "PYTHONPATH=$CODE_DIR/src" --env "HF_HOME=$MNT/hf-cache" \
    "$IMAGE" python "$@"
echo "=== done ==="
