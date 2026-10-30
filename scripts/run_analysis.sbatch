#!/usr/bin/env bash
#SBATCH --job-name=lorehopqa-analysis
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%x_%N_%j.out
#SBATCH --error=slurm_%x_%N_%j.err

# Every analysis the paper's numbers come from, then the paper's tables, for one split.
#
#   sbatch scripts/run_analysis.sbatch [split]       # default test
#
# Queued last, with afterany on every reader job: a reader
# that failed leaves a hole the analysis reports, rather than holding back everything else. Each
# step runs even if an earlier one failed, and the job exits non-zero if any did.

set -uo pipefail

SPLIT=${1:-${SPLIT:-test}}
MNT=${MNT:?set MNT to the root of your shared tree}
CODE_DIR=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
IMAGE=${IMAGE:?set IMAGE to an Apptainer image providing vLLM}
HOTPOT=${HOTPOT:-$MNT/hotpot_dev_distractor_v1.json}
EXP=$CODE_DIR/experiments
DIST=${DIST_DIR:-$CODE_DIR/dist}
RESULTS=$EXP/results/$SPLIT
E5=$EXP/e5_bm25_retrieval/$SPLIT

cd "$CODE_DIR"
mkdir -p "$RESULTS"
echo "=== LoreHopQA analysis ($SPLIT) on $(hostname), job ${SLURM_JOB_ID:-local} ==="

failed=()
step() {  # step <name> <command...>
    local name=$1
    shift
    echo
    echo "--- $name"
    "$@" || { echo "[analysis] $name FAILED"; failed+=("$name"); }
}
py() {
    apptainer exec --bind "$MNT:$MNT" --env "PYTHONPATH=$CODE_DIR/src" --env "HF_HOME=$MNT/hf-cache" \
        "$IMAGE" python "$@"
}

step summary          py scripts/analyze_experiments.py --output-dir "$EXP" --split "$SPLIT"
step ablation_paired  python3 scripts/ablation_paired.py --experiments "$EXP" --split "$SPLIT" --output "$EXP/ablation_paired.json"
step ablation_table   python3 scripts/ablation_table.py --experiments "$EXP" --split "$SPLIT" --output "$RESULTS/ablation_table.json"
step no_support       python3 scripts/no_support_paired.py --experiments "$EXP" --split "$SPLIT" --output "$EXP/no_support_paired.json"
step scale_knowledge  python3 scripts/scale_knowledge.py --experiments "$EXP" --config configs/readers.yaml --split "$SPLIT" --output "$EXP/scale_knowledge.json"
step unanswered       python3 scripts/unanswered_agreement.py --experiments "$EXP" --split "$SPLIT" --output "$RESULTS/unanswered_agreement.json"
step rankings         python3 scripts/summarise_rankings.py --root "$E5"
step adaptation       python3 scripts/adaptation_ranking.py --root "$E5" --adapters "$CODE_DIR/adapters" --output "$RESULTS/adaptation_ranking.json"
step transfer_slope   python3 scripts/transfer_slope.py --root "$E5" --report "$RESULTS/transfer_slope.json"
step corpus_stats     python3 scripts/corpus_statistics.py --dist-dir "$DIST" --hotpot "$HOTPOT" --split "$SPLIT" --lorepedia "$DIST/lorepedia/lorepedia.jsonl" --output "$EXP/corpus_statistics.json"
step fragmentation    py scripts/tokenizer_fragmentation.py --dist-dir "$DIST" --hotpot "$HOTPOT" --lorepedia "$DIST/lorepedia/lorepedia.jsonl" --output "$EXP/tokenizer_fragmentation.json"
step paper_tables     python3 scripts/paper_tables.py --experiments "$EXP" --dist "$DIST" --split "$SPLIT" --all --tex-dir "$RESULTS/tables"

echo
if (( ${#failed[@]} )); then
    echo "=== analysis finished with failures: ${failed[*]}"
    exit 1
fi
echo "=== analysis complete: $RESULTS (tables in $RESULTS/tables) ==="
