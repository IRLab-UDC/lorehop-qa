#!/usr/bin/env bash
#SBATCH --job-name=lorehopqa-dense
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_%x_%N_%j.out
#SBATCH --error=slurm_%x_%N_%j.err

# Build the dense and hybrid rankings for the release, one embedder at a time.
#
#   sbatch scripts/run_dense.sbatch
#   sbatch scripts/run_dense.sbatch simcse minilm-adapted2      # a subset
#
# WHY A COMPUTE NODE. build_dense_tasks.py was written to run on the access node, because that
# is the only host with a route to Elasticsearch on localhost. The access node is also a shared
# login box with 7GB of RAM, and this job holds the document vectors, the query vectors and the
# whole corpus text at once -- so it competes with whoever else is logged in and is killed when
# they win. socat on the login node publishes Elasticsearch at the address in $ES, which the compute
# nodes can reach, so the memory constraint disappears and the run stops depending on how busy
# the login node happens to be.
#
# Elasticsearch itself is still shared. Its search thread pool rejects batches under load, which
# build_dense_tasks now retries; this script adds a second, coarser retry around the whole
# embedder for the case where the server is busy for longer than that.
#
# No GPU: the vectors were written by embed_corpus.py already, and this only searches with them.

set -uo pipefail

MNT=${MNT:?set MNT to the root of your shared tree}
CODE_DIR=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
IMAGE=${IMAGE:?set IMAGE to an Apptainer image providing vLLM}
DIST_DIR=${DIST_DIR:-$CODE_DIR/dist}
OUTPUT_DIR=${OUTPUT_DIR:-$CODE_DIR/experiments}
HOTPOT=${HOTPOT:-$MNT/hotpot_dev_distractor_v1.json}
SPLIT=${SPLIT:-test}
ES=${ES:?set ES to the Elasticsearch endpoint the compute nodes can reach, e.g. http://10.0.0.1:9200}
INDEX_PREFIX=${INDEX_PREFIX:-lorehopqa}
ATTEMPTS=${ATTEMPTS:-4}

TAGS=("$@")
if [[ ${#TAGS[@]} -eq 0 ]]; then
    # The paper's embedders: the three base models and their Lorepedia adaptations (Table 8).
    TAGS=(jina minilm simcse jina-adapted2 minilm-adapted2 simcse-adapted2)
fi

echo "=== dense ($SPLIT) on $(hostname) ==="
echo "es      : $ES"
echo "dist    : $DIST_DIR"
echo "output  : $OUTPUT_DIR"
echo "tags    : ${TAGS[*]}"
# Elasticsearch is reached through a tunnel on the login node that has to be started by hand,
# and this job may be queued days before it runs. Wait for it (ES_WAIT_HOURS, default 24) instead
# of failing on arrival and taking every dependent job down with it.
es_wait() {
    local hours=${ES_WAIT_HOURS:-24} waited=0
    until curl -s -m 15 -o /dev/null "$ES"; do
        if (( waited >= hours * 3600 )); then
            echo "[fatal] Elasticsearch unreachable at $ES from $(hostname) after ${hours}h" >&2
            exit 1
        fi
        (( waited == 0 )) && echo "[es] $ES unreachable; waiting for the tunnel (up to ${hours}h)"
        sleep 300
        waited=$((waited + 300))
    done
    (( waited > 0 )) && echo "[es] reachable after $((waited / 60)) min"
    return 0
}
es_wait

REPORTS=$OUTPUT_DIR/e5_bm25_retrieval/$SPLIT
failed=()
for tag in "${TAGS[@]}"; do
    # An embedder that already has its report is finished; re-running would redo an hour of
    # search to write the same numbers.
    if [[ -f $REPORTS/retrieval-dense-$tag.json ]]; then
        echo "=== $tag: report exists, skipping"
        continue
    fi
    if [[ $tag == jina ]]; then field=vec; else field="vec_${tag//-/_}"; fi
    echo
    echo "=============== $tag (field $field) ==============="

    ok=0
    for attempt in $(seq 1 "$ATTEMPTS"); do
        apptainer exec --bind "$MNT:$MNT" --env "PYTHONPATH=$CODE_DIR/src" \
            "$IMAGE" python "$CODE_DIR/scripts/build_dense_tasks.py" \
                --dist-dir "$DIST_DIR" --hotpot "$HOTPOT" \
                --output-dir "$OUTPUT_DIR" --split "$SPLIT" \
                --es "$ES" --index-prefix "$INDEX_PREFIX" \
                --tag "$tag" --field "$field" --top-k 10
        if [[ -f $REPORTS/retrieval-dense-$tag.json ]]; then ok=1; break; fi
        echo "[retry] $tag: attempt $attempt produced no report; pausing before the next"
        sleep $((60 * attempt))
    done
    [[ $ok == 1 ]] || failed+=("$tag")
done

echo
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "=== incomplete: ${failed[*]}"
    exit 1
fi
echo "=== all requested embedders reported ==="
