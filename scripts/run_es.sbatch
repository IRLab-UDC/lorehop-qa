#!/usr/bin/env bash
#SBATCH --job-name=lorehopqa-es
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm_%x_%N_%j.out
#SBATCH --error=slurm_%x_%N_%j.err

# Run an Elasticsearch-side stage (indexing, kNN, reranking) on a compute node.
#
#   sbatch scripts/run_es.sbatch scripts/build_es_tasks.py --dist-dir dist ...
#
# The Valinor tunnel is bound to the access node's own address rather than to loopback, so
# compute nodes reach it at the address in $ES. Running these stages under Slurm lets several
# embedders or collections proceed at once instead of queueing behind one login-node process.
#
# Slurm copies the batch script to a spool directory before running it, so the project root
# comes from CODE_DIR rather than from the script's own path.

set -euo pipefail

MNT=${MNT:?set MNT to the root of your shared tree}
CODE_DIR=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
ES=${ES:?set ES to the Elasticsearch endpoint the compute nodes can reach, e.g. http://10.0.0.1:9200}

cd "$CODE_DIR"
echo "=== elasticsearch stage ==="
echo "job  : ${SLURM_JOB_ID:-local} on $(hostname)"
echo "es   : $ES"
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

exec python3 "$@" --es "$ES"
