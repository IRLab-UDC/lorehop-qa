# Sourced by the GPU drivers.
#
#   gpu_preflight        fail the job in seconds when an allocated GPU is already holding memory
#   wait_for_free_gpus   between phases: wait until the previous model's processes have let go
#   run_phase CMD...     run one phase, and once more after the GPUs are clear if it fails
#
# Why each exists:
#
# * Probe job 11745 landed on b15, whose GPU 2 held 13.4 GB for a defunct process of another
#   user, six days old, with no Slurm job behind it. Only an administrator can release that, so
#   the job stops and names the node:   sbatch --exclude=<node> scripts/...
#
# * Two jobs died loading one model right after another. vLLM's
#   shutdown had force-killed Gemma's engine with its two GPU workers "still running after grace
#   period", and the next phase started 19 seconds later. Mistral's weight load peaks at the
#   card's full 80 GB, so whatever the old workers still held was enough to fail it (7,504
#   allocator OOM retries, then "CUDA error: invalid argument"). On empty cards the same load
#   passed three probes out of three.

gpu_used_report() {
    local limit_mib=${GPU_PREFLIGHT_MAX_USED_MIB:-1024}
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' -v limit="$limit_mib" '$2 > limit {printf "GPU %s holds %s MiB; ", $1, $2}'
}

gpu_preflight() {
    local busy
    busy=$(gpu_used_report)
    if [[ -n $busy ]]; then
        echo "[preflight] $(hostname): ${busy}before this job loaded anything." >&2
        echo "[preflight] Something outside this job owns that memory (see nvidia-smi" \
             "--query-compute-apps). Resubmit with --exclude=$(hostname) and report the node." >&2
        exit 3
    fi
    echo "[preflight] all allocated GPUs are empty"
}

# Our own processes still on the GPUs: the previous phase's vLLM workers, which outlive their
# engine for a while after a forced shutdown. Only this user's, and only those holding a GPU.
own_gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r pid; do
        if [[ -n $pid && $(ps -o user= -p "$pid" 2>/dev/null) == "$USER" ]]; then echo "$pid"; fi
    done
}

wait_for_free_gpus() {
    local timeout=${GPU_WAIT_SECONDS:-180} waited=0 busy pids
    while busy=$(gpu_used_report) && [[ -n $busy ]]; do
        if (( waited >= timeout )); then
            pids=$(own_gpu_pids | tr '\n' ' ')
            if [[ -n $pids ]]; then
                echo "[driver] after ${waited}s ${busy}killing leftover processes of ours: $pids" >&2
                kill -9 $pids 2>/dev/null || true
                sleep 15
                busy=$(gpu_used_report)
            fi
            if [[ -n $busy ]]; then
                echo "[driver] GPUs still not free after ${waited}s: ${busy}" >&2
                return 1
            fi
            break
        fi
        sleep 5
        waited=$(( waited + 5 ))
    done
    (( waited > 0 )) && echo "[driver] GPUs free after ${waited}s"
    return 0
}

# Phases are resumable — each checkpoints its task list — so running one again after a failure
# repeats at most one chunk of work.
run_phase() {
    local attempt
    for attempt in 1 2; do
        wait_for_free_gpus || return 1
        if "$@"; then
            return 0
        fi
        echo "[driver] phase failed (attempt $attempt/2): $*" >&2
    done
    return 1
}

# Every compile and JIT cache on node-local disk, one directory per job.
#
# Two jobs sharing a node reached the same model at the same second. That
# configuration was not compiled yet, so both ran torch.compile into the same cache in the NFS
# home (~/.cache/vllm/torch_compile_cache). An NFS call never returned: both engines hung for 25
# minutes with no CPU or GPU activity, and after scancel one worker stayed a zombie with a thread
# in rpc_wait_bit_killable, still holding 27.6 GB of GPU 3. Reading the shared cache is safe;
# writing it from two jobs at once is not. So each job copies what is already compiled into its
# own directory under /tmp (local xfs, bound into the container by default) and writes only there.
#
# Appends the cache variables to the caller's ENV_ARGS array.
setup_local_caches() {
    JOB_CACHE=${JOB_CACHE:-/tmp/${USER}-lorehopqa-${SLURM_JOB_ID:-$$}}
    mkdir -p "$JOB_CACHE"/vllm/torch_compile_cache "$JOB_CACHE"/triton "$JOB_CACHE"/inductor "$JOB_CACHE"/xdg
    local shared=$HOME/.cache/vllm/torch_compile_cache
    if [[ -d $shared ]]; then
        # fa095d6e4c is the entry the hung compile left half-written.
        rsync -a --exclude 'fa095d6e4c/' "$shared/" "$JOB_CACHE/vllm/torch_compile_cache/" \
            && echo "[driver] compile cache seeded from $shared ($(du -sh "$JOB_CACHE/vllm" | cut -f1))" \
            || echo "[driver] could not seed the compile cache; compiling from scratch" >&2
    fi
    ENV_ARGS+=(
        --env "VLLM_CACHE_ROOT=$JOB_CACHE/vllm"
        --env "TRITON_CACHE_DIR=$JOB_CACHE/triton"
        --env "TORCHINDUCTOR_CACHE_DIR=$JOB_CACHE/inductor"
        --env "XDG_CACHE_HOME=$JOB_CACHE/xdg"
        --env "FLASHINFER_WORKSPACE_BASE=$JOB_CACHE/xdg"
    )
    trap 'rm -rf -- "${JOB_CACHE:?}"' EXIT
    echo "[driver] job caches in $JOB_CACHE"
}
