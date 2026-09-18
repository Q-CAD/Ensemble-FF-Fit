#!/usr/bin/env bash
#
# Run from inside an already-active interactive Slurm allocation, either
# CPU-only:
#   interactive -N 1 -n 4 -c 1 --mem=32gb --time=1-00:00:00
#   bash matensemble_submission.sh
# or GPU (1-6 GPUs -- Pathfinder's docs mention 2/4/6 specifically, but
# nothing here hardcodes those three; any count in that range works):
#   interactive -N 1 -n 4 -c 1 --mem=32gb --time=1-00:00:00 -p gpu --gres=gpu:2
#   bash matensemble_submission.sh
#
# Whether any GPUs were actually requested is detected from Slurm itself
# (see detect_requested_gpus below), not from which flags were typed --
# --nv is only added to the apptainer invocation, and the nvidia-smi sanity
# check only runs inside the container, when that detection finds a nonzero
# GPU count. A CPU-only allocation skips both entirely and gets a GPU-less
# R.json, rather than erroring out the way `nvidia-smi` reporting 0 GPUs
# used to unconditionally do here.
#
# Launches a single-rank Flux instance inside the Apptainer container. GPUs
# are manually enumerated in a static R.json rather than relying on Flux's
# own live NVML/hwloc discovery -- confirmed (2026-09) that discovery
# doesn't see this system's GPUs even with --nv, while `nvidia-smi` itself
# works fine inside the container, so this is specifically a Flux/hwloc-side
# gap (very plausibly this Flux build's hwloc wasn't compiled with
# NVML/CUDA support at all), not a driver-passthrough problem -- no bind
# flag fixes that. This mirrors the exact same static-R.json workaround
# examples/Perlmutter/VASP_ReaxFF_LAMMPs/launch_multi_node.slurm already
# needed (for a different reason -- multi-node broker merging there; single
# -node GPU discovery here), same underlying mechanism (a resource.toml
# with noverify=true pointed at a hand-written R.json, via FLUX_CONF_DIR).
#
# --bind flags are built as an array, not a single backslash-continued
# string -- a stray trailing space after a `\` line-continuation silently
# breaks continuation and can produce mangled arguments, which may be what
# caused the earlier "/software: destination must be an absolute path"
# error (both sides of that bind spec are genuinely absolute paths, and
# nothing in the container's own rootfs or this login node conflicts with
# it -- see pipeline/FRICTION_LOG.md). The pre-flight loop below also
# surfaces immediately, before apptainer runs at all, if any bind source
# genuinely isn't visible from this specific compute node.

set -euo pipefail

# Host-side (outside the container, before any apptainer/nvidia-smi call) --
# how many GPUs, if any, Slurm actually handed this allocation. Checked
# against a real -p gpu --gres=gpu:2 allocation (2026-09-17): none of these
# env vars were confirmed set in this job's own environment (this cluster's
# Slurm build apparently doesn't export them to job steps the way some
# do), so `scontrol show job` (parsing the `gres/gpu=N` field of
# ReqTRES/AllocTRES) is the confirmed-working path here, not just a
# fallback -- $SLURM_JOB_ID itself IS always set inside any Slurm job/
# allocation, unlike the naming of any specific GPU-count env var, which
# varies across Slurm versions/GRES plugin configs. The env-var checks are
# kept first anyway (cheap, no subprocess) in case a differently-configured
# partition on this same cluster does set one of them.
detect_requested_gpus() {
    local n=""
    # SLURM_GPUS_ON_NODE/SLURM_GPUS are plain counts (e.g. "6"); tested
    # directly (unlike the ID-list vars below, confirmed against the real
    # allocation's own env would need one to actually be set, which wasn't
    # the case here -- kept as plain counts per Slurm's own documented
    # meaning for these two, not assumed comma-separated).
    for var in SLURM_GPUS_ON_NODE SLURM_GPUS; do
        local val="${!var:-}"
        if [ -n "${val}" ]; then
            n="${val}"
            break
        fi
    done
    # SLURM_JOB_GPUS/SLURM_STEP_GPUS are comma-separated GPU device IDs
    # (e.g. "0,1"), not counts -- only checked if neither plain-count var
    # above was set.
    if [ -z "${n}" ]; then
        for var in SLURM_JOB_GPUS SLURM_STEP_GPUS; do
            local val="${!var:-}"
            if [ -n "${val}" ]; then
                n=$(( $(grep -o ',' <<< "${val}" | wc -l) + 1 ))
                break
            fi
        done
    fi
    if [ -z "${n}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
        n=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oP 'gres/gpu=\K[0-9]+' | head -1 || true)
    fi
    echo "${n:-0}"
}

HOST_NGPUS=$(detect_requested_gpus)
if [ "${HOST_NGPUS}" -gt 0 ]; then
    echo "=== Slurm allocation requests ${HOST_NGPUS} GPU(s) -- will pass --nv and run the nvidia-smi check ==="
    APPTAINER_GPU_ARGS=(--nv)
else
    echo "=== no GPUs detected in this Slurm allocation -- launching CPU-only, no --nv, no nvidia-smi check ==="
    APPTAINER_GPU_ARGS=()
fi

# SANDBOX must point at your OWN built sandbox -- there is no default that
# works for anyone else, and it must NOT live under /scratch (see this
# example's own README.md, "Building the container": /scratch is Lustre,
# and a writable Apptainer sandbox there was observed to have its Python
# stdlib silently corrupted mid-session more than once; build/keep it under
# a /projects/<proj>/proj-shared/<you>/... path -- NFS-backed -- instead).
SANDBOX=/projects/<your-project>/proj-shared/<you>/containers/pathfinder/matensemble_sandbox

# ENSEMBLE_FF_FIT/PIPELINE_DIR are self-located from this script's own path
# (repo_root/examples/<cluster>/<example>/, this example's own fixed depth
# under the repo root -- same as every other example here), not hardcoded,
# so cloning/copying this example elsewhere needs no path edits beyond
# SANDBOX above.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENSEMBLE_FF_FIT="$(cd "${SELF_DIR}/../../.." && pwd)"
PIPELINE_DIR="${SELF_DIR}"
# GPU count is not hardcoded here -- HOST_NGPUS above only gates WHETHER the
# nvidia-smi check runs at all; the inner script below still derives the
# actual count from `nvidia-smi -L` at runtime, inside the container, after
# --nv passthrough, same as before. That's the real ground truth for
# however many GPUs --gres=gpu:N handed this job, the same way CORE_RANGE is
# derived from `nproc` rather than hardcoded -- no edit needed here when
# changing --gres (or dropping it entirely) in the `interactive` command.

# --- Pre-flight: confirm every bind source is actually visible from this
# shell before handing them to apptainer.
#
# /etc/flexiblasrc(.d) and the two flexiblas plugin dirs are required, not
# precautionary -- CONFIRMED (2026-09-16): pw.x needs libflexiblas.so.3 (from
# /lib64, covered by the /lib64:/host_lib64 bind below) AND FlexiBLAS's own
# config file plus its backend plugin .so (libflexiblas_openblas-openmp.so),
# which FlexiBLAS resolves by bare filename via its own internal search
# path, not visible to plain LD_LIBRARY_PATH. These are narrow, RHEL-specific
# paths that don't already exist in this container's Debian rootfs, so
# binding them at their exact host path is safe (unlike binding all of
# /lib64 over the container's own -- see the /lib64:/host_lib64 comment
# below for why that's NOT done).
for src in /software /lib64 /etc/flexiblasrc /etc/flexiblasrc.d /lib64/flexiblas /usr/lib64/flexiblas \
           "$ENSEMBLE_FF_FIT" "$PIPELINE_DIR" "$SANDBOX"; do
    if [ ! -e "$src" ]; then
        echo "ERROR: bind source '$src' does not exist/isn't visible from this shell -- aborting." >&2
        exit 1
    fi
done

BIND_ARGS=(
    --bind /software:/software
    --bind /lib64:/host_lib64
    --bind /etc/flexiblasrc:/etc/flexiblasrc
    --bind /etc/flexiblasrc.d:/etc/flexiblasrc.d
    --bind /lib64/flexiblas:/lib64/flexiblas
    --bind /usr/lib64/flexiblas:/usr/lib64/flexiblas
    --bind "${ENSEMBLE_FF_FIT}:/opt/ensemble-ff-fit"
    --bind "${PIPELINE_DIR}:/opt/pipeline"
)

# Inner script, run INSIDE the container by the srun'd apptainer exec below.
# Written under PIPELINE_DIR (bind-mounted at /opt/pipeline) so the same
# physical file is reachable from both outside (to write it) and inside
# (to execute it) the container -- referenced by its container-internal
# path, not its host path, in the apptainer exec command further down.
INNER_SCRIPT_HOST=$(mktemp --tmpdir="${PIPELINE_DIR}" flux_container_XXXXXX.sh)
INNER_SCRIPT_CONTAINER="/opt/pipeline/$(basename "${INNER_SCRIPT_HOST}")"
trap 'rm -f "${INNER_SCRIPT_HOST}"' EXIT

cat > "${INNER_SCRIPT_HOST}" << INNER_EOF
#!/usr/bin/env bash
set -euo pipefail

NPROC=\$(nproc)
CORE_RANGE="0-\$((NPROC - 1))"

# HOST_NGPUS is baked in at generation time from the outer script's own
# Slurm-based detection (see detect_requested_gpus above) -- not re-derived
# in here, so a CPU-only allocation never even attempts nvidia-smi (which
# may not exist, or may hang/error, on a node with no NVIDIA driver at all).
HOST_NGPUS=${HOST_NGPUS}
if [ "\${HOST_NGPUS}" -gt 0 ]; then
    echo "=== nvidia-smi sanity check (\${HOST_NGPUS} GPU(s) requested via Slurm) ==="
    GPU_LIST=\$(nvidia-smi -L) || GPU_LIST=""
    echo "\${GPU_LIST:-<nvidia-smi produced no output -- GPU passthrough (--nv) may not be working>}"
    NGPUS=\$(printf '%s\n' "\${GPU_LIST}" | grep -c '^GPU ' || true)
    if [ "\${NGPUS}" -eq 0 ]; then
        echo "ERROR: Slurm allocated \${HOST_NGPUS} GPU(s) but nvidia-smi reported 0 inside the container -- check --nv passthrough." >&2
        exit 1
    fi
    GPU_RANGE="0-\$((NGPUS - 1))"
    GPU_STANZA=", \"gpu\": \"\${GPU_RANGE}\""
else
    echo "=== no GPU flags detected in this allocation -- skipping nvidia-smi, CPU-only R.json ==="
    NGPUS=0
    GPU_STANZA=""
fi

echo "=== detected \${NPROC} core(s) (range \${CORE_RANGE}), \${NGPUS} GPU(s) ==="

mkdir -p /tmp/fluxcfg
cat > /tmp/R.json << JSON_EOF
{
  "version": 1,
  "execution": {
    "R_lite": [
      {"rank": "0", "children": {"core": "\${CORE_RANGE}"\${GPU_STANZA}}}
    ],
    "starttime": 0.0,
    "expiration": 0.0,
    "nodelist": ["\$(hostname)"]
  }
}
JSON_EOF

cat > /tmp/fluxcfg/resource.toml << 'TOML_EOF'
[resource]
path = "/tmp/R.json"
noverify = true
TOML_EOF

echo "=== R.json written: ==="
cat /tmp/R.json
echo
echo "=== starting Flux with this static resource description ==="
echo "    Run 'flux resource list' once inside to confirm cores + GPUs both show up."
echo "    Then: cd /opt/pipeline && /opt/venv/bin/python run_pipeline.py --config workflow_config.yaml --stage converge_dft_data"
echo

export FLUX_CONF_DIR=/tmp/fluxcfg
flux start
INNER_EOF
chmod +x "${INNER_SCRIPT_HOST}"

echo "=== Launching single-rank Flux instance (srun -N1 -n1) ==="
echo

# NOTE: --mpi=pmi2 matches Frontier's own convention (see
# examples/Frontier/RMG_MACE_ASE/README.md) for bootstrapping the Flux
# broker itself via srun -- Pathfinder's module env separately reported
# SLURM_MPI_TYPE=pmix when the QE module was loaded, so if this srun fails
# to bootstrap Flux (not the same thing as the GPU-discovery issue above),
# try --mpi=pmix instead (check available plugins with `srun --mpi=list`).
srun -N 1 -n 1 --external-launcher --mpi=pmi2 --pty \
    apptainer exec "${APPTAINER_GPU_ARGS[@]}" "${BIND_ARGS[@]}" "${SANDBOX}" bash "${INNER_SCRIPT_CONTAINER}"
