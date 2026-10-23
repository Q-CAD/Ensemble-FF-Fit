#!/usr/bin/bash

# Close the gap between rmg-gpu through Flux in the container (~48-61 s per
# SCF step, 99-atom Bi54Se45, 2 nodes, flux_preinit_rmg.slurm 5745915) and
# native srun (~29-31 s). Same input per case (RMG_CASE, capped at a few SCF
# steps), one variable at a time; each rank records its CPU/GPU binding.
#
#   CASES="base cpubind cpubind_numa" RMG_CASE=<dir> sbatch -N 2 rmg_timing.slurm
#
# Cases (all: libhip_preinit.so preload, MPICH_GPU_SUPPORT_ENABLED=0):
#   base             NIC policy BLOCK, no CPU/GPU binding
#   cpubind          + flux -o cpu-affinity=per-task
#   cpubind_numa     cpubind + MPICH_OFI_NIC_POLICY=NUMA
#   cpubind_gpubind  cpubind + flux -o gpu-affinity=per-task
#   cpubind_xnack0   cpubind + HSA_XNACK=0
#   cpubind_hmem     cpubind, NO preload, FI_HMEM=system instead (libfabric
#                    skips ROCr -- the other way found to keep HIP working)
#   native           srun outside the container (needs nothing from Flux)

#SBATCH -A mat201
#SBATCH -J rmgtiming
#SBATCH -o %x-%j.out
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -N 2

set -uo pipefail
HERE="${HERE:-$SLURM_SUBMIT_DIR}"
SIF="${SIF:-/lustre/orion/mat201/scratch/rjmorelock/MatEnsemble_Update/images/matensemble-frontier-v0.5.8-ensemblefffit.sif}"
RMG="${RMG:-/lustre/orion/mat201/scratch/rjmorelock/MatEnsemble2/rmgdft/build-frontier-gpu/rmg-gpu}"
RMG_CASE="${RMG_CASE:?set RMG_CASE}"
RMG_TIMEOUT="${RMG_TIMEOUT:-300}"
CASES="${CASES:-base cpubind cpubind_numa}"
OUT="${OUT:-$PWD/rmgtiming-${SLURM_JOB_ID}}"
N=$SLURM_NNODES; NP=$((N * 8))
mkdir -p "$OUT"
RMG_LDP=/opt/cray/pe/lib64:/opt/cray/libfabric/2.3.1/lib64:/opt/cray/pals/1.8/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/rocm-6.3.3/lib:/opt/xpmem/2.7.4/lib:/sw/frontier/spack-envs/cpe24.03-cpu/opt/gcc-13.2/boost-1.85.0-3gvl6ws5xm7hrfkeyevgh45bv434ceol/lib:/sw/frontier/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/bzip2-1.0.8-st7di5r4yikef76nw4xenvocycgp3god/lib:/opt/cray_extra/lib64
PRE=$HERE/libhip_preinit.so

# Per-rank wrapper: record binding, then exec the program.
cat > "$OUT/task.sh" <<'EOF'
#!/bin/bash
r=${FLUX_TASK_RANK:-${SLURM_PROCID:-x}}
echo "rank $r host $(hostname) cpus $(grep Cpus_allowed_list /proc/self/status | cut -f2) mems $(grep Mems_allowed_list /proc/self/status | cut -f2) ROCR=${ROCR_VISIBLE_DEVICES:--} CUDA=${CUDA_VISIBLE_DEVICES:--} HIP=${HIP_VISIBLE_DEVICES:--}" > "$BIND_OUT.$r"
exec "$@"
EOF
chmod +x "$OUT/task.sh"

report() {  # case rc t0 t1
    local d="$OUT/$1" l="$OUT/$1/rmg_input.00.log"
    echo "== $1: rc=$2 wall=$(( $4 - $3 ))s  $(grep -m1 -o '[0-9]* MPI processes/image' $l 2>/dev/null)  banners: $(grep -c 'R   M   G' $d/rmg.stdout 2>/dev/null)  SCF steps: $(grep -c 'TOTAL ENERGY' $l 2>/dev/null)  hsaKmtSVMSetAttr: $(grep -c hsaKmtSVMSetAttr $d/rmg.stdout 2>/dev/null)  MPI_Init errors: $(grep -c 'Fatal error in PMPI_Init' $d/rmg.stdout 2>/dev/null)"
    grep "quench:" "$l" 2>/dev/null | sed 's/^/   /'
    grep "TOTAL ENERGY" "$l" 2>/dev/null | tail -2 | sed 's/^/   /'
    cat "$d"/bind.* 2>/dev/null | sort -k2 -n | sed -n '1p;2p;9p' | sed 's/^/   bind: /'
}
newdir() { rm -rf "$OUT/$1"; mkdir -p "$OUT/$1"; cp "$RMG_CASE"/rmg_input "$RMG_CASE"/*.upf "$OUT/$1/"; }

echo "Job $SLURM_JOB_ID: $N node(s) $SLURM_JOB_NODELIST  CASES=$CASES  RMG_CASE=$RMG_CASE"

if [[ " $CASES " == *" native "* ]]; then
    module load PrgEnv-gnu/8.6.0 gcc-native/13.2 Core/24.00 bzip2 boost/1.85.0 craype-x86-milan cray-fftw \
        cray-hdf5-parallel craype-accel-amd-gfx90a rocm/6.3.1 >/dev/null 2>&1
    newdir native; cd "$OUT/native"; t0=$(date +%s)
    BIND_OUT=$OUT/native/bind OMP_PROC_BIND=false OMP_NUM_THREADS=6 RMG_NUM_THREADS=5 MPICH_GPU_SUPPORT_ENABLED=0 \
        timeout "$RMG_TIMEOUT" srun -N$N -n$NP -c7 --gpus-per-task=1 --gpu-bind=closest --threads-per-core=1 --cpu-bind=cores \
        "$OUT/task.sh" "$RMG" rmg_input > rmg.stdout 2>&1
    rc=$?; report native $rc $t0 $(date +%s)
    module reset >/dev/null 2>&1
fi

FLUX_CASES=$(echo " $CASES " | sed 's/ native / /')
if [[ -n "${FLUX_CASES// /}" ]]; then
cat > "$OUT/inner.sh" <<EOF
cd "$OUT"
flux resource list | sed -n 2p
base="-N$N -n$NP -c6 -g1 --env=LD_LIBRARY_PATH=$RMG_LDP --env=LD_PRELOAD=$PRE --env=MPICH_GPU_SUPPORT_ENABLED=0 --env=OMP_PROC_BIND=false --env=OMP_NUM_THREADS=6 --env=RMG_NUM_THREADS=5"
for c in $FLUX_CASES; do
    case \$c in
        base)            extra="--env=MPICH_OFI_NIC_POLICY=BLOCK" ;;
        cpubind)         extra="--env=MPICH_OFI_NIC_POLICY=BLOCK -o cpu-affinity=per-task" ;;
        cpubind_numa)    extra="--env=MPICH_OFI_NIC_POLICY=NUMA -o cpu-affinity=per-task" ;;
        cpubind_gpubind) extra="--env=MPICH_OFI_NIC_POLICY=BLOCK -o cpu-affinity=per-task -o gpu-affinity=per-task" ;;
        cpubind_xnack0)  extra="--env=MPICH_OFI_NIC_POLICY=BLOCK -o cpu-affinity=per-task --env=HSA_XNACK=0" ;;
        cpubind_hmem)    extra="--env=MPICH_OFI_NIC_POLICY=BLOCK -o cpu-affinity=per-task --env=FI_HMEM=system --env=LD_PRELOAD=" ;;
        *) echo "unknown case \$c"; continue ;;
    esac
    flux cancel --all >/dev/null 2>&1; flux queue idle --timeout=60s >/dev/null 2>&1
    rm -rf \$c; mkdir -p \$c; cp $RMG_CASE/rmg_input $RMG_CASE/*.upf \$c/; cd \$c; t0=\$(date +%s)
    timeout $RMG_TIMEOUT flux run \$base \$extra --env=BIND_OUT=$OUT/\$c/bind $OUT/task.sh $RMG rmg_input > rmg.stdout 2>&1
    echo "\$? \$t0 \$(date +%s)" > rc_t0
    cd "$OUT"
done
EOF
srun -N "$N" -n "$N" --external-launcher --gpu-bind=closest --mpi=pmi2 \
    apptainer exec --bind /opt/cray/:/opt/cray --bind /usr/lib64/:/opt/cray_extra/lib64 --bind /var/spool/slurmd \
    "$SIF" flux start bash "$OUT/inner.sh"
for c in $FLUX_CASES; do
    [[ -f "$OUT/$c/rc_t0" ]] && { read rc t0 t1 < "$OUT/$c/rc_t0"; report $c $rc $t0 $t1; }
done
fi
