#!/usr/bin/bash

# Bottom-up multi-node check, in one allocation (INVESTIGATION_NOTES.md,
# Phase 2):
#   native:  srun mpi_hello; srun rmg-gpu                (baseline, no container)
#   flux:    one Flux instance in the container across all nodes, then
#            flux run mpi_hello  with Cray's libpmi first (RMG_LD_LIBRARY_PATH
#                                as in run_pipeline.py) and with Flux's first,
#            flux run mpi_hello gpu (GPU-buffer Allreduce),
#            flux run rmg-gpu    with whichever PMI order joined all ranks.
#
#   RMG_CASE=<dir with rmg_input + pseudopotentials> sbatch -N 2 multinode_test.slurm
#
# Each case runs in its own copy of RMG_CASE under $OUT, with a timeout.

#SBATCH -A mat201
#SBATCH -J multinode
#SBATCH -o %x-%j.out
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -N 2

set -uo pipefail

HERE="${HERE:-$SLURM_SUBMIT_DIR}"                       # this mpi_tests directory
SIF="${SIF:-/lustre/orion/mat201/scratch/rjmorelock/MatEnsemble_Update/images/matensemble-frontier-v0.5.8-ensemblefffit.sif}"
RMG="${RMG:-/lustre/orion/mat201/scratch/rjmorelock/MatEnsemble2/rmgdft/build-frontier-gpu/rmg-gpu}"
RMG_CASE="${RMG_CASE:?set RMG_CASE to a directory holding rmg_input and its pseudopotentials}"
OUT="${OUT:-$PWD/multinode-${SLURM_JOB_ID}}"
RMG_TIMEOUT="${RMG_TIMEOUT:-420}"
NATIVE_RMG="${NATIVE_RMG:-1}"
# Cray MPICH NIC policy for every Flux run. NOT NUMA: under Flux each task's
# cpuset spans the node, and NUMA then aborts MPI_Init on >1 node
# ("Rank N is not confined to a single NUMA node"; flux_mpi_diag.slurm).
NIC_POLICY="${NIC_POLICY:-BLOCK}"
N=$SLURM_NNODES
NP=$((N * 8))
mkdir -p "$OUT"

# Same as run_pipeline.py's RMG_LD_LIBRARY_PATH; FLUX_PMI_FIRST additionally
# puts the image's Flux libpmi/libpmi2 ahead of Cray's (the Perlmutter VASP fix).
RMG_LDP=/opt/cray/pe/lib64:/opt/cray/libfabric/2.3.1/lib64:/opt/cray/pals/1.8/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/rocm-6.3.3/lib:/opt/xpmem/2.7.4/lib:/sw/frontier/spack-envs/cpe24.03-cpu/opt/gcc-13.2/boost-1.85.0-3gvl6ws5xm7hrfkeyevgh45bv434ceol/lib:/sw/frontier/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/bzip2-1.0.8-st7di5r4yikef76nw4xenvocycgp3god/lib:/opt/cray_extra/lib64
FLUX_PMI_FIRST=/usr/local/lib/flux:$RMG_LDP

new_case() {  # fresh copy of the RMG input
    rm -rf "$OUT/$1"; mkdir -p "$OUT/$1"
    cp "$RMG_CASE"/rmg_input "$RMG_CASE"/*.upf "$OUT/$1/"
    [[ -f "$RMG_CASE/POSCAR" ]] && cp "$RMG_CASE/POSCAR" "$OUT/$1/"
}
rmg_report() {  # dir rc t0
    local l="$OUT/$1/rmg_input.00.log" o="$OUT/$1/rmg.stdout"
    echo "== $1: rc=$2 wall=$(( $(date +%s) - $3 ))s  MPI: $(grep -m1 -o '[0-9]* MPI processes/image' $l 2>/dev/null)  startup banners: $(grep -c 'R   M   G' $o 2>/dev/null)  SCF steps: $(grep -c 'TOTAL ENERGY' $l 2>/dev/null)  HSA/hipHostMalloc errors: $(grep -c -E 'hsaKmtSVMSetAttr|hipHostMalloc failed' $o 2>/dev/null)"
    grep -m1 -A4 "Processor Topology" "$l" 2>/dev/null | tr -s ' ' | tr '\n' ' '; echo
    grep 'TOTAL ENERGY' "$l" 2>/dev/null | tail -3
    grep -m1 "1-TOTAL " "$l" 2>/dev/null
}

echo "Job $SLURM_JOB_ID: $N node(s) $SLURM_JOB_NODELIST  RMG_CASE=$RMG_CASE  OUT=$OUT  NIC_POLICY=$NIC_POLICY"

# ---- native baseline (no container, no Flux) ------------------------------
module load PrgEnv-gnu/8.6.0 gcc-native/13.2 Core/24.00 bzip2 boost/1.85.0 craype-x86-milan cray-fftw \
    cray-hdf5-parallel craype-accel-amd-gfx90a rocm/6.3.1 >/dev/null 2>&1
SRUN="srun -N$N -n$NP -c7 --gpus-per-task=1 --gpu-bind=closest --threads-per-core=1 --cpu-bind=cores"
echo "== native mpi_hello"
MPICH_GPU_SUPPORT_ENABLED=1 $SRUN "$HERE/mpi_hello" gpu 2>&1 | sort -t' ' -k2 -n | sed 's/^/   /'
if [[ "$NATIVE_RMG" == 1 ]]; then
    new_case native_rmg; cd "$OUT/native_rmg"; t0=$(date +%s)
    OMP_PROC_BIND=false OMP_NUM_THREADS=6 RMG_NUM_THREADS=5 MPICH_GPU_SUPPORT_ENABLED=0 \
        timeout "$RMG_TIMEOUT" $SRUN "$RMG" rmg_input > rmg.stdout 2>&1
    rmg_report native_rmg $? $t0
fi
module reset >/dev/null 2>&1

# ---- Flux inside the container, one instance across all nodes -------------
INNER="$OUT/flux_inner.sh"
cat > "$INNER" <<EOF
cd "$OUT"
echo "== flux resource list (expect $N node(s), $((N * 8)) GPUs)"; flux resource list
common="-N$N -n$NP -c6 -g1 --env=OMP_PROC_BIND=false --env=OMP_NUM_THREADS=6 --env=RMG_NUM_THREADS=5 --env=MPICH_OFI_NIC_POLICY=$NIC_POLICY"
for order in cray_pmi_first flux_pmi_first; do
    if [[ \$order == cray_pmi_first ]]; then ldp=$RMG_LDP; else ldp=$FLUX_PMI_FIRST; fi
    echo "== flux mpi_hello, \$order"
    timeout 120 flux run \$common --env=LD_LIBRARY_PATH=\$ldp --env=MPICH_GPU_SUPPORT_ENABLED=0 $HERE/mpi_hello 2>&1 \
        | sort -t' ' -k2 -n | sed 's/^/   /' | tee hello_\$order.txt
    grep -q "rank 0/$NP " hello_\$order.txt && joined=\$order
    echo "== flux mpi_hello gpu (MPICH_GPU_SUPPORT_ENABLED=1), \$order"
    timeout 120 flux run \$common --env=LD_LIBRARY_PATH=\$ldp --env=MPICH_GPU_SUPPORT_ENABLED=1 $HERE/mpi_hello gpu 2>&1 \
        | grep -E "allreduce|rank 0/|rror" | sed 's/^/   /'
done
echo "== PMI order that joined all $NP ranks: \${joined:-NONE}"
if [[ -n "\${joined:-}" ]]; then
    if [[ \$joined == cray_pmi_first ]]; then ldp=$RMG_LDP; else ldp=$FLUX_PMI_FIRST; fi
    rm -rf flux_rmg; mkdir flux_rmg; cp "$RMG_CASE"/rmg_input "$RMG_CASE"/*.upf flux_rmg/; cd flux_rmg; t0=\$(date +%s)
    timeout $RMG_TIMEOUT flux run \$common --env=LD_LIBRARY_PATH=\$ldp --env=MPICH_GPU_SUPPORT_ENABLED=0 $RMG rmg_input > rmg.stdout 2>&1
    echo "\$? \$t0" > rc_t0
fi
EOF
srun -N "$N" -n "$N" --external-launcher --gpu-bind=closest --mpi=pmi2 \
    apptainer exec --bind /opt/cray/:/opt/cray --bind /usr/lib64/:/opt/cray_extra/lib64 --bind /var/spool/slurmd \
    "$SIF" flux start bash "$INNER"
if [[ -f "$OUT/flux_rmg/rc_t0" ]]; then
    read rc t0 < "$OUT/flux_rmg/rc_t0"; rmg_report flux_rmg $rc $t0
fi
