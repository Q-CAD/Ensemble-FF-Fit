#!/usr/bin/bash

# Does initialising HIP before MPI_Init (libhip_preinit.so via LD_PRELOAD)
# let rmg-gpu run across nodes through Flux inside the container?
# (flux_gpu_mpi_matrix.slurm, 5745756: mpi_hello built to query the GPUs
# before MPI_Init saw them on all 16 ranks and passed a GPU-buffer
# Allreduce; queried after MPI_Init, 0 GPUs.)
#   1. mpi_hello + preload, GPU support off and on
#   2. rmg-gpu + preload on RMG_CASE (e.g. capped at a few SCF steps)
#   3. retry the cases 5745756 left inconclusive (its runs queued behind a
#      timed-out job), draining the Flux queue first:
#      FI_HMEM=system, FI_MR_ROCR_CACHE_MONITOR_ENABLED=0, HSA_XNACK unset
#
#   RMG_CASE=<dir> sbatch -N 2 flux_preinit_rmg.slurm

#SBATCH -A mat201
#SBATCH -J fluxpreinit
#SBATCH -o %x-%j.out
#SBATCH -t 00:20:00
#SBATCH -q debug
#SBATCH -N 2

set -uo pipefail
HERE="${HERE:-$SLURM_SUBMIT_DIR}"
SIF="${SIF:-/lustre/orion/mat201/scratch/rjmorelock/MatEnsemble_Update/images/matensemble-frontier-v0.5.8-ensemblefffit.sif}"
RMG="${RMG:-/lustre/orion/mat201/scratch/rjmorelock/MatEnsemble2/rmgdft/build-frontier-gpu/rmg-gpu}"
RMG_CASE="${RMG_CASE:?set RMG_CASE}"
RMG_TIMEOUT="${RMG_TIMEOUT:-480}"
OUT="${OUT:-$PWD/fluxpreinit-${SLURM_JOB_ID}}"
N=$SLURM_NNODES; NP=$((N * 8))
mkdir -p "$OUT"
RMG_LDP=/opt/cray/pe/lib64:/opt/cray/libfabric/2.3.1/lib64:/opt/cray/pals/1.8/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/rocm-6.3.3/lib:/opt/xpmem/2.7.4/lib:/sw/frontier/spack-envs/cpe24.03-cpu/opt/gcc-13.2/boost-1.85.0-3gvl6ws5xm7hrfkeyevgh45bv434ceol/lib:/sw/frontier/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/bzip2-1.0.8-st7di5r4yikef76nw4xenvocycgp3god/lib:/opt/cray_extra/lib64
PRE=$HERE/libhip_preinit.so

cat > "$OUT/task.sh" <<'EOF'
#!/bin/bash
for v in ${UNSET:-}; do unset "$v"; done
exec "$@"
EOF
chmod +x "$OUT/task.sh"

cat > "$OUT/inner.sh" <<EOF
cd "$OUT"
flux resource list
base="-N$N -n$NP -c6 -g1 --env=LD_LIBRARY_PATH=$RMG_LDP --env=MPICH_OFI_NIC_POLICY=BLOCK --env=OMP_PROC_BIND=false --env=OMP_NUM_THREADS=6 --env=RMG_NUM_THREADS=5"
drain() { flux cancel --all >/dev/null 2>&1; flux queue idle --timeout=60s >/dev/null 2>&1; }
hello() {  # tag gpu-support extra-args...
    local tag=\$1 g=\$2; shift 2
    local arg=""; [[ \$g == 1 ]] && arg=gpu
    drain
    timeout 90 flux run \$base --env=MPICH_GPU_SUPPORT_ENABLED=\$g "\$@" $OUT/task.sh $HERE/mpi_hello \$arg > h_\$tag.txt 2>&1
    echo "== \$tag (GPU support \$g): rc=\$? \$(grep -c '^rank ' h_\$tag.txt)/$NP ranks, \$(grep -c 'gpu_count=[1-9]' h_\$tag.txt) with a GPU; \$(grep -o 'allreduce: [A-Z]*' h_\$tag.txt | tr '\n' ' ') \$(grep -m1 -o 'hsa_init.*HSA_STATUS_[A-Z_]*' h_\$tag.txt)"
}
hello preload_gpu0 0 --env=LD_PRELOAD=$PRE --env=HIP_PREINIT_VERBOSE=1
hello preload_gpu1 1 --env=LD_PRELOAD=$PRE --env=HIP_PREINIT_VERBOSE=1
grep -h "^rank" h_preload_gpu1.txt | awk '{print \$3, \$6}' | sort | head -16 | sed 's/^/   /'

drain
mkdir -p rmg_preload; cp $RMG_CASE/rmg_input $RMG_CASE/*.upf rmg_preload/; cd rmg_preload; t0=\$(date +%s)
timeout $RMG_TIMEOUT flux run \$base --env=LD_PRELOAD=$PRE --env=MPICH_GPU_SUPPORT_ENABLED=0 $RMG rmg_input > rmg.stdout 2>&1
rc=\$?; l=rmg_input.00.log
echo "== rmg + preload: rc=\$rc wall=\$(( \$(date +%s) - t0 ))s  \$(grep -m1 -o '[0-9]* MPI processes/image' \$l 2>/dev/null)  devices listed: \$(grep -c '^device' \$l 2>/dev/null)  banners: \$(grep -c 'R   M   G' rmg.stdout)  SCF steps: \$(grep -c 'TOTAL ENERGY' \$l 2>/dev/null)  HSA errors: \$(grep -c -E 'hsaKmtSVMSetAttr|hipHostMalloc failed|HSA_STATUS' rmg.stdout)"
grep -m1 -A4 "Processor Topology" \$l 2>/dev/null | tr -s ' ' | tr '\n' ' '; echo
grep "quench:" \$l 2>/dev/null | tail -4
grep "TOTAL ENERGY" \$l 2>/dev/null | tail -3
cd "$OUT"

hello hmem_system 0 --env=FI_HMEM=system
hello rocr_mon0 0 --env=FI_MR_ROCR_CACHE_MONITOR_ENABLED=0
hello xnack_unset 0 --env=UNSET=HSA_XNACK
EOF

srun -N "$N" -n "$N" --external-launcher --gpu-bind=closest --mpi=pmi2 \
    apptainer exec --bind /opt/cray/:/opt/cray --bind /usr/lib64/:/opt/cray_extra/lib64 --bind /var/spool/slurmd \
    "$SIF" flux start bash "$OUT/inner.sh"
