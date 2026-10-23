#!/usr/bin/bash

# In a Flux job spanning >1 node inside the container, HIP sees 0 GPUs after
# MPI_Init (GTL: hsa_init -> HSA_STATUS_ERROR_OUT_OF_RESOURCES), though the
# same tasks see all GPUs without MPI (flux_gpu_diag.slurm, 5745716). Try
# each candidate fix on mpi_hello (2 nodes x 8), then run rmg-gpu (RMG_CASE,
# e.g. capped at a few SCF steps) with the first one that gives every rank
# a GPU. Cases (env added to the working RMG chore env, NIC policy BLOCK):
#   hipfirst      mpi_hello_hipfirst: GPUs queried before MPI_Init
#   xnack0        HSA_XNACK=0 (the image sets 1, for LAMMPS)
#   xnack_unset   HSA_XNACK removed
#   hmem_system   FI_HMEM=system (libfabric skips ROCr)
#   rocr_mon0     FI_MR_ROCR_CACHE_MONITOR_ENABLED=0

#SBATCH -A mat201
#SBATCH -J fluxgpumatrix
#SBATCH -o %x-%j.out
#SBATCH -t 00:20:00
#SBATCH -q debug
#SBATCH -N 2

set -uo pipefail
HERE="${HERE:-$SLURM_SUBMIT_DIR}"
SIF="${SIF:-/lustre/orion/mat201/scratch/rjmorelock/MatEnsemble_Update/images/matensemble-frontier-v0.5.8-ensemblefffit.sif}"
RMG="${RMG:-/lustre/orion/mat201/scratch/rjmorelock/MatEnsemble2/rmgdft/build-frontier-gpu/rmg-gpu}"
RMG_CASE="${RMG_CASE:-}"
RMG_TIMEOUT="${RMG_TIMEOUT:-420}"
OUT="${OUT:-$PWD/fluxgpumatrix-${SLURM_JOB_ID}}"
N=$SLURM_NNODES; NP=$((N * 8))
mkdir -p "$OUT"
RMG_LDP=/opt/cray/pe/lib64:/opt/cray/libfabric/2.3.1/lib64:/opt/cray/pals/1.8/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/rocm-6.3.3/lib:/opt/xpmem/2.7.4/lib:/sw/frontier/spack-envs/cpe24.03-cpu/opt/gcc-13.2/boost-1.85.0-3gvl6ws5xm7hrfkeyevgh45bv434ceol/lib:/sw/frontier/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/bzip2-1.0.8-st7di5r4yikef76nw4xenvocycgp3god/lib:/opt/cray_extra/lib64

# Per-task wrapper: UNSET="VAR ..." removes variables before exec.
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
case_env() {
    case \$1 in
        hipfirst|baseline) echo "" ;;
        xnack0)       echo "--env=HSA_XNACK=0" ;;
        xnack_unset)  echo "--env=UNSET=HSA_XNACK" ;;
        hmem_system)  echo "--env=FI_HMEM=system" ;;
        rocr_mon0)    echo "--env=FI_MR_ROCR_CACHE_MONITOR_ENABLED=0" ;;
    esac
}
good=""
for c in baseline hipfirst xnack0 xnack_unset hmem_system rocr_mon0; do
    bin=$HERE/mpi_hello; [[ \$c == hipfirst ]] && bin=$HERE/mpi_hello_hipfirst
    for gpu in 0 1; do
        arg=""; [[ \$gpu == 1 ]] && arg=gpu
        timeout 90 flux run \$base \$(case_env \$c) --env=MPICH_GPU_SUPPORT_ENABLED=\$gpu $OUT/task.sh \$bin \$arg > h_\${c}_gpu\$gpu.txt 2>&1
        ng=\$(grep -c 'gpu_count=[1-9]' h_\${c}_gpu\$gpu.txt); nr=\$(grep -c '^rank ' h_\${c}_gpu\$gpu.txt)
        echo "== \$c MPICH_GPU_SUPPORT_ENABLED=\$gpu: \$nr/$NP ranks printed, \$ng with a GPU; \$(grep -o -m1 '[a-z-]* allreduce: [A-Z]*' h_\${c}_gpu\$gpu.txt | tr '\n' ' ') \$(grep -m1 -o 'hsa_init.*HSA_STATUS_[A-Z_]*' h_\${c}_gpu\$gpu.txt)"
        [[ \$gpu == 1 && \$ng == $NP && -z "\$good" && \$c != hipfirst ]] && good=\$c
    done
done
echo "== first case giving all $NP ranks a GPU (with GPU support on): \${good:-NONE}"
if [[ -n "\$good" && -n "$RMG_CASE" ]]; then
    mkdir -p rmg_\$good; cp $RMG_CASE/rmg_input $RMG_CASE/*.upf rmg_\$good/; cd rmg_\$good; t0=\$(date +%s)
    timeout $RMG_TIMEOUT flux run \$base \$(case_env \$good) --env=MPICH_GPU_SUPPORT_ENABLED=0 $OUT/task.sh $RMG rmg_input > rmg.stdout 2>&1
    rc=\$?
    l=rmg_input.00.log
    echo "== rmg (\$good): rc=\$rc wall=\$(( \$(date +%s) - t0 ))s  \$(grep -m1 -o '[0-9]* MPI processes/image' \$l)  devices listed: \$(grep -c '^device' \$l)  banners: \$(grep -c 'R   M   G' rmg.stdout)  SCF steps: \$(grep -c 'TOTAL ENERGY' \$l)"
    grep -m1 -A4 "Processor Topology" \$l | tr -s ' ' | tr '\n' ' '; echo
    grep "quench:" \$l | tail -4
fi
EOF

srun -N "$N" -n "$N" --external-launcher --gpu-bind=closest --mpi=pmi2 \
    apptainer exec --bind /opt/cray/:/opt/cray --bind /usr/lib64/:/opt/cray_extra/lib64 --bind /var/spool/slurmd \
    "$SIF" flux start bash "$OUT/inner.sh"
