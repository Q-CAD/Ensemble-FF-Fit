#!/usr/bin/bash

# Why does Cray MPICH abort in MPI_Init ("MPIDI_CRAY_ofi_get_nic_index: OFI
# invalid value for environment variable", every process reporting rank 0)
# when flux run spans 2 nodes inside the container, when the same thing on 1
# node works? (multinode_test.slurm, job 5745339; INVESTIGATION_NOTES.md.)
# Runs only mpi_hello, a few seconds per case:
#   env dump per node; 1-node control; 2 nodes x each MPICH_OFI_NIC_POLICY
#   with MPICH_OFI_NIC_VERBOSE=2; 2 nodes with the outer srun's
#   SLURM_*/PALS_*/PMI_* variables removed (Flux's own PMI_FD/PMI_RANK/
#   PMI_SIZE kept).

#SBATCH -A mat201
#SBATCH -J fluxmpidiag
#SBATCH -o %x-%j.out
#SBATCH -t 00:15:00
#SBATCH -q debug
#SBATCH -N 2

set -uo pipefail
HERE="${HERE:-$SLURM_SUBMIT_DIR}"
SIF="${SIF:-/lustre/orion/mat201/scratch/rjmorelock/MatEnsemble_Update/images/matensemble-frontier-v0.5.8-ensemblefffit.sif}"
OUT="${OUT:-$PWD/fluxmpidiag-${SLURM_JOB_ID}}"
N=$SLURM_NNODES; NP=$((N * 8))
mkdir -p "$OUT"
RMG_LDP=/opt/cray/pe/lib64:/opt/cray/libfabric/2.3.1/lib64:/opt/cray/pals/1.8/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/rocm-6.3.3/lib:/opt/xpmem/2.7.4/lib:/opt/cray_extra/lib64

# Wrapper run as each Flux task: optionally strip the outer launcher's env,
# optionally dump env, then exec mpi_hello.
cat > "$OUT/task.sh" <<'EOF'
#!/bin/bash
if [[ "${STRIP_OUTER:-0}" == 1 ]]; then
    keep_fd=$PMI_FD keep_rank=$PMI_RANK keep_size=$PMI_SIZE
    for v in $(compgen -e | grep -E '^(SLURM_|PALS_|PMI_|PMIX_|CRAY_PMI|SLURMD_)'); do unset "$v"; done
    export PMI_FD=$keep_fd PMI_RANK=$keep_rank PMI_SIZE=$keep_size
fi
if [[ -n "${DUMP_ENV:-}" && ( "$FLUX_TASK_LOCAL_ID" == 0 ) ]]; then
    env | sort | grep -E '^(SLURM_|PALS_|PMI|FLUX_|MPICH|FI_|CRAY|OMP|HSA|ROCR|CUDA|HIP)' > "$DUMP_ENV.$(hostname)"
fi
exec "$@"
EOF
chmod +x "$OUT/task.sh"

cat > "$OUT/inner.sh" <<EOF
cd "$OUT"
flux resource list
base="-c6 -g1 --env=LD_LIBRARY_PATH=$RMG_LDP --env=MPICH_GPU_SUPPORT_ENABLED=0 --env=MPICH_OFI_NIC_VERBOSE=2"
summ() { echo "   -> \$(grep -c '^rank ' \$1) rank lines, max size: \$(grep -o 'rank [0-9]*/[0-9]*' \$1 | cut -d/ -f2 | sort -n | tail -1)  allreduce: \$(grep -o 'allreduce: [A-Z]*' \$1)  MPI_Init errors: \$(grep -c 'Fatal error in PMPI_Init' \$1)"; grep -m3 -i -E 'nic|policy|invalid' \$1 | grep -v '^rank' | cut -c1-240 | sed 's/^/      /'; }

echo "== env dump, 2 nodes"
flux run -N$N -n$N --env=DUMP_ENV=$OUT/env $OUT/task.sh true

echo "== control: 1 node x 8, NIC policy NUMA (as the pipeline's RMG chores)"
timeout 60 flux run -N1 -n8 \$base --env=MPICH_OFI_NIC_POLICY=NUMA $OUT/task.sh $HERE/mpi_hello > c1.txt 2>&1; summ c1.txt

for pol in NUMA BLOCK GPU ROUND-ROBIN; do
    echo "== 2 nodes x 8, NIC policy \$pol"
    timeout 60 flux run -N$N -n$NP \$base --env=MPICH_OFI_NIC_POLICY=\$pol $OUT/task.sh $HERE/mpi_hello > n2_\$pol.txt 2>&1; summ n2_\$pol.txt
done

echo "== 2 nodes x 8, NUMA, outer SLURM_/PALS_/PMI_ env stripped"
timeout 60 flux run -N$N -n$NP \$base --env=MPICH_OFI_NIC_POLICY=NUMA --env=STRIP_OUTER=1 $OUT/task.sh $HERE/mpi_hello > n2_strip.txt 2>&1; summ n2_strip.txt

echo "== 2 nodes x 1 (one rank per node), NUMA"
timeout 60 flux run -N$N -n$N \$base --env=MPICH_OFI_NIC_POLICY=NUMA $OUT/task.sh $HERE/mpi_hello > n2_1pn.txt 2>&1; summ n2_1pn.txt
EOF

srun -N "$N" -n "$N" --external-launcher --gpu-bind=closest --mpi=pmi2 \
    apptainer exec --bind /opt/cray/:/opt/cray --bind /usr/lib64/:/opt/cray_extra/lib64 --bind /var/spool/slurmd \
    "$SIF" flux start bash "$OUT/inner.sh"
echo "== env dumps"; for f in "$OUT"/env.*; do echo "-- $f"; cat "$f"; done
