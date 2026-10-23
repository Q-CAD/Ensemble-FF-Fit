#!/usr/bin/bash

# Why do Flux tasks inside the container see 0 GPUs (hsa_init:
# HSA_STATUS_ERROR_OUT_OF_RESOURCES) in multinode_test.slurm's 2-node job
# (5745571), when every 1-node job saw all 8? That job ran native srun GPU
# steps first; this one starts Flux straight away. Per Flux task: device
# nodes, device cgroup, thread/process limits, rocminfo agent count, and
# mpi_hello's GPU count -- for -N1 and -N2 runs in the same 2-node instance.

#SBATCH -A mat201
#SBATCH -J fluxgpudiag
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH -N 2

set -uo pipefail
HERE="${HERE:-$SLURM_SUBMIT_DIR}"
SIF="${SIF:-/lustre/orion/mat201/scratch/rjmorelock/MatEnsemble_Update/images/matensemble-frontier-v0.5.8-ensemblefffit.sif}"
OUT="${OUT:-$PWD/fluxgpudiag-${SLURM_JOB_ID}}"
N=$SLURM_NNODES
mkdir -p "$OUT"
RMG_LDP=/opt/cray/pe/lib64:/opt/cray/libfabric/2.3.1/lib64:/opt/cray/pals/1.8/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/rocm-6.3.3/lib:/opt/xpmem/2.7.4/lib:/opt/cray_extra/lib64

cat > "$OUT/probe.sh" <<'EOF'
#!/bin/bash
# One line block per task: what this task can see of the GPUs.
h=$(hostname); r=$FLUX_TASK_RANK
{
echo "--- task $r on $h (local $FLUX_TASK_LOCAL_ID) CUDA=$CUDA_VISIBLE_DEVICES ROCR=${ROCR_VISIBLE_DEVICES:--}"
echo "kfd: $(ls -la /dev/kfd 2>&1)"
echo "dri render nodes: $(ls /dev/dri 2>&1 | tr '\n' ' ')"
echo "cgroup: $(cat /proc/self/cgroup | tr '\n' ' ')"
cg=$(awk -F: '$1==0{print $3}' /proc/self/cgroup)
[[ -n "$cg" ]] && echo "cgroup pids.max=$(cat /sys/fs/cgroup$cg/pids.max 2>&1) pids.current=$(cat /sys/fs/cgroup$cg/pids.current 2>&1)"
echo "ulimit -u=$(ulimit -u) -l=$(ulimit -l) threads-max=$(cat /proc/sys/kernel/threads-max)"
echo "rocminfo GPU agents: $(timeout 30 /opt/rocm-6.3.3/bin/rocminfo 2>&1 | grep -c 'gfx90a') ; first error: $(timeout 30 /opt/rocm-6.3.3/bin/rocminfo 2>&1 | grep -i -m1 -E 'error|fail')"
} > "$PROBE_OUT.$r" 2>&1
EOF
chmod +x "$OUT/probe.sh"

cat > "$OUT/inner.sh" <<EOF
cd "$OUT"
flux resource list
echo "== initial program (rank-0 broker): kfd \$(ls -la /dev/kfd 2>&1)  rocminfo agents: \$(timeout 30 /opt/rocm-6.3.3/bin/rocminfo 2>&1 | grep -c gfx90a)"
for shape in "-N1 -n1" "-N2 -n2" "-N2 -n16"; do
    tag=\$(echo \$shape | tr -d ' -')
    echo "== probe \$shape"
    flux run \$shape -c6 -g1 --env=PROBE_OUT=$OUT/probe_\$tag $OUT/probe.sh
    cat $OUT/probe_\$tag.* | grep -E "^---|kfd|render|rocminfo|pids|ulimit" | sort -u | head -12
    echo "== mpi_hello \$shape"
    timeout 60 flux run \$shape -c6 -g1 --env=LD_LIBRARY_PATH=$RMG_LDP --env=MPICH_OFI_NIC_POLICY=BLOCK \
        --env=MPICH_GPU_SUPPORT_ENABLED=0 $HERE/mpi_hello 2>&1 | grep -E "gpu_count|allreduce|hsa_init" | awk '{print \$1,\$2,\$3,\$5,\$6}' | sort | uniq -c | head -6
done
EOF

srun -N "$N" -n "$N" --external-launcher --gpu-bind=closest --mpi=pmi2 \
    apptainer exec --bind /opt/cray/:/opt/cray --bind /usr/lib64/:/opt/cray_extra/lib64 --bind /var/spool/slurmd \
    "$SIF" flux start bash "$OUT/inner.sh"
