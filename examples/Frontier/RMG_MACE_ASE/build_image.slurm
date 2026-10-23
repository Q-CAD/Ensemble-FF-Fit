#!/usr/bin/bash

# Pull a MatEnsemble Frontier image into a .sif with Apptainer, on a node's
# NVMe (much faster than building on Lustre), then copy the finished .sif to
# IMAGE_DIR. A plain .sif is all the rest of this example needs -- no
# writable sandbox: launch_flux.slurm runs the .sif read-only, with
# Ensemble-FF-Fit and pyRMG put on PYTHONPATH from your own checkouts, and
# extra Python packages go into a derived image (matensemble-frontier.def,
# see README.md step 1c).
#
#   sbatch build_image.slurm
#   IMAGE_TAG=frontier-v0.5.9 IMAGE_DIR=/path/to/images sbatch build_image.slurm
#
# Keep IMAGE_TAG in step with the MatEnsemble version Ensemble-FF-Fit was
# last validated against (README.md step 1a). A pull takes ~5 min.

#SBATCH -A mat201
#SBATCH -J build_image
#SBATCH -o %x-%j.out
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -C nvme

set -euo pipefail

IMAGE_REPO="${IMAGE_REPO:-docker://ghcr.io/freddude2004/matensemble}"
IMAGE_TAG="${IMAGE_TAG:-frontier-v0.5.8}"
IMAGE_DIR="${IMAGE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIF_NAME="${SIF_NAME:-matensemble-${IMAGE_TAG}.sif}"

# Compute nodes reach ghcr.io only through the OLCF proxy.
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

NVME_DIR="/mnt/bb/${USER}"
mkdir -p "$NVME_DIR/apptainer_cache" "$NVME_DIR/apptainer_tmp" "$IMAGE_DIR"
export APPTAINER_CACHEDIR="$NVME_DIR/apptainer_cache"
export APPTAINER_TMPDIR="$NVME_DIR/apptainer_tmp"

cd "$NVME_DIR"
echo "Pulling ${IMAGE_REPO}:${IMAGE_TAG} -> ${SIF_NAME}"
apptainer pull --force "$SIF_NAME" "${IMAGE_REPO}:${IMAGE_TAG}"

# Record what's inside, so the README/notes can be checked against it.
apptainer exec "$SIF_NAME" bash -c '
    for p in matensemble flux-python torch mace-torch openequivariance; do
        python -c "import importlib.metadata as m, sys; print(sys.argv[1], m.version(sys.argv[1]))" $p 2>/dev/null \
            || echo "$p NOT INSTALLED"
    done
    flux version | head -3
    ls -d /opt/rocm-* /opt/mpich/* 2>/dev/null'

cp "$SIF_NAME" "$IMAGE_DIR/"
echo "Wrote ${IMAGE_DIR}/${SIF_NAME}"
