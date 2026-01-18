#!/usr/bin/env bash
# build_lammps.sh
# Usage: bash build_unified.sh /absolute/path/to/env

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <env_path>"
  # if script is sourced, return; else exit
  (return 0 2>/dev/null) && return 1
fi

ENV_PATH=$(realpath "$1")
echo "Environment path: $ENV_PATH"

# Remove any existing env (optional)
if [ -d "$ENV_PATH" ]; then
  echo "Removing existing environment at $ENV_PATH"
  rm -rf "$ENV_PATH"
fi

# Ensure conda command is available in this shell
# (this is safe to run even if conda is already in PATH)
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "$CONDA_BASE" ]; then
  echo "conda not found in PATH. Please ensure conda is installed and available."
  exit 1
fi

# Create environment with desired conda packages (non-interactive)
echo "Creating conda env ..."
conda create --yes \
	     --prefix "$ENV_PATH" \
	     -c conda-forge \
	     python=3.11 \
	     "numpy<2" \
	     cmake \
	     cython \
	     pymatgen \
             pymatgen-analysis-defects \
             matminer \
             mp-api \
             ase \
             scikit-learn \
             seaborn \
             frozendict \
             py3dmol \
             jupyter \
	     pyarrow

source activate $ENV_PATH
ENV_PY="$ENV_PATH/bin/python"
ENV_PIP="$ENV_PATH/bin/pip"

# Export NumPy ABI flags
export PIP_NO_BUILD_ISOLATION=1
export CMAKE_ARGS="-DNUMPY_EXPERIMENTAL_ARRAY_API=0"

# Activate the environment
"$ENV_PY" -m pip cache purge
"$ENV_PY" -m pip install --no-cache-dir --force-reinstall --upgrade pip setuptools wheel 

#  --extra-index-url https://download.pytorch.org/whl/cu124 \
#  torch==2.5.0+cu124 \
#  cuequivariance==0.6.1 \
#  cuequivariance-torch==0.6.1 \
#  cuequivariance-ops-torch-cu12==0.6.1 \
#  pymatgen ase pyarrow

#  --extra-index-url https://download.pytorch.org/whl/cu124 \
#  torch==2.5.0+cu124 \
#  numpy==2.0.0 \
#  cuequivariance==0.6.0 \
#  cuequivariance-torch==0.6.0 \
#  cuequivariance-ops-torch-cu12==0.3.0 \
#  mace-torch "numpy==2.0.0" \
#  cupy-cuda12x==13.5.1 \
#  pymatgen "numpy==2.0.0" \
#  ase "numpy==2.0.0" \
#  pyarrow

# torch-sim-atomistic "numpy==2.0.0" \

# MPICC="cc -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py==4.0.3

# Install ovito into the same prefix (uses a different channel)
# conda install -y --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.12.1

if [ ! -x "$ENV_PY" ]; then
  echo "Expected python executable not found at $ENV_PY"
  exit 1
fi

# Install Ensemble-FF-Fit
"$ENV_PY" -m pip install --no-cache-dir -e . --no-deps

# Remove mace-torch to avoid clashes with mace-freeze branch
#"$ENV_PY" -m pip uninstall -y mace-torch

# Make a build dir for cloning and installing repos
BUILD_DIR="$(pwd)/unified_build"
[ -d "$BUILD_DIR" ] && rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Build mace-torch from the mace-freeze branch 
REPO0="mace-freeze"
[ -d "$REPO0" ] && rm -rf "$REPO0"
git clone -b mace-freeze https://github.com/7radians/mace-freeze.git # "$REPO0"
cd "$REPO0"
"$ENV_PY" -m pip install --no-cache-dir .
cd ..

# Clone and install MatEnsemble (editable, skip deps to avoid changing env)
REPO1="MatEnsemble"
[ -d "$REPO1" ] && rm -rf "$REPO1"
git clone https://github.com/Q-CAD/MatEnsemble.git "$REPO1"
cd "$REPO1"
"$ENV_PY" -m pip install --no-cache-dir -e . --no-deps
cd ..

# Clone and install vaspFlux (editable install recommended for development)
REPO2="vaspflux"
[ -d "$REPO2" ] && rm -rf "$REPO2"
git clone https://code.ornl.gov/rym/vaspflux.git "$REPO2"
cd "$REPO2"
"$ENV_PY" -m pip install --no-cache-dir -e . --no-deps
cd ..

# Clone and install the develop branch of HeteroBuilder
REPO3="vdw_structures"
[ -d "$REPO3" ] && rm -rf "$REPO3"
git clone --branch develop --single-branch https://github.com/Q-CAD/HeteroBuilder.git "$REPO3"
cd "$REPO3"
"$ENV_PY" -m pip install --no-cache-dir -e . --no-deps
cd ..

# Clone and install the develop branch of parse2fit
REPO4="parse2fit"
[ -d "$REPO4" ] && rm -rf "$REPO4"
git clone --branch develop --single-branch https://code.ornl.gov/rym/parse2fit.git "$REPO4"
cd "$REPO4"
"$ENV_PY" -m pip install --no-cache-dir -e . --no-deps
cd ..

# Return to original working dir and pip install Ensemble-FF-Fit
cd ..
"$ENV_PY" -m pip install --no-cache-dir -e . --no-deps

# Return to original working dir
# cd ../../

echo "Done. Packages have been installed into: $ENV_PATH"
echo
echo "To use the environment interactively:"
echo "  source \"${CONDA_BASE}/etc/profile.d/conda.sh\""
echo "  conda activate \"$ENV_PATH\""
echo
echo "Or run commands directly with the env python/pip:"
