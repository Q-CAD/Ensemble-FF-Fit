#!/usr/bin/env bash
# build_jaxreaxff.sh
# Usage: bash build_base.sh /absolute/path/to/env

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
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create environment with desired conda packages (non-interactive)
echo "Creating conda env (python=3.13, pymatgen, ase, mpi4py, scikit-learn, seaborn)..."
conda create --yes --prefix "$ENV_PATH" python=3.13 pymatgen pymatgen-analysis-defects mp-api ase mpi4py scikit-learn seaborn frozendict -c conda-forge

# Activate the environment
source activate $ENV_PATH

# Install ovito into the same prefix (uses a different channel)
conda install -y --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.12.1

# Use the environment's python and pip explicitly (guaranteed)
ENV_PY="$ENV_PATH/bin/python"
ENV_PIP="$ENV_PATH/bin/pip"

if [ ! -x "$ENV_PY" ]; then
  echo "Expected python executable not found at $ENV_PY"
  exit 1
fi

# Upgrade pip/tools inside the env
"$ENV_PY" -m pip install --upgrade pip setuptools wheel

# Make a build dir for cloning and installing repos
BUILD_DIR="$(pwd)/base_build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone and install vaspFlux (editable install recommended for development)
REPO1="vaspflux"
[ -d "$REPO1" ] && rm -rf "$REPO1"
git clone https://code.ornl.gov/rym/vaspflux.git "$REPO1"
cd "$REPO1"
"$ENV_PY" -m pip install --no-cache-dir -e .
cd ..

# Clone and install MatEnsemble (editable, skip deps to avoid changing env)
REPO2="MatEnsemble"
[ -d "$REPO2" ] && rm -rf "$REPO2"
git clone https://github.com/Q-CAD/MatEnsemble.git "$REPO2"
cd "$REPO2"

# Clone and build LAMMPS
git clone https://github.com/lammps/lammps.git
cd lammps
mkdir build
cd build
cmake  -D CMAKE_BUILD_TYPE=Release \
            -D LAMMPS_EXCEPTIONS=ON \
            -D BUILD_SHARED_LIBS=ON \
            -D PKG_MANYBODY=ON -D PKG_MC=ON -D PKG_MOLECULE=ON -D PKG_KSPACE=ON -D PKG_REPLICA=ON -D PKG_ASPHERE=ON -D PKG_ML-SNAP=ON -D PKG_REAXFF=ON \
            -D PKG_RIGID=ON -D PKG_MPIIO=ON -D PKG_QEQ=ON -D PKG_PYTHON=On \
            -D PKG_INTERLAYER=ON -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_EXE_FLAGS="-dynamic" ../cmake

make -j64
make install-python

# return back to repo root and install MatEnsemble
cd ../../
"$ENV_PY" -m pip install --no-cache-dir -e . --no-deps
cd ..

# Return to original working dir and pip install Ensemble-FF-Fit
cd ..
"$ENV_PY" -m pip install --no-cache-dir -e . --no-deps

echo "Done. Packages have been installed into: $ENV_PATH"
echo
echo "To use the environment interactively:"
echo "  source \"${CONDA_BASE}/etc/profile.d/conda.sh\""
echo "  conda activate \"$ENV_PATH\""
echo
echo "Or run commands directly with the env python/pip:"
