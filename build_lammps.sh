#!/usr/bin/env bash
# build_lammps.sh
# Usage: bash build_lammps.sh /absolute/path/to/env

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

# Load the modules required to compile
module load PrgEnv-gnu/8.5.0
module load gcc-native/12.3 
module load cudatoolkit/12.4
module load craype-accel-nvidia80

export MPICH_GPU_SUPPORT_ENABLED=1

# Create environment with desired conda packages (non-interactive)
echo "Creating conda env ..."
conda create --yes --prefix "$ENV_PATH" python=3.11 cmake cython -c conda-forge
source activate $ENV_PATH
ENV_PY="$ENV_PATH/bin/python"
ENV_PIP="$ENV_PATH/bin/pip"

# Activate the environment
"$ENV_PY" -m pip cache purge
"$ENV_PY" -m pip install --no-cache-dir --force-reinstall \
  --upgrade pip setuptools wheel \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  cuequivariance \
  cuequivariance-torch \
  cuequivariance-ops-torch-cu12 \
  cupy-cuda12x \
  pymatgen \
  ase \
  pymatgen-analysis-defects \
  mp-api \
  jupyter \
  pyarrow

MPICC="cc -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py==4.0.3

# Install ovito into the same prefix (uses a different channel)
# conda install -y --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.12.1

if [ ! -x "$ENV_PY" ]; then
  echo "Expected python executable not found at $ENV_PY"
  exit 1
fi

# Install Ensemble-FF-Fit
"$ENV_PY" -m pip install --no-cache-dir -e . --no-deps

# Make a build dir for cloning and installing repos
BUILD_DIR="$(pwd)/lammps_build"
[ -d "$BUILD_DIR" ] && rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Build mace from source 
REPO0="mace"
[ -d "$REPO0" ] && rm -rf "$REPO0"
git clone https://github.com/ACEsuit/mace.git
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

# git clone https://github.com/lammps/lammps.git
git clone -b patch_2Apr2025 https://github.com/lammps/lammps.git

cd lammps
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_Fortran_COMPILER=ftn \
      -D CMAKE_C_COMPILER=cc \
      -D CMAKE_CXX_COMPILER=CC \
      -D MPI_C_COMPILER=cc \
      -D MPI_CXX_COMPILER=CC \
      -D LAMMPS_EXCEPTIONS=ON \
      -D BUILD_SHARED_LIBS=ON \
      -D PKG_KOKKOS=yes \
      -D Kokkos_ARCH_AMPERE80=ON \
      -D Kokkos_ENABLE_CUDA=yes \
      -D PKG_MOLECULE=on \
      -D PKG_BODY=on \
      -D PKG_RIGID=on \
      -D PKG_MC=on \
      -D PKG_MANYBODY=on \
      -D PKG_REAXFF=on \
      -D PKG_REPLICA=on \
      -D PKG_QEQ=on \
      -D PKG_INTERLAYER=on \
      -D MLIAP_ENABLE_PYTHON=yes \
      -D PKG_PYTHON=yes \
      -D PKG_ML-SNAP=yes \
      -D PKG_ML-IAP=yes \
      -D PKG_ML-PACE=yes \
      -D PKG_SPIN=yes \
      -D PYTHON_EXECUTABLE=$ENV_PATH/bin/python \
      -D PYTHON_INCLUDE_DIR=$ENV_PATH/include/python3.11 \
      -D PYTHON_LIBRARY=$ENV_PATH/lib/libpython3.11.so \
      -D CMAKE_PREFIX_PATH=$ENV_PATH \
      -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_EXE_FLAGS="-dynamic" ../cmake

# Hack: Make twice to avoid errors with uncrecognized cuda flags, avoid sed issue
make -j32
conda deactivate
find . -path '*/CMakeFiles/lmp.dir/flags.make*' -print \
  -exec sed -i 's/ -Xcudafe --diag_suppress=unrecognized_pragma,--diag_suppress=128//' {} +
source activate $ENV_PATH
make -j32

make install-python

# Return to original working dir
cd ../../

echo "Done. Packages have been installed into: $ENV_PATH"
echo
echo "To use the environment interactively:"
echo "  source \"${CONDA_BASE}/etc/profile.d/conda.sh\""
echo "  conda activate \"$ENV_PATH\""
echo
echo "Or run commands directly with the env python/pip:"
