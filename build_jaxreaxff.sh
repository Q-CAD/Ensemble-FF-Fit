#!/usr/bin/env bash
# build_jaxreaxff.sh
# Usage: bash build_jaxreaxff.sh /absolute/path/to/env

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
echo "Creating conda env (python=3.13, jax[cuda12]==0.4.35, pymatgen, ase, mpi4py, scikit-learn, seaborn)..."
conda create --yes --prefix "$ENV_PATH" python=3.13 numpy=2.1.3 "jax[cuda12]==0.4.35"
conda install --yes --prefix "$ENV_PATH" pymatgen pymatgen-analysis-defects mp-api ase mpi4py scikit-learn seaborn frozendict -c conda-forge

# Install ovito into the same prefix (uses a different channel)
# echo "Installing ovito into environment..."
# conda install --yes --prefix "$ENV_PATH" -c https://conda.ovito.org -c conda-forge ovito=3.12.1

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
BUILD_DIR="$(pwd)/jaxreaxff_build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone and install JAX-ReaxFF (editable install recommended for development)
REPO1="Jax-ReaxFF"
[ -d "$REPO1" ] && rm -rf "$REPO1"
git clone https://github.com/Q-CAD/JAX-ReaxFF.git "$REPO1"
cd "$REPO1"
"$ENV_PY" -m pip install --no-cache-dir -e .
cd ..

# Clone and install MatEnsemble (editable, skip deps to avoid changing env)
REPO2="MatEnsemble"
[ -d "$REPO2" ] && rm -rf "$REPO2"
git clone https://github.com/Q-CAD/MatEnsemble.git "$REPO2"
cd "$REPO2"
"$ENV_PY" -m pip install --no-cache-dir -e . --no-deps
cd ..

# Return to original working dir and pip install Ensemble-FF-Fit
cd ..
"$ENV_PY" -m pip install --no-cache-dir -e . --no-deps

# Uninstall data-classes backport to avoid runtime errors
"$ENV_PY" -m pip uninstall --yes dataclasses

echo "Done. Packages have been installed into: $ENV_PATH"
echo
echo "To use the environment interactively:"
echo "  source \"${CONDA_BASE}/etc/profile.d/conda.sh\""
echo "  conda activate \"$ENV_PATH\""
echo
echo "Or run commands directly with the env python/pip:"
echo "  $ENV_PY -c 'import sys; print(sys.version)'"
echo "  $ENV_PY -c 'import jax; print(jax.__version__)'"
