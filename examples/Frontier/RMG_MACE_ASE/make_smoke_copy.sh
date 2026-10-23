#!/usr/bin/bash

# Make a ready-to-run scratch copy of this example for a 1-node smoke test
# with workflow_config.smoke.yaml (README.md step 6): everything except
# DFT/, plus only a dozen small structures under DFT/ (both isolated atoms,
# the six 11-atom BiSe vacancies, four 12-atom BiSe EoS points -- about a
# minute of RMG each), with DFT/vdW_quench.yml pointed at your rmg-gpu build
# and the MACE foundation model copied into its three places.
#
#   bash make_smoke_copy.sh <dest> <rmgdft-dir> [model.model]
#
# <rmgdft-dir> is the RMG source tree you built in (it must hold
# build-frontier-gpu/rmg-gpu and XC/vdW_kernel_table). Without a
# model.model argument the MACE-OMat-0 medium checkpoint is downloaded
# (login nodes only -- compute nodes have no direct internet).
# Writes only under <dest>, which must not already exist.

set -euo pipefail

[[ $# -ge 2 ]] || { sed -n '3,16p' "$0"; exit 2; }
DEST="$1"
RMGDFT="$(cd "$2" && pwd)"
MODEL="${3:-}"
SRC="$(cd "$(dirname "$0")" && pwd)"

[[ ! -e "$DEST" ]] || { echo "$DEST already exists" >&2; exit 1; }
[[ -x "$RMGDFT/build-frontier-gpu/rmg-gpu" ]] || { echo "no $RMGDFT/build-frontier-gpu/rmg-gpu" >&2; exit 1; }
[[ -f "$RMGDFT/XC/vdW_kernel_table" ]] || { echo "no $RMGDFT/XC/vdW_kernel_table" >&2; exit 1; }

KEEP_STRUCTURES=(
    isolated_elements/Bi
    isolated_elements/Se
    training/Defects/Bi-Se/BiSe/mp-27902/vacancy/vacancy{0,1,2,3,4,5}
    validation/EoS/Bi-Se/BiSe/mp-27902/volume_{0.85,0.925,1.075,1.15}
)

mkdir -p "$DEST"
rsync -a --exclude /DFT/ --exclude '__pycache__' "$SRC"/ "$DEST"/
mkdir -p "$DEST/DFT/uq"
cp "$SRC"/DFT/*.py "$DEST/DFT/"
for rel in "${KEEP_STRUCTURES[@]}"; do
    mkdir -p "$DEST/DFT/$rel"
    cp "$SRC/DFT/$rel/POSCAR" "$DEST/DFT/$rel/POSCAR"
done

sed -e "s|^rmg_executable: .*|rmg_executable: \"$RMGDFT/build-frontier-gpu/rmg-gpu\"|" \
    -e "s|^vdwdf_kernel_filepath: .*|vdwdf_kernel_filepath: \"$RMGDFT/XC/vdW_kernel_table\"|" \
    "$SRC/DFT/vdW_quench.yml" > "$DEST/DFT/vdW_quench.yml"

if [[ -z "$MODEL" ]]; then
    MODEL="$DEST/model.model"
    curl -fL -o "$MODEL" \
        https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model
fi
for d in FF/generation_0/mace-omat-medium MD/finite_temperature/uq/generation_0/mace-omat-medium \
         FF/generation_1/mace-omat-medium; do
    mkdir -p "$DEST/$d"
    cp "$MODEL" "$DEST/$d/model.model"
done

echo "Smoke copy ready: $DEST ($(find "$DEST/DFT" -name POSCAR | wc -l) structures)"
echo "  cd $DEST && CONFIG=workflow_config.smoke.yaml sbatch launch_flux.slurm"
