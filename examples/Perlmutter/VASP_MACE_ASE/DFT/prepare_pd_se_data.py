"""
One-time conversion of this example's pre-existing Pd-Se DFT data into the
extxyz shape the pipeline trains/tests against -- already run once to
produce DFT/training/ and DFT/testing/ as shipped; kept here for provenance
and so the conversion can be redone (or pointed at new data) by hand. Not a
pipeline stage: nothing in run_pipeline.py calls this.

Training (DFT/training/<label>/<label>.xyz): every
<training_source>/<label>/fitsnap_run_0/XYZ/*.xyz -- parse2fit's FitSNAP-
format output (see Pd-Se_MACE/Gen_1/Data/mace_training.yml). Energies/forces
are already eV / eV/Angstrom, but parse2fit writes stress in bar *and* in
VASP's own sign convention (FitSNAPEntry's units_dct, converted from
pymatgen's kBar ionic-step stress -- positive = compressive). MACE/ASE use
eV/Angstrom^3 with the opposite sign, so stress is rewritten as
-bar * 1e-4 * GPa. Each <label> becomes one combinable label in
build_ff_inputs.

Testing (DFT/testing/<chemsys>/<composition>/<mp-id>.xyz): one file per AIMD
trajectory under <testing_source>/<composition>/<mp-id>/<frame>/vasprun.xml,
frames in numeric order, read with pymatgen exactly the way DFT/vasp_dft.py
reads its own runs (final_energy, ionic_steps[-1] forces/stress) so test and
training energies share one convention. Incomplete vasprun.xml files (no
closing </modeling> tag -- e.g. 13 of Pd34Se11's 20 frames) and runs that
didn't converge electronically are skipped, not fatal.

Stress sign check: for the first testing frame converted, the
pymatgen-kBar -> ASE conversion used here (and in DFT/vasp_dft.py) is
compared against ASE's own vasprun.xml reader, which applies VASP's sign
convention itself; a mismatch aborts.

Writes through ASE's SinglePointCalculator + extxyz writer, the same as
EnsembleFFFit.potential.mace.write_training_xyz, so energy_key/forces_key/
stress_key in the MACE config.yml match by construction.

Usage:
    python prepare_pd_se_data.py \
        --training_source /pscratch/sd/r/rym/Pd-Se_MACE/Gen_1/Data/training_data/mace_data \
        --testing_source /pscratch/sd/r/rym/Pd-Se_MACE/NB_example/DFT/structures/aimd_sp/Pd-Se
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.stress import voigt_6_to_full_3x3_stress
from ase.units import GPa
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.outputs import Vasprun

HERE = Path(__file__).resolve().parent
BAR_TO_EV_PER_A3 = 1e-4 * GPa
KBAR_TO_EV_PER_A3 = 1e-1 * GPa


def _full_3x3(stress):
    stress = np.asarray(stress, dtype=float)
    return voigt_6_to_full_3x3_stress(stress) if stress.shape == (6,) else stress.reshape(3, 3)


def _write_frame(atoms, energy, forces, stress, output_path):
    atoms = atoms.copy()
    # float(): pymatgen's final_energy is a FloatWithUnit, which ASE's extxyz
    # writer would otherwise serialize as a quoted string ("-56.98 eV").
    atoms.calc = SinglePointCalculator(atoms, energy=float(energy), forces=forces, stress=stress)
    write(output_path, atoms, format='extxyz', append=True)


def convert_training(training_source, training_dest):
    total = 0
    for label_dir in sorted(p for p in Path(training_source).iterdir() if p.is_dir()):
        sources = sorted(glob.glob(str(label_dir / 'fitsnap_run_0' / 'XYZ' / '*.xyz')))
        if not sources:
            continue
        out_path = Path(training_dest) / label_dir.name / f"{label_dir.name}.xyz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()  # extxyz append -- start clean

        n = 0
        for source in sources:
            for atoms in read(source, index=':', format='extxyz'):
                results = atoms.calc.results
                stress = -_full_3x3(results['stress']) * BAR_TO_EV_PER_A3 if 'stress' in results else None
                _write_frame(atoms, results['energy'], results['forces'], stress, out_path)
                n += 1
        print(f"training/{label_dir.name}: {n} frame(s) from {len(sources)} file(s) -> {out_path}")
        total += n
    return total


def _frame_dirs(trajectory_dir):
    frames = [p for p in Path(trajectory_dir).iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(frames, key=lambda p: int(p.name))


def _is_complete(vasprun_path):
    with open(vasprun_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 4096))
        return b'</modeling>' in f.read()


def _check_stress_sign(vasprun_path, pymatgen_stress):
    ase_stress = read(vasprun_path, index=-1, format='vasp-xml').get_stress(voigt=False)
    if not np.allclose(pymatgen_stress, ase_stress, atol=1e-6):
        raise RuntimeError(
            f"Stress sign/unit check failed for {vasprun_path}:\n converted={pymatgen_stress}\n ase={ase_stress}")
    print(f"Stress sign check passed against ase.io.read on {vasprun_path}")


def convert_testing(testing_source, testing_dest, chemsys):
    total, checked = 0, False
    for comp_dir in sorted(p for p in Path(testing_source).iterdir() if p.is_dir()):
        for traj_dir in sorted(p for p in comp_dir.iterdir() if p.is_dir()):
            out_path = Path(testing_dest) / chemsys / comp_dir.name / f"{traj_dir.name}.xyz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                out_path.unlink()

            n, skipped = 0, []
            for frame_dir in _frame_dirs(traj_dir):
                vasprun_path = frame_dir / 'vasprun.xml'
                if not vasprun_path.exists() or not _is_complete(vasprun_path):
                    skipped.append(frame_dir.name)
                    continue
                vasprun = Vasprun(str(vasprun_path), parse_potcar_file=False, parse_dos=False,
                                   parse_eigen=False, parse_projected_eigen=False)
                if not vasprun.converged_electronic:
                    skipped.append(frame_dir.name)
                    continue
                final_step = vasprun.ionic_steps[-1]
                atoms = AseAtomsAdaptor.get_atoms(final_step['structure'])
                stress = -np.array(final_step['stress']) * KBAR_TO_EV_PER_A3
                if not checked:
                    _check_stress_sign(str(vasprun_path), stress)
                    checked = True
                _write_frame(atoms, vasprun.final_energy, np.array(final_step['forces']), stress, out_path)
                n += 1
            note = f" (skipped incomplete/unconverged frames {skipped})" if skipped else ""
            print(f"testing/{chemsys}/{comp_dir.name}/{traj_dir.name}: {n} frame(s) -> {out_path}{note}")
            total += n
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--training_source', required=True)
    parser.add_argument('--testing_source', required=True)
    parser.add_argument('--training_dest', default=str(HERE / 'training'))
    parser.add_argument('--testing_dest', default=str(HERE / 'testing'))
    parser.add_argument('--chemsys', default='Pd-Se')
    args = parser.parse_args()

    n_train = convert_training(args.training_source, args.training_dest)
    n_test = convert_testing(args.testing_source, args.testing_dest, args.chemsys)
    print(f"\nWrote {n_train} training frame(s) and {n_test} testing frame(s).")


if __name__ == '__main__':
    main()
