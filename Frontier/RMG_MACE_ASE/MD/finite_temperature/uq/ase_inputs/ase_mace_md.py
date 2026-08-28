"""
Finite-temperature MACE MD driver, reformatted to match ase_mace.py's
loading convention so MatEnsemble can dispatch it via
MDMatEnsemble.run_individual: a plain entry function taking parallel lists
of inputs, runnable standalone via
`python ase_mace_md.py '["ff1", ...]' '["struct1", ...]' '["output1", ...]' '["cfg1.json", ...]'`
using parse_list to parse each sys.argv entry.

`in_file` (the 4th list) is this driver's per-task MD-config json (timestep/
temperature schedule/friction/cycles/etc.) -- MDMatEnsemble.run_individual
now always passes a 4th list for this purpose (see base.py), populated here
by ff_training_and_uq.py's copy_and_spawn_md whenever 'in_file' is among the
matched option keys.

All the actual MD/defect-generation/supercell logic below is unchanged from
the original notebook implementation -- only the loading/entry convention
changed (named entry function instead of a bare __main__ block, corrected
import path for parse_list, and matching MDMatEnsemble's (ffield, structure,
output, in_file) argument order rather than the original (ffield, in_file,
structure, output)).
"""
import sys
import os
import torch
import gc
import math
from mace.calculators import MACECalculator
from EnsembleFFFit.utilities.general import parse_list
from ase import Atoms
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from itertools import count, product
import ase.units as units
import json
import numpy as np
import random


class MDInstabilityError(Exception):
    """
    Raised when the trajectory's energy has drifted too far from its last
    post-relaxation reference -- i.e. the structure has physically blown up
    (e.g. atoms overlapping after a defect mutation in a small cell) rather
    than genuinely sampling a new configuration. Continuing such a
    trajectory just burns compute on data that won't be DFT-convergible or
    useful for fine-tuning anyway (see the ~10^6-10^9 eV single-step energy
    jumps observed in practice before this check existed).
    """
    pass


def check_energy_stability(energy, reference_energy, n_atoms, max_deviation_per_atom):
    """
    Raise MDInstabilityError if `energy` is non-finite, or has drifted from
    `reference_energy` by more than `max_deviation_per_atom` eV/atom. A
    per-atom threshold (not an absolute one) since absolute energy scale
    depends heavily on atom count/composition (E0 references etc.) --
    typical solid-state cohesive energies are a few eV/atom, so a
    deviation of that same order already covers legitimate high-temperature/
    defect-formation energy shifts with headroom, while still catching
    genuine blow-ups (which are orders of magnitude larger in practice).
    """
    if not math.isfinite(energy):
        raise MDInstabilityError(f"energy is non-finite ({energy}) -- structure has blown up")

    deviation_per_atom = abs(energy - reference_energy) / n_atoms
    if deviation_per_atom > max_deviation_per_atom:
        raise MDInstabilityError(
            f"energy deviated {deviation_per_atom:.3f} eV/atom from the last relaxed reference "
            f"({reference_energy:.3f} eV) -- exceeds max_energy_deviation_per_atom="
            f"{max_deviation_per_atom} eV/atom (energy={energy:.3f} eV, n_atoms={n_atoms})"
        )


def generate_defect(atoms, defect_prob=0.75, rng=None, force=None):

    if rng is None:
        rng = random

    atoms = atoms.copy()

    # probability of doing nothing
    if rng.random() > defect_prob:
        return atoms

    defect_types = ["vacancy", "insertion", "mutation", "swap"]
    if not force:
        defect = rng.choice(defect_types)
    else:
        defect = force

    n = len(atoms)
    species = list(set(atoms.get_chemical_symbols()))
    cell = atoms.get_cell()

    if defect == "vacancy" and n > 1:
        idx = rng.randrange(n)
        del atoms[idx]

    elif defect == "insertion": # Could screen with atoms.get_all_distances(mic=True)
        sym = rng.choice(species)

        # random fractional position
        frac = np.random.rand(3)
        insert = Atoms(symbols=[sym], scaled_positions=[frac],
                       cell=atoms.cell, pbc=True)
        atoms += insert

    elif defect == "mutation":
        idx = rng.randrange(n)
        new_sym = rng.choice([s for s in species if s != atoms[idx].symbol])
        atoms[idx].symbol = new_sym

    elif defect == "swap":
        if n > 1 and len(species) > 1:
            sym1, sym2 = rng.sample(species, 2)
            i = rng.choice([k for k,a in enumerate(atoms) if a.symbol == sym1])
            j = rng.choice([k for k,a in enumerate(atoms) if a.symbol == sym2])
            atoms[i].symbol, atoms[j].symbol = atoms[j].symbol, atoms[i].symbol

    return atoms

def balanced_supercell(atoms, max_atoms=100, max_mult=6, score_std=False):

    natoms = len(atoms)
    cell = atoms.get_cell()
    lengths = np.linalg.norm(cell, axis=1)

    best_score = np.inf
    best_mult = None

    for nx, ny, nz in product(range(1, max_mult+1),
                              range(1, max_mult+1),
                              range(1, max_mult+1)):

        mult_atoms = natoms * nx * ny * nz
        if mult_atoms > max_atoms:
            continue

        new_lengths = lengths * np.array([nx, ny, nz])

        if score_std is True:
            score = np.std(new_lengths)
        else: # aspect ratio, usually better scoring
            score = max(new_lengths) / min(new_lengths)

        if score < best_score and (nx,ny,nz) != (1,1,1):
            best_score = score
            best_mult = (nx, ny, nz)

    if best_mult is None:
        return atoms.copy()

    return atoms.repeat(best_mult)

def run_dynamics(atoms, timestep, temperature, friction, frequency,
                 num_steps, to_attach, trajectory_path):

    dyn = Langevin(atoms=atoms,
                   timestep=timestep,
                   temperature_K=temperature,
                   friction=friction)

    # attach the callback to run based on frequency
    dyn.attach(to_attach, frequency)

    # --- trajectory and attachments
    traj = Trajectory(trajectory_path, 'a', atoms)
    dyn.attach(traj.write, frequency, atoms)

    # Run the MD -- traj.close() must still happen if to_attach raises
    # MDInstabilityError (it propagates straight out of dyn.run()), so
    # whatever frames were already written stay flushed/readable rather
    # than left in a half-closed state.
    try:
        dyn.run(num_steps)
    finally:
        traj.close()

    return

def temperature_scheduler(temp_value):

    if type(temp_value) == int:
        return temp_value
    elif type(temp_value) == tuple or type(temp_value) == list:
        options = np.linspace(temp_value[0], temp_value[1], temp_value[2])
        return float(random.sample(list(options), 1)[0])


def run_finite_temperature_md(ff_list, struct_list, output_list, in_file_list):
    """
    For each (force field, structure, output, config) quadruple, run a
    cycle of cell-minimization / low-temperature MD / cell-minimization /
    high-temperature MD / defect mutation (per generate_defect), recording
    energy/forces periodically, then write the trajectory and property
    history to `output`.
    """
    n = len(ff_list)
    assert all(len(lst) == n for lst in (struct_list, output_list, in_file_list)), \
        "All lists must be same length"

    for ff, struct, output, inp in zip(ff_list, struct_list, output_list, in_file_list):
        os.makedirs(output, exist_ok=True)

        # 1a) Load the MACE model
        calculator = MACECalculator(model_path=ff, device="cuda" if torch.cuda.is_available() else "cpu")

        # 1b) Load the structure and configuration files
        with open(inp) as fh:
            cfg = json.load(fh)
        init_conf = read(struct)
        #init_conf = balanced_supercell(init_conf, max_atoms=100)
        init_conf.set_calculator(calculator)

        # 2) Initialize the calculation
        step_counter = count()
        trajectory_path = os.path.join(output, 'md_run.traj')
        if os.path.exists(trajectory_path): # Remove file if it exists
            os.remove(trajectory_path)

        # 3) Set the MD integration and property attachment
        dt = cfg['timestep'] * units.fs
        friction = cfg['friction'] / units.fs

        property_dict = {}
        max_deviation_per_atom = cfg.get('max_energy_deviation_per_atom', 5.0)
        reference_energy_holder = {}  # mutable cell so update_status sees each cycle's latest reference

        def update_status():
            step = next(step_counter) * cfg['frequency']
            energy = init_conf.get_potential_energy()
            forces = init_conf.get_forces()
            property_dictionary = {
                'energy': float(energy),
                'fx': [float(f[0]) for f in forces],
                'fy': [float(f[1]) for f in forces],
                'fz': [float(f[2]) for f in forces],
            }
            property_dict[step] = property_dictionary

            reference_energy = reference_energy_holder.get('value')
            if reference_energy is not None:
                check_energy_stability(energy, reference_energy, len(init_conf), max_deviation_per_atom)

        # 4) Run the MD cycles
        aborted, abort_reason = False, None

        for i in range(cfg.get("cycles", 1)):

            # Cell Minimization

            fcf = FrechetCellFilter(init_conf)
            opt = FIRE(fcf)
            opt.run(fmax=cfg['fmax'], steps=cfg['optsteps'])
            # Reference resets after every minimization -- a defect mutation
            # legitimately changes atom count/composition (and so the
            # energy scale) between cycles, so the stability check must
            # compare against THIS cycle's own relaxed baseline, not a
            # stale one from a structure that no longer exists.
            reference_energy_holder['value'] = init_conf.get_potential_energy()

            # low temperature

            lt = temperature_scheduler(cfg['low_temperature'])
            try:
                run_dynamics(init_conf, dt, lt, friction, cfg['frequency'], cfg['nsteps'], update_status, trajectory_path)
            except MDInstabilityError as e:
                aborted, abort_reason = True, str(e)
                break

            # Cell Minimization

            fcf = FrechetCellFilter(init_conf)
            opt = FIRE(fcf)
            opt.run(fmax=cfg['fmax'], steps=cfg['optsteps'])
            reference_energy_holder['value'] = init_conf.get_potential_energy()

            # high temperature

            ht = temperature_scheduler(cfg['high_temperature'])
            MaxwellBoltzmannDistribution(init_conf, temperature_K=ht)
            try:
                run_dynamics(init_conf, dt, ht, friction, cfg['frequency'], cfg['nsteps'], update_status, trajectory_path)
            except MDInstabilityError as e:
                aborted, abort_reason = True, str(e)
                break

            # Potential Mutation
            init_conf = generate_defect(init_conf, defect_prob=cfg.get("defect_prob", 1.0))
            init_conf.set_calculator(calculator)

        if aborted:
            print(f"WARNING: {output}: MD aborted early -- {abort_reason}", file=sys.stderr)

        # --- after run: write the property_dict to disk once -- _status/
        # _abort_reason are extra top-level keys alongside the per-step
        # ones, not a restructure, so anything already reading this file
        # expecting {step: {...}, ...} keeps working unchanged. Writing
        # properties.json even on abort (rather than skipping it) matters:
        # finished_file-based skip logic only checks for this file's
        # existence, not its content, so leaving it unwritten would just
        # get this same doomed structure resubmitted on every future rerun.
        output_name = os.path.join(output, 'properties.json')
        record = dict(property_dict)
        record['_status'] = 'aborted_unstable' if aborted else 'complete'
        if abort_reason:
            record['_abort_reason'] = abort_reason
        with open(output_name, "w") as f:
            json.dump(record, f, indent=4)

        # --- after run: free Python memory ---
        if torch.cuda.is_available():
            del calculator, init_conf   # remove large objects
            gc.collect()                # free Python memory
            torch.cuda.empty_cache()    # release unreferenced GPU memory back to CUDA driver

    return {"status": "complete"}


if __name__ == "__main__":
    run_finite_temperature_md(parse_list(sys.argv[1]), parse_list(sys.argv[2]),
                               parse_list(sys.argv[3]), parse_list(sys.argv[4]))
