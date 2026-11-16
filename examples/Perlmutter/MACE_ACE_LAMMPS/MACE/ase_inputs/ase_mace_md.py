"""
Finite-temperature MACE MD driver. Consolidates the companion Pd-Se
pipeline's own two near-identical files (the "shared" finite_temperature/
mace_ase_inputs/ase_mace_md.py and finite_temperature/training_2/
mace_ase_inputs/ase_mace_md.py, which differed ONLY by passing
constant_volume=True to both FrechetCellFilter(...) calls) into one driver
with a `constant_volume` config flag (read from each task's own in_file
json, same mechanism 'fmax'/'optsteps' already use) -- default False
(today's "shared driver" behavior).

Why constant_volume matters (same reasoning the companion pipeline's own
training_2 copy documented): a tier whose whole point is to hold cell
volume fixed at a specific rescaled value (e.g. an EOS-style sample) needs
constant_volume=True, because the plain cell minimization below would
otherwise relax the cell back toward its own preferred ~100% volume on the
very first minimization step, silently erasing the rescaling before any
thermal MD even starts. constant_volume=True (a real ASE FrechetCellFilter
kwarg) keeps ion positions/cell SHAPE free to relax (removing
rattle/rescaling-induced local stress) while keeping the box VOLUME itself
fixed. Set `"constant_volume": true` in a tier's own in_file json to get
this; omit it (or set false) for the original ramped-temperature behavior
that lets the cell find its own volume.

Dispatched by MatEnsemble via EnsembleFFFit.base.MDMatEnsemble.run_individual.
"""
import gc
import json
import math
import os
import random
import sys
from itertools import count, product

import ase.units as units
import numpy as np
import torch
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE
from mace.calculators import MACECalculator

from EnsembleFFFit.utilities.general import parse_list


class MDInstabilityError(Exception):
    """
    Raised when the trajectory's energy has drifted too far from its last
    post-relaxation reference -- i.e. the structure has physically blown up
    (e.g. atoms overlapping after a defect mutation in a small cell) rather
    than genuinely sampling a new configuration. Continuing such a
    trajectory just burns compute on data that won't be useful for
    fine-tuning anyway.
    """
    pass


def check_energy_stability(energy, reference_energy, n_atoms, max_deviation_per_atom):
    """
    Raise MDInstabilityError if `energy` is non-finite, or has drifted from
    `reference_energy` by more than `max_deviation_per_atom` eV/atom. A
    per-atom threshold (not an absolute one) since absolute energy scale
    depends heavily on atom count/composition -- typical solid-state
    cohesive energies are a few eV/atom, so a deviation of that same order
    already covers legitimate high-temperature/defect-formation energy
    shifts with headroom, while still catching genuine blow-ups (which are
    orders of magnitude larger in practice).
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
    defect = force if force else rng.choice(defect_types)

    n = len(atoms)
    species = list(set(atoms.get_chemical_symbols()))

    if defect == "vacancy" and n > 1:
        idx = rng.randrange(n)
        del atoms[idx]

    elif defect == "insertion":
        sym = rng.choice(species)
        frac = np.random.rand(3)
        insert = Atoms(symbols=[sym], scaled_positions=[frac], cell=atoms.cell, pbc=True)
        atoms += insert

    elif defect == "mutation":
        idx = rng.randrange(n)
        new_sym = rng.choice([s for s in species if s != atoms[idx].symbol])
        atoms[idx].symbol = new_sym

    elif defect == "swap":
        if n > 1 and len(species) > 1:
            sym1, sym2 = rng.sample(species, 2)
            i = rng.choice([k for k, a in enumerate(atoms) if a.symbol == sym1])
            j = rng.choice([k for k, a in enumerate(atoms) if a.symbol == sym2])
            atoms[i].symbol, atoms[j].symbol = atoms[j].symbol, atoms[i].symbol

    return atoms


def balanced_supercell(atoms, max_atoms=100, max_mult=6, score_std=False):
    natoms = len(atoms)
    cell = atoms.get_cell()
    lengths = np.linalg.norm(cell, axis=1)

    best_score = np.inf
    best_mult = None

    for nx, ny, nz in product(range(1, max_mult + 1), range(1, max_mult + 1), range(1, max_mult + 1)):
        mult_atoms = natoms * nx * ny * nz
        if mult_atoms > max_atoms:
            continue

        new_lengths = lengths * np.array([nx, ny, nz])
        score = np.std(new_lengths) if score_std else max(new_lengths) / min(new_lengths)

        if score < best_score and (nx, ny, nz) != (1, 1, 1):
            best_score = score
            best_mult = (nx, ny, nz)

    if best_mult is None:
        return atoms.copy()
    return atoms.repeat(best_mult)


def run_dynamics(atoms, timestep, temperature, friction, frequency, num_steps, to_attach, trajectory_path):
    dyn = Langevin(atoms=atoms, timestep=timestep, temperature_K=temperature, friction=friction)
    dyn.attach(to_attach, frequency)

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


def temperature_scheduler(temp_value):
    if isinstance(temp_value, int):
        return temp_value
    elif isinstance(temp_value, (tuple, list)):
        options = np.linspace(temp_value[0], temp_value[1], temp_value[2])
        return float(random.sample(list(options), 1)[0])


def run_finite_temperature_md(ff_list, struct_list, output_list, in_file_list):
    """
    For each (force field, structure, output, config) quadruple, run a
    cycle of cell-minimization / low-temperature MD / cell-minimization /
    high-temperature MD / defect mutation (per generate_defect), recording
    energy/forces periodically, then write the trajectory and property
    history to `output`.

    `constant_volume` (in the per-task in_file json, default False): see
    this module's own docstring.
    """
    n = len(ff_list)
    assert all(len(lst) == n for lst in (struct_list, output_list, in_file_list)), \
        "All lists must be same length"

    for ff, struct, output, inp in zip(ff_list, struct_list, output_list, in_file_list):
        os.makedirs(output, exist_ok=True)

        calculator = MACECalculator(model_path=ff, enable_cueq=True,
                                    device="cuda" if torch.cuda.is_available() else "cpu")

        with open(inp) as fh:
            cfg = json.load(fh)
        constant_volume = cfg.get('constant_volume', False)

        init_conf = read(struct)
        # Rescale MP-queried structures (typically small primitive/
        # conventional cells) to a reasonably-sized supercell before
        # thermalizing -- a confirmed-safe no-op for structures that are
        # already supercell-sized (e.g. tier-2/3/4's own already-rescaled
        # starting points).
        init_conf = balanced_supercell(init_conf, max_atoms=cfg.get('supercell_max_atoms', 100))
        init_conf.set_calculator(calculator)

        step_counter = count()
        trajectory_path = os.path.join(output, 'md_run.traj')
        if os.path.exists(trajectory_path):
            os.remove(trajectory_path)

        dt = cfg['timestep'] * units.fs
        friction = cfg['friction'] / units.fs

        property_dict = {}
        max_deviation_per_atom = cfg.get('max_energy_deviation_per_atom', 5.0)
        reference_energy_holder = {}  # mutable cell so update_status sees each cycle's latest reference

        def update_status():
            step = next(step_counter) * cfg['frequency']
            energy = init_conf.get_potential_energy()
            forces = init_conf.get_forces()
            property_dict[step] = {
                'energy': float(energy),
                'fx': [float(f[0]) for f in forces],
                'fy': [float(f[1]) for f in forces],
                'fz': [float(f[2]) for f in forces],
            }

            reference_energy = reference_energy_holder.get('value')
            if reference_energy is not None:
                check_energy_stability(energy, reference_energy, len(init_conf), max_deviation_per_atom)

        aborted, abort_reason = False, None

        for i in range(cfg.get("cycles", 1)):

            # Cell minimization -- constant_volume per this task's own
            # config (see this module's own docstring for why this matters
            # for any tier that deliberately holds a rescaled volume fixed).
            fcf = FrechetCellFilter(init_conf, constant_volume=constant_volume)
            opt = FIRE(fcf)
            opt.run(fmax=cfg['fmax'], steps=cfg['optsteps'])
            # Reference resets after every minimization -- a defect
            # mutation legitimately changes atom count/composition (and so
            # the energy scale) between cycles, so the stability check must
            # compare against THIS cycle's own relaxed baseline.
            reference_energy_holder['value'] = init_conf.get_potential_energy()

            lt = temperature_scheduler(cfg['low_temperature'])
            try:
                run_dynamics(init_conf, dt, lt, friction, cfg['frequency'], cfg['nsteps'], update_status, trajectory_path)
            except MDInstabilityError as e:
                aborted, abort_reason = True, str(e)
                break

            fcf = FrechetCellFilter(init_conf, constant_volume=constant_volume)
            opt = FIRE(fcf)
            opt.run(fmax=cfg['fmax'], steps=cfg['optsteps'])
            reference_energy_holder['value'] = init_conf.get_potential_energy()

            ht = temperature_scheduler(cfg['high_temperature'])
            MaxwellBoltzmannDistribution(init_conf, temperature_K=ht)
            try:
                run_dynamics(init_conf, dt, ht, friction, cfg['frequency'], cfg['nsteps'], update_status, trajectory_path)
            except MDInstabilityError as e:
                aborted, abort_reason = True, str(e)
                break

            init_conf = generate_defect(init_conf, defect_prob=cfg.get("defect_prob", 1.0))
            init_conf.set_calculator(calculator)

        if aborted:
            print(f"WARNING: {output}: MD aborted early -- {abort_reason}", file=sys.stderr)

        # Writing properties.json even on abort (rather than skipping it)
        # matters: finished_file-based skip logic only checks for this
        # file's existence, not its content, so leaving it unwritten would
        # just get this same doomed structure resubmitted on every future
        # rerun.
        record = dict(property_dict)
        record['_status'] = 'aborted_unstable' if aborted else 'complete'
        if abort_reason:
            record['_abort_reason'] = abort_reason
        with open(os.path.join(output, 'properties.json'), "w") as f:
            json.dump(record, f, indent=4)

        if torch.cuda.is_available():
            del calculator, init_conf
            gc.collect()
            torch.cuda.empty_cache()

    return {"status": "complete"}


if __name__ == "__main__":
    run_finite_temperature_md(parse_list(sys.argv[1]), parse_list(sys.argv[2]),
                               parse_list(sys.argv[3]), parse_list(sys.argv[4]))
