"""
Run a single-core ASE molecular-dynamics trajectory using pyace's own ASE
Calculator (PyACECalculator), driven by MDMatEnsemble.run_individual's
dynamic-dispatch contract -- entry_point(ffield_list, structure_list,
output_list, in_file_list), one MD run per zipped tuple (see base.py's
MDMatEnsemble.run_individual docstring). This supersedes ase_mace.py's own
CLI-style (sys.argv) convention, which predates that dispatch contract --
see EnsembleFFFit/CLAUDE.md's own note flagging ase_mace.py as stale/
unexercised against the current Pipeline/chore pattern.

Deliberately CPU-only, unlike ase_mace.py's torch.cuda-conditional device
selection: PyACECalculator has no GPU/device argument at all -- pyace's own
C++ evaluator is CPU-only by construction. Also confirmed (2026-09-17, via
`ldd` on pyace/calculator*.so) that evaluator has no OpenMP/pthread linkage
either, so a single MD run here is bound to ONE core's worth of compute
regardless of how many cores a chore is given -- cores_per_task for this
driver mainly buys concurrent chores (e.g. many ensemble members' MD runs
in parallel), not a faster individual trajectory. Genuine multi-core
speedup on one large-supercell trajectory needs LAMMPS's own ML-PACE
pair_style (real MPI domain decomposition) -- not currently built into
this project's Pathfinder container (see pipeline/FRICTION_LOG.md for both
findings). pyace's GPU path (`tensorpotential`/TensorFlow) is separately
already confirmed incompatible with this project's Python 3.12 containers.

`in_file` (per task) is a YAML recipe with:
  ensemble        'nvt' (ase.md.langevin.Langevin, default) or 'nve'
                  (ase.md.verlet.VelocityVerlet)
  temperature_K   initial Maxwell-Boltzmann draw + Langevin target (nvt
                  only; ignored for nve)
  friction        Langevin friction coefficient, ASE's own units (ignored
                  for nve)
  timestep_fs     MD timestep in femtoseconds
  nsteps          total steps to run
  traj_interval   steps between trajectory frames / property-history entries
  seed            MaxwellBoltzmannDistribution's own rng seed, for
                  reproducible initial velocities (optional; omit for
                  numpy's own default global rng)
Missing/omitted keys fall back to DEFAULT_RECIPE below, same convention as
build_ace_ensemble_inputs' DEFAULT_*_CONFIG dicts.
"""
import json
import os

import numpy as np
import yaml
from ase.constraints import FixCom
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
import ase.units as units

DEFAULT_RECIPE = {
    "ensemble": "nvt",
    "temperature_K": 300.0,
    "friction": 0.02,
    "timestep_fs": 1.0,
    "nsteps": 2000,
    "traj_interval": 100,
    "seed": None,
}


def run_ace_md(ffield_list, structure_list, output_list, in_file_list):
    """
    One Langevin (nvt) or VelocityVerlet (nve) MD trajectory per zipped
    (ffield, structure, output, in_file) tuple -- see module docstring for
    the recipe format and the CPU-only/single-core rationale. Returns a
    list of per-task result dicts, same convention as ace_fit.py's
    run_ace_fit.
    """
    from pyace.asecalc import PyACECalculator

    results = []
    for ff, struct, output, in_file in zip(ffield_list, structure_list, output_list, in_file_list):
        recipe = dict(DEFAULT_RECIPE)
        if in_file:
            with open(in_file) as f:
                recipe.update(yaml.safe_load(f) or {})

        os.makedirs(output, exist_ok=True)

        atoms = read(struct)
        atoms.calc = PyACECalculator(ff)

        # rng expects a numpy Generator (has .normal()/.random()), not a
        # bare int/None -- confirmed via ASE 3.28's own
        # MaxwellBoltzmannDistribution signature before writing this.
        rng = np.random.default_rng(recipe["seed"]) if recipe["seed"] is not None else None
        MaxwellBoltzmannDistribution(atoms, temperature_K=recipe["temperature_K"], rng=rng)

        timestep = recipe["timestep_fs"] * units.fs
        if recipe["ensemble"] == "nve":
            dyn = VelocityVerlet(atoms, timestep)
        else:
            # fixcm=False + an explicit FixCom constraint, not Langevin's own
            # fixcm=True default -- ASE 3.28 deprecated fixcm=True (it doesn't
            # strictly sample the correct NVT distribution); FixCom is its
            # documented replacement, same net effect (remove center-of-mass
            # drift) without the FutureWarning on every run.
            atoms.set_constraint(FixCom())
            dyn = Langevin(atoms, timestep, temperature_K=recipe["temperature_K"],
                            friction=recipe["friction"], fixcm=False)

        property_history = {}

        def record_step(dyn=dyn, atoms=atoms, property_history=property_history):
            step = dyn.get_number_of_steps()
            forces = atoms.get_forces()
            property_history[step] = {
                "energy": float(atoms.get_potential_energy()),
                "temperature_K": float(atoms.get_temperature()),
                "max_force": float(np.abs(forces).max()),
            }

        dyn.attach(record_step, interval=recipe["traj_interval"])

        traj_path = os.path.join(output, "md_run.traj")
        traj = Trajectory(traj_path, "w", atoms)
        dyn.attach(traj.write, interval=recipe["traj_interval"])

        dyn.run(recipe["nsteps"])
        traj.close()

        with open(os.path.join(output, "properties.json"), "w") as f:
            json.dump(property_history, f, indent=2)

        results.append({
            "status": "complete",
            "output": output,
            "n_atoms": len(atoms),
            "nsteps": recipe["nsteps"],
            "final_energy": float(atoms.get_potential_energy()),
        })

    return results


if __name__ == "__main__":
    import sys

    from EnsembleFFFit.molecular_dynamics.helpers import parse_list

    run_ace_md(
        parse_list(sys.argv[1]), parse_list(sys.argv[2]),
        parse_list(sys.argv[3]), parse_list(sys.argv[4]),
    )
