import os
import numpy as np
from ase.calculators.calculator import FileIOCalculator
from ase.stress import full_3x3_to_voigt_6_stress
from pymatgen.io.ase import AseAtomsAdaptor

from EnsembleFFFit.density_functional_theory.rmg.rmg_input import RMGInput
from EnsembleFFFit.density_functional_theory.rmg.forcefield import Forcefield
from EnsembleFFFit.density_functional_theory.rmg.rmg_log import RMGLog
from EnsembleFFFit.density_functional_theory.rmg.convergence import RMGConvergence

HARTREE_TO_EV = 27.211386245988

# RMG's energy_output_units input key (default "Hartrees", or "Rydbergs")
# scales energy and per-ion forces by the identical factor -- confirmed in
# RMG's own source (RMG/Common/Quench.cpp and RMG/Common/write_force.cpp both
# apply ct.energy_output_conversion[ct.energy_output_units] to what they
# print) -- so both units 'Ha' and 'Ry' appear on RMGLog's parsed energy line,
# and both need to be handled rather than assuming Hartree.
UNIT_TO_EV = {'Ha': HARTREE_TO_EV, 'Ry': HARTREE_TO_EV / 2}


class RMG(FileIOCalculator):
    """
    ASE FileIOCalculator wrapping the compiled RMG DFT binary.

    write_input always regenerates rmg_name from the atoms it's given, rather
    than the usual FileIOCalculator idiom of skipping generation if the file
    already exists -- for a continuation run the final structure differs from
    whatever rmg_name already has on disk, so skipping would run RMG against a
    stale structure. Because the processor grid and node count depend on the
    lattice (via the wavefunction grid) and total valence electron count,
    write_input also recomputes them fresh from the current atoms via
    RMGInput.from_yaml(target_nodes=allocated_nodes) -- constraining the
    search to fit within what was actually reserved for this chore, matching
    pyRMG's original CLI usage (`target_nodes=args.nodes`) -- and compares the
    result against allocated_nodes. Under "Relax Structure" the lattice can
    change enough between steps to change that sizing, so a mismatch is a
    real problem -- this raises rather than silently running under/over-
    provisioned.

    Site properties: this only looks for 'selective_dynamics' and
    'magnetic_properties' on atoms.arrays (not on whatever pymatgen Structure
    produced the atoms) -- whatever builds the Atoms object passed in (e.g.
    the RMG_testing driver script, from EnsembleFFFit.density_functional_theory
    .rmg.pick_structure.pick_best_structure's returned Structure) is
    responsible for carrying those over explicitly with
    atoms.set_array(...), since round-tripping arbitrary pymatgen
    site_properties through AseAtomsAdaptor is not guaranteed.

    `command` defaults to a bare `{rmg_executable} {rmg_name}` invocation --
    deliberately no srun/mpirun/flux-run wrapper. Multi-task/multi-node
    launch is Flux's job: a MatEnsemble chore's own Resources
    (num_tasks/cores_per_task/gpus_per_task) already tell Flux how to place
    and launch the chore's process(es) across whatever was allocated, so
    wrapping the command here on top of that would double up on (or conflict
    with) launch semantics Flux already owns. This is genuinely untested
    against a real multi-node chore, though (every existing MACE/MD chore in
    this codebase used num_tasks=1) -- if the bare binary alone turns out not
    to be enough (e.g. RMG needs some other way to learn its rank/size inside
    a Flux-launched process), override `command` explicitly via a 'command'
    key in rmg_yaml, which test/RMG_testing/rmg_dft.py reads.
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, rmg_yaml, allocated_nodes, restart=None, label='rmg', atoms=None,
                 command=None, rmg_executable='rmg-gpu', rmg_name='rmg_input',
                 pseudopotentials_directory='', gpus_per_node=8, electrons_per_gpu=10,
                 grid_divisibility_exponent=3, directory='.', **kwargs):
        self.rmg_yaml = rmg_yaml
        self.allocated_nodes = allocated_nodes
        self.rmg_name = rmg_name
        self.pseudopotentials_directory = pseudopotentials_directory
        self.gpus_per_node = gpus_per_node
        self.electrons_per_gpu = electrons_per_gpu
        self.grid_divisibility_exponent = grid_divisibility_exponent
        self.target_nodes = None

        if command is None:
            command = f'{rmg_executable} {rmg_name}'

        FileIOCalculator.__init__(self, restart=restart, label=label, atoms=atoms,
                                   command=command, directory=directory, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        structure = AseAtomsAdaptor.get_structure(atoms)
        selective_dynamics = atoms.arrays.get('selective_dynamics')
        magnetic_properties = atoms.arrays.get('magnetic_properties')
        if selective_dynamics is not None:
            structure.add_site_property('selective_dynamics', list(selective_dynamics))
        if magnetic_properties is not None:
            structure.add_site_property('magnetic_properties', list(magnetic_properties))

        rmg_input = RMGInput.from_yaml(
            yaml_path=self.rmg_yaml,
            structure_path=None,
            structure_obj=structure,
            pseudopotentials_directory=self.pseudopotentials_directory,
            magmom_path=None,
            target_nodes=self.allocated_nodes,
            gpus_per_node=self.gpus_per_node,
            electrons_per_gpu=self.electrons_per_gpu,
            grid_divisibility_exponent=self.grid_divisibility_exponent,
        )

        if rmg_input.target_nodes != self.allocated_nodes:
            raise ValueError(
                f"Recomputed node requirement ({rmg_input.target_nodes}) for the current structure "
                f"does not match the resources allocated to this chore ({self.allocated_nodes}) in "
                f"{self.directory}. This can happen if the lattice changed enough (e.g. under cell "
                f"relaxation) to change the processor grid sizing -- resubmit this task with "
                f"allocated_nodes={rmg_input.target_nodes}, or pin 'processor_grid' explicitly in "
                f"{self.rmg_yaml} to fix it regardless of structure."
            )

        self.target_nodes = rmg_input.target_nodes
        rmg_input.save(os.path.join(self.directory, self.rmg_name))

    def read_results(self):
        forcefield_path = os.path.join(self.directory, 'forcefield.xml')
        forcefield = Forcefield(forcefield_xml_path=forcefield_path)
        rmg_input = RMGInput(input_file=os.path.join(self.directory, self.rmg_name))
        convergence = RMGConvergence(rmg_input=rmg_input, forcefield=forcefield)
        if not convergence.is_converged():
            raise RuntimeError(f"RMG run in {self.directory} did not converge ({convergence.calculation_mode}).")

        rmg_logs = RMGLog(self.directory)
        log_files = sorted(rmg_logs.logs_data.keys(), reverse=True)
        if not log_files:
            raise RuntimeError(f"No {self.rmg_name}.*.log found in {self.directory} to read energy/forces from.")

        latest = rmg_logs.logs_data[log_files[0]]
        if not latest['energies'] or not latest['forces']:
            raise RuntimeError(f"Latest RMG log {log_files[0]} in {self.directory} has no parsed energy/forces.")

        energy_unit = latest.get('energy_unit')
        if energy_unit not in UNIT_TO_EV:
            raise ValueError(
                f"Unrecognized or missing energy unit ({energy_unit!r}) parsed from {log_files[0]} in "
                f"{self.directory} -- expected 'Ha' or 'Ry' (RMG's energy_output_units setting)."
            )
        conversion = UNIT_TO_EV[energy_unit]

        self.results['energy'] = latest['energies'][-1] * conversion
        self.results['forces'] = np.array(latest['forces'][-1]) * conversion

        # RMGLog already converts kbar -> eV/Angstrom^3 (RMG's own energy-unit
        # scaling doesn't apply to stress -- it's printed directly in kbar
        # regardless of energy_output_units), so no `conversion` factor here.
        # Stored as ASE's standard Voigt-6 (not the full 3x3 RMGLog returns) to
        # match what atoms.get_stress()/other generic ASE tooling expects from
        # calc.results['stress']; full-3x3 is reconstructed only where something
        # downstream (e.g. extxyz writing) specifically needs that shape.
        stresses = latest.get('stresses')
        if stresses:
            self.results['stress'] = full_3x3_to_voigt_6_stress(np.array(stresses[-1]))
