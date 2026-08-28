import os
import re
import glob
import numpy as np
from pymatgen.core import Structure

ENERGY_LINE_PATTERN = re.compile(r'final total energy from eig sum\s*=\s*([\-\d\.]+)\s*(\S+)')


class RMGLog:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.logs_data = self._parse_logs()
        self.logs_keys = list(self.logs_data.keys())

    def _parse_logs(self):
        log_files = glob.glob(os.path.join(self.directory_path, 'rmg_input.*.log'))

        logs_data = {}
        bohr_factor = 1.8897259886  # Bohr per Angstrom -- divide a Bohr-based quantity by this to get it per Angstrom

        for log_file in sorted(log_files):
            structures, forces, energies = [], [], []
            all_lattices, all_positions, all_species, all_forces = [], [], [], []
            current_lattice, current_position, current_specie, current_force = [], [], [], []
            energy_unit = None
            # "stress total   in unit of kbar" header, followed by exactly 3 rows
            # of 3 floats each (full 3x3 tensor, not Voigt-6) -- confirmed against
            # rmgdft/tests/RMG/graphite_stress/input.ref.log, where it prints once
            # per ionic/quench step, same frequency as energies/forces, so pairing
            # by occurrence order (like the @ION blocks above) keeps them aligned.
            all_stresses, current_stress, stress_rows_remaining = [], [], 0

            def flush_ion_block():
                # A real @ION block is many consecutive lines (one per atom).
                # This is called when a new block's header line arrives (so the
                # *previous* block, if any, gets recorded before starting the
                # new one) and once more at EOF for whichever block was last.
                if current_position:
                    all_positions.append(current_position)
                    all_species.append(current_specie)
                    all_forces.append(current_force)

            with open(log_file, 'r') as f:
                for line in f:
                    if "X Basis Vector" in line or "Y Basis Vector" in line or "Z Basis Vector" in line:
                        split_lines = line.split()
                        try:
                            current_lattice.append([float(split_lines[3]), float(split_lines[4]), float(split_lines[5])])
                            if len(current_lattice) == 3:
                                all_lattices.append(current_lattice)
                                current_lattice = []
                        except IndexError:
                            continue

                    elif "lattice" in line:
                        split_lines = line.split()
                        try:
                            current_lattice.append([float(split_lines[2]), float(split_lines[3]), float(split_lines[4])])
                            if len(current_lattice) == 3:
                                all_lattices.append(current_lattice)
                                current_lattice = []
                        except IndexError:
                            continue

                    elif "@ION" in line:
                        split_lines = line.split()
                        try:
                            int(split_lines[1])
                        except (ValueError, IndexError):
                            # Header line ("@ION  Ion  Species  X  Y  Z ...") --
                            # marks the start of a new per-step ion block. RMG
                            # prints a step's energy *before* that same step's
                            # per-ion forces, so pairing energies/positions/
                            # forces by occurrence order (not by which text
                            # comes first) is what keeps them aligned per step.
                            flush_ion_block()
                            current_position, current_specie, current_force = [], [], []
                        else:
                            try:
                                current_position.append([float(split_lines[3]), float(split_lines[4]), float(split_lines[5])])
                                current_specie.append(split_lines[2])
                                current_force.append([float(split_lines[7]), float(split_lines[8]), float(split_lines[9])])
                            except IndexError:
                                continue

                    if "stress total" in line and "unit of kbar" in line:
                        current_stress = []
                        stress_rows_remaining = 3
                    elif stress_rows_remaining:
                        split_lines = line.split()
                        try:
                            current_stress.append([float(v) for v in split_lines[:3]])
                        except (ValueError, IndexError):
                            stress_rows_remaining = 0
                            current_stress = []
                        else:
                            stress_rows_remaining -= 1
                            if stress_rows_remaining == 0:
                                all_stresses.append(current_stress)
                                current_stress = []

                    energy_match = ENERGY_LINE_PATTERN.search(line)
                    if energy_match:
                        energies.append(float(energy_match.group(1)))
                        energy_unit = energy_match.group(2)

                # The loop above only flushes a block when the *next* one's
                # header arrives; the last block in the file needs this
                # explicit flush since no such header ever follows it.
                flush_ion_block()

            if len(all_lattices) == 1:
                check_lattices = [all_lattices[0] for _ in range(len(all_positions))]
            else:
                check_lattices = all_lattices

            number_complete = min(len(check_lattices), len(all_positions))
            energies = energies[:number_complete]

            kbar_per_ev_per_ang3 = 1602.176634  # 1 eV/Angstrom^3 = 1602.176634 kbar
            stresses = [
                (np.array(s) / kbar_per_ev_per_ang3).tolist()
                for s in all_stresses[:number_complete]
            ]

            for i in range(number_complete):
                lattice_angstroms = np.divide(np.array(check_lattices[i]), bohr_factor)
                lattice_positions = np.divide(np.array(all_positions[i]), bohr_factor)
                # Forces are printed in [energy_unit]/a0 (Bohr) -- RMG's
                # energy_output_units setting scales energy and per-ion forces
                # by the identical factor (confirmed in RMG/Common/Quench.cpp
                # and RMG/Common/write_force.cpp: both use the same
                # ct.energy_output_conversion[ct.energy_output_units]), so only
                # the Bohr->Angstrom distance part is converted here; the
                # energy-unit part is left as printed (Ha by default, or Ry)
                # and resolved to eV by the caller using `energy_unit` below,
                # the same way it resolves `energies`.
                force_per_angstrom = np.divide(np.array(all_forces[i]), bohr_factor)

                s = Structure(lattice=lattice_angstroms, species=all_species[i],
                              coords=lattice_positions, coords_are_cartesian=True)
                structures.append(s)
                forces.append(force_per_angstrom)

            logs_data[log_file] = {
                "structures": structures,
                "forces": forces,
                "energies": energies,
                "energy_unit": energy_unit,
                # eV/Angstrom^3, full 3x3 (not Voigt-6) -- empty list if this run's
                # yaml didn't set stress: True, so callers must handle that case.
                "stresses": stresses,
            }

        return logs_data

    def get_log_data(self, log_file=None):
        if log_file:
            return self.logs_data.get(log_file, None)
        return self.logs_data
