import lammps
import lammps.mliap
import torch
import sys
import os
from EnsembleFFFit.matensemble.lammps.helpers import parse_list
from EnsembleFFFit.matensemble.lammps.helpers import get_elements

if __name__ == "__main__":
    ff_list       = parse_list(sys.argv[1]) # Force field file paths
    input_list    = parse_list(sys.argv[2]) # Input file paths
    struct_list   = parse_list(sys.argv[3]) # Structure file paths
    output_list   = parse_list(sys.argv[4]) # LAMMPs output write paths

    n = len(ff_list)
    assert all(len(lst) == n for lst in (input_list, struct_list, output_list)), "All lists must be same length"

    lmp = lammps.lammps(cmdargs=['-k', 'on', 'g', '4', '-sf', 'kk', 
                      '-pk', 'kokkos', 'neigh', 'half', 
                      'newton', 'off', '-echo', 'both', 
                      "-log", "none", "-screen", "os.devnull"]) # Or similar command
    lammps.mliap.activate_mliappy_kokkos(lmp)

    for idx, (ff, inp, struct, output) in enumerate(zip(ff_list, input_list, struct_list, output_list), start=0):
        # 1a) Open a new log file in the output path
        lmp.command(f"log {os.path.join(output, 'log.lammps')}")

        # lb) Set the lammps dumpfile name
        lmp.command(f"variable dump_file string {os.path.join(output, 'dump_*.dump')}")

        # 2) Pass in your filenames
        if ff:
            lmp.command(f"variable ff_filename string {ff}")
        if struct:
            lmp.command(f"variable structure string {struct}")

        # 3) Determine the pair coefficient
        elements = get_elements(struct)
        lmp.command(f'variable elements string "{elements}"')

        # 4) Run the LAMMPS input
        lmp.file(inp)

        # 5) Clear for the next iteration
        lmp.command("clear")

    # Final cleanup
    lmp.close()
