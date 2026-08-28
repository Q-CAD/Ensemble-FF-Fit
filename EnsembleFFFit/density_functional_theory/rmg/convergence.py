from EnsembleFFFit.density_functional_theory.rmg.forcefield import Forcefield
from EnsembleFFFit.density_functional_theory.rmg.rmg_input import RMGInput


class RMGConvergence:
    def __init__(self, rmg_input: RMGInput, forcefield: Forcefield):
        self.rmg_input = rmg_input
        self.calculation_mode = rmg_input.keywords['calculation_mode']
        self.forcefield = forcefield

    def is_converged(self):
        if self.calculation_mode == "Relax Structure":
            return (self.forcefield.force and self.forcefield.scf)
        elif self.calculation_mode == "Quench Electrons":
            return self.forcefield.scf
        else:
            raise ValueError(f'Convergence checking not currently supported for {self.calculation_mode}')
