"""KMC_MC: Monte Carlo and Kinetic Monte Carlo simulations for Lennard-Jones fluids."""

from .backend import NUMBA_AVAILABLE, require_numba
from .utils import minimum_image, init_lattice
from .lj import (
    lj_shifted_energy,
    lj_force_shifted_energy,
    lj_force_shifted_force_magnitude,
    total_energy,
    delta_energy_particle_move,
    virial_pressure,
)
from .mc import run_metropolis_mc
from .kmc import run_equilibrium_kmc
from .rigid import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_rotate_vector,
    uniform_random_orientation,
    apply_rigid_transform,
    rigid_body_move_proposal,
)
from .molecule import RigidMolecule
from .water import make_spce_water
from .ewald import (
    EwaldParams,
    EwaldCache,
    ewald_energy_total,
    build_cache,
    delta_energy_move,
    apply_move,
)
from .waterbox import (
    WaterBox,
    water_sites_lab,
    flatten_sites,
    oxygen_positions,
    total_energy_waterbox,
    delta_energy_rigid_move,
    apply_rigid_move,
)
from .lj_kmc import (
    LJKMCrates,
    compute_relocation_rates,
    sample_event,
    apply_relocation,
    compute_widom_weights,
)
from .eos import a_res, Z, P, u_res, mu_res

__all__ = [
    "minimum_image",
    "init_lattice",
    "lj_shifted_energy",
    "total_energy",
    "delta_energy_particle_move",
    "virial_pressure",
    "run_metropolis_mc",
    "run_equilibrium_kmc",
    "quaternion_normalize",
    "quaternion_multiply",
    "quaternion_conjugate",
    "quaternion_rotate_vector",
    "uniform_random_orientation",
    "apply_rigid_transform",
    "rigid_body_move_proposal",
    "RigidMolecule",
    "make_spce_water",
    "EwaldParams",
    "EwaldCache",
    "ewald_energy_total",
    "build_cache",
    "delta_energy_move",
    "apply_move",
    "WaterBox",
    "water_sites_lab",
    "flatten_sites",
    "oxygen_positions",
    "total_energy_waterbox",
    "delta_energy_rigid_move",
    "apply_rigid_move",
    "LJKMCrates",
    "compute_relocation_rates",
    "sample_event",
    "apply_relocation",
    "compute_widom_weights",
    "a_res",
    "Z",
    "P",
    "u_res",
    "mu_res",
]

