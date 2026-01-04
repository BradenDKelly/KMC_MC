"""Lennard-Jones energy calculations: pairwise energy, total energy, and delta energy."""

import numpy as np
from .utils import minimum_image


def lj_shifted_energy(r2, rc2):
    """Compute Lennard-Jones energy with shift to ensure u(rc) = 0.
    
    Args:
        r2: Squared distance(s), scalar or array
        rc2: Squared cutoff distance
        
    Returns:
        Shifted LJ energy: 4 * (r^-12 - r^-6) - shift, where shift makes u(rc) = 0
    """
    # LJ + shift at rc so u(rc)=0
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    u = 4.0 * (inv_r12 - inv_r6)
    # shift
    rc = np.sqrt(rc2)
    inv_rc2 = 1.0 / (rc * rc)
    inv_rc6 = inv_rc2**3
    inv_rc12 = inv_rc6**2
    u_shift = 4.0 * (inv_rc12 - inv_rc6)
    return u - u_shift


def total_energy(x, L, rc):
    """Compute total potential energy of the system.
    
    Args:
        x: Particle positions, shape (N, 3)
        L: Box length
        rc: Cutoff distance
        
    Returns:
        Total potential energy
    """
    N = x.shape[0]
    rc2 = rc * rc
    U = 0.0
    for i in range(N-1):
        dr = minimum_image(x[i+1:] - x[i], L)
        r2 = np.sum(dr*dr, axis=1)
        mask = r2 < rc2
        if np.any(mask):
            U += np.sum(lj_shifted_energy(r2[mask], rc2))
    return U


def delta_energy_particle_move(i, new_xi, x, L, rc):
    """Compute energy change for moving particle i to new_xi.
    
    Args:
        i: Index of particle to move
        new_xi: New position of particle i, shape (3,)
        x: Current particle positions, shape (N, 3)
        L: Box length
        rc: Cutoff distance
        
    Returns:
        Energy change: U(new) - U(old)
    """
    N = x.shape[0]
    rc2 = rc * rc
    old_xi = x[i]
    dU = 0.0
    for j in range(N):
        if j == i:
            continue
        # old
        dr_old = minimum_image(old_xi - x[j], L)
        r2_old = np.dot(dr_old, dr_old)
        if r2_old < rc2:
            dU -= lj_shifted_energy(r2_old, rc2)
        # new
        dr_new = minimum_image(new_xi - x[j], L)
        r2_new = np.dot(dr_new, dr_new)
        if r2_new < rc2:
            dU += lj_shifted_energy(r2_new, rc2)
    return dU

