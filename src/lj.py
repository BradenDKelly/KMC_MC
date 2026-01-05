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


def lj_force_shifted_energy(r2, rc2):
    """Compute Lennard-Jones energy with force shift (energy and force go to zero at rc).
    
    u_FS(r) = u_LJ(r) - u_LJ(rc) - (r - rc) * u_LJ'(rc)   for r < rc
    u_FS(r) = 0                                            for r >= rc
    
    where u_LJ(r) = 4 * (r^-12 - r^-6)
    and u_LJ'(r) = 24*r^-7 - 48*r^-13
    
    Args:
        r2: Squared distance(s), scalar or array
        rc2: Squared cutoff distance
        
    Returns:
        Force-shifted LJ energy
    """
    if isinstance(r2, np.ndarray):
        result = np.zeros_like(r2)
        mask = r2 < rc2
        if not np.any(mask):
            return result
        
        r = np.sqrt(r2[mask])
        rc = np.sqrt(rc2)
        
        # Unshifted LJ energy: u_LJ(r) = 4 * (r^-12 - r^-6)
        inv_r2 = 1.0 / r2[mask]
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        u_LJ_r = 4.0 * (inv_r12 - inv_r6)
        
        # Unshifted LJ energy at rc: u_LJ(rc)
        inv_rc2 = 1.0 / rc2
        inv_rc6 = inv_rc2**3
        inv_rc12 = inv_rc6**2
        u_LJ_rc = 4.0 * (inv_rc12 - inv_rc6)
        
        # Derivative of u_LJ at rc: u_LJ'(rc) = 24*rc^-7 - 48*rc^-13
        u_LJ_prime_rc = 24.0 * (inv_rc6 / rc) - 48.0 * (inv_rc12 / rc)
        
        # Force-shifted energy
        result[mask] = u_LJ_r - u_LJ_rc - (r - rc) * u_LJ_prime_rc
        return result
    else:
        # Scalar case
        if r2 >= rc2:
            return 0.0
        
        r = np.sqrt(r2)
        rc = np.sqrt(rc2)
        
        # Unshifted LJ energy: u_LJ(r) = 4 * (r^-12 - r^-6)
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        u_LJ_r = 4.0 * (inv_r12 - inv_r6)
        
        # Unshifted LJ energy at rc: u_LJ(rc)
        inv_rc2 = 1.0 / rc2
        inv_rc6 = inv_rc2**3
        inv_rc12 = inv_rc6**2
        u_LJ_rc = 4.0 * (inv_rc12 - inv_rc6)
        
        # Derivative of u_LJ at rc: u_LJ'(rc) = 24*rc^-7 - 48*rc^-13
        u_LJ_prime_rc = 24.0 * (inv_rc6 / rc) - 48.0 * (inv_rc12 / rc)
        
        # Force-shifted energy
        u_FS = u_LJ_r - u_LJ_rc - (r - rc) * u_LJ_prime_rc
        
        return u_FS


def lj_force_shifted_force_magnitude(r2, rc2):
    """Compute the magnitude of the force for force-shifted Lennard-Jones.
    F(r) = -dU_FS/dr
    
    For force-shifted: u_FS(r) = u_LJ(r) - u_LJ(rc) - (r - rc) * u_LJ'(rc)
    dU_FS/dr = u_LJ'(r) - u_LJ'(rc)
    F = -dU_FS/dr = u_LJ'(rc) - u_LJ'(r)
    
    where u_LJ'(r) = 24*r^-7 - 48*r^-13
    
    Args:
        r2: Squared distance(s)
        rc2: Squared cutoff distance
        
    Returns:
        Force magnitude
    """
    if isinstance(r2, np.ndarray):
        result = np.zeros_like(r2)
        mask = (r2 > 0) & (r2 < rc2)
        if not np.any(mask):
            return result
        
        r = np.sqrt(r2[mask])
        rc = np.sqrt(rc2)
        
        # u_LJ'(r) = 24*r^-7 - 48*r^-13
        inv_r2 = 1.0 / r2[mask]
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        u_LJ_prime_r = 24.0 * (inv_r6 / r) - 48.0 * (inv_r12 / r)
        
        # u_LJ'(rc) = 24*rc^-7 - 48*rc^-13
        inv_rc2 = 1.0 / rc2
        inv_rc6 = inv_rc2**3
        inv_rc12 = inv_rc6**2
        u_LJ_prime_rc = 24.0 * (inv_rc6 / rc) - 48.0 * (inv_rc12 / rc)
        
        # Force = -dU_FS/dr = u_LJ'(rc) - u_LJ'(r)
        result[mask] = u_LJ_prime_rc - u_LJ_prime_r
        return result
    else:
        # Scalar case
        if r2 == 0 or r2 >= rc2:
            return 0.0
        
        r = np.sqrt(r2)
        rc = np.sqrt(rc2)
        
        # u_LJ'(r) = 24*r^-7 - 48*r^-13
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        u_LJ_prime_r = 24.0 * (inv_r6 / r) - 48.0 * (inv_r12 / r)
        
        # u_LJ'(rc) = 24*rc^-7 - 48*rc^-13
        inv_rc2 = 1.0 / rc2
        inv_rc6 = inv_rc2**3
        inv_rc12 = inv_rc6**2
        u_LJ_prime_rc = 24.0 * (inv_rc6 / rc) - 48.0 * (inv_rc12 / rc)
        
        # Force = -dU_FS/dr = u_LJ'(rc) - u_LJ'(r)
        F = u_LJ_prime_rc - u_LJ_prime_r
        
        return F


def lj_shifted_force_magnitude(r2, rc2):
    """Compute magnitude of LJTS force: -dU/dr.
    
    For LJTS: U(r) = 4*((1/r)^12 - (1/r)^6) - shift, where shift makes U(rc)=0.
    Force magnitude: F(r) = -dU/dr = (48/r) * ((1/r)^12 - 0.5*(1/r)^6)
    
    Note: The shift is constant, so it doesn't contribute to the force.
    
    Args:
        r2: Squared distance
        rc2: Squared cutoff distance
        
    Returns:
        Force magnitude (in reduced units where ε=σ=1)
    """
    r = np.sqrt(r2)
    if r <= 0:
        return 0.0
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    
    # Force magnitude: F = (48/r) * (r^-12 - 0.5*r^-6)
    F = (48.0 / r) * (inv_r12 - 0.5 * inv_r6)
    
    return F


def virial_pressure(x, L, rc, T):
    """Compute instantaneous virial pressure for LJTS.
    
    Uses the virial pressure formula:
    P = ρkT + (1/(3V)) * sum_{i<j} r_ij · F_ij
    
    For central forces with pairwise interactions:
    - r_ij = r_j - r_i (minimum image convention)
    - F_ij is the force on particle i due to particle j
    - For repulsive interactions, F_ij points from j to i
    - r_ij · F_ij = -r * F where r = |r_ij| and F is the force magnitude
    
    Therefore: P = ρkT - (1/(3V)) * sum(-r*F) = ρkT + (1/(3V)) * sum(r*F)
    
    This function computes W = sum_{i<j} (r * F) where:
    - Sum is over pairs i<j (no double counting)
    - r is the distance between particles (minimum image)
    - F is the force magnitude from lj_shifted_force_magnitude (positive for repulsive)
    - Only pairs with r < rc are included
    
    Then: P = ρkT + W/(3V)
    
    Args:
        x: Particle positions, shape (N, 3)
        L: Box length
        rc: Cutoff distance
        T: Temperature (reduced units)
        
    Returns:
        Pressure P in reduced units (P* = P*σ³/ε).
        To get compressibility factor: Z = P / (ρ*kT) = P / (ρ*T)
    """
    N = x.shape[0]
    V = L**3
    rho = N / V
    rc2 = rc * rc
    
    # Ideal gas contribution
    P_ideal = rho * T
    
    # Virial contribution: W = -1/3 * sum_{i<j} r_ij * F(r_ij)
    W = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            dr = minimum_image(x[j] - x[i], L)
            r2 = np.dot(dr, dr)
            if r2 < rc2 and r2 > 0:
                r = np.sqrt(r2)
                F = lj_shifted_force_magnitude(r2, rc2)
                # Virial contribution: r_ij * F(r_ij) (scalar product for central forces)
                W += r * F
    
    # Standard virial pressure formula:
    # P = ρkT - (1/(3V)) * sum_{i<j} r_ij · F_ij
    # For central forces, F_ij is the force on i due to j, pointing from j to i.
    # r_ij = r_j - r_i points from i to j.
    # For repulsive interactions, F_ij points from j to i, so r_ij · F_ij = -r * F
    # where F = |F_ij| is the force magnitude (positive for repulsive).
    # Therefore: P = ρkT - (1/(3V)) * sum(-r*F) = ρkT + (1/(3V)) * sum(r*F)
    # Since lj_shifted_force_magnitude returns F = |dU/dr| (positive for repulsive),
    # and we compute W = sum(r*F), the virial contribution is:
    P_virial = W / (3.0 * V)
    P_total = P_ideal + P_virial
    
    return P_total


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

