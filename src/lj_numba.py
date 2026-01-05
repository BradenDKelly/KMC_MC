"""Numba-accelerated kernels for Lennard-Jones calculations.

These are the primary implementations used in performance-critical code paths.
They maintain the same physics as the reference Python implementations in src/lj.py.

Note: This module requires numba to be installed. Functions will raise ImportError
if called without numba. Use backend.require_numba() to check availability before use.
"""

import numpy as np
from .backend import njit, NUMBA_AVAILABLE

if not NUMBA_AVAILABLE:
    # If numba is not available, create stub functions that raise on call
    # This allows the module to be imported, but functions fail clearly when used
    def _raise_numba_error():
        raise ImportError(
            "Numba is required for performance kernels. "
            "Install with: pip install numba"
        )
    
    def total_energy_numba(*args, **kwargs):
        _raise_numba_error()
    
    def virial_pressure_numba(*args, **kwargs):
        _raise_numba_error()
    
    def delta_energy_particle_move_numba(*args, **kwargs):
        _raise_numba_error()
else:
    # Numba is available - define the actual accelerated functions
    @njit(cache=True)
    def lj_shifted_energy_numba(r2, rc2):
        """Numba-accelerated LJ shifted energy."""
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

    @njit(cache=True)
    def minimum_image_scalar(dr, L):
        """Minimum image convention for a single vector (scalar operations)."""
        half_L = 0.5 * L
        dr = dr - L * np.floor((dr + half_L) / L)
        return dr

    @njit(cache=True)
    def total_energy_numba(x, L, rc):
        """Numba-accelerated total energy calculation.
        
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
        
        for i in range(N - 1):
            for j in range(i + 1, N):
                dr = x[j] - x[i]
                dr = minimum_image_scalar(dr, L)
                r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
                if r2 < rc2 and r2 > 0:
                    U += lj_shifted_energy_numba(r2, rc2)
        
        return U

    @njit(cache=True)
    def lj_shifted_force_magnitude_numba(r2, rc2):
        """Numba-accelerated force magnitude."""
        r = np.sqrt(r2)
        if r <= 0:
            return 0.0
        
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        
        F = (48.0 / r) * (inv_r12 - 0.5 * inv_r6)
        return F

    @njit(cache=True)
    def virial_pressure_numba(x, L, rc, T):
        """Numba-accelerated virial pressure.
        
        Args:
            x: Particle positions, shape (N, 3)
            L: Box length
            rc: Cutoff distance
            T: Temperature
            
        Returns:
            Pressure in reduced units
        """
        N = x.shape[0]
        V = L**3
        rho = N / V
        rc2 = rc * rc
        
        # Ideal gas contribution
        P_ideal = rho * T
        
        # Virial contribution: W = sum_{i<j} r_ij * F(r_ij)
        W = 0.0
        for i in range(N - 1):
            for j in range(i + 1, N):
                dr = x[j] - x[i]
                dr = minimum_image_scalar(dr, L)
                r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
                if r2 < rc2 and r2 > 0:
                    r = np.sqrt(r2)
                    F = lj_shifted_force_magnitude_numba(r2, rc2)
                    W += r * F
        
        # Pressure = ideal + W/(3V)
        P_virial = W / (3.0 * V)
        P_total = P_ideal + P_virial
        
        return P_total

    @njit(cache=True)
    def delta_energy_particle_move_numba(positions, i, new_pos, L, rc2):
        """Numba-accelerated local energy change for moving particle i.
        
        Args:
            positions: All particle positions, shape (N, 3)
            i: Index of particle to move
            new_pos: New position of particle i, shape (3,)
            L: Box length
            rc2: Squared cutoff distance
            
        Returns:
            Energy change: U(new) - U(old)
        """
        N = positions.shape[0]
        old_pos = positions[i]
        dU = 0.0
        
        for j in range(N):
            if j == i:
                continue
            # old
            dr_old = old_pos - positions[j]
            dr_old = minimum_image_scalar(dr_old, L)
            r2_old = dr_old[0]*dr_old[0] + dr_old[1]*dr_old[1] + dr_old[2]*dr_old[2]
            if r2_old < rc2:
                dU -= lj_shifted_energy_numba(r2_old, rc2)
            # new
            dr_new = new_pos - positions[j]
            dr_new = minimum_image_scalar(dr_new, L)
            r2_new = dr_new[0]*dr_new[0] + dr_new[1]*dr_new[1] + dr_new[2]*dr_new[2]
            if r2_new < rc2:
                dU += lj_shifted_energy_numba(r2_new, rc2)
        
        return dU
