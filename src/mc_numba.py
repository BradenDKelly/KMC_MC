"""Numba-accelerated MC sweep kernels for brute-force mode."""

import numpy as np
from .backend import njit, NUMBA_AVAILABLE
from .lj_numba import delta_energy_particle_move_numba

if not NUMBA_AVAILABLE:
    def _raise_numba_error():
        raise ImportError(
            "Numba is required for MC sweep kernels. "
            "Install with: pip install numba"
        )
    
    def mc_sweeps_bruteforce_numba(*args, **kwargs):
        _raise_numba_error()
else:
    @njit(cache=True)
    def mc_sweeps_bruteforce_numba(
        positions,      # (N, 3) float64, C-contig, mutated in-place
        L,              # float64
        rc2,            # float64, squared cutoff
        beta,           # float64, 1/T
        step,           # float64, max displacement
        displacements,  # (n_sweeps, N, 3) float64, pre-generated displacements
        uniforms,       # (n_sweeps, N) float64, pre-generated uniform [0,1) for accept/reject
        particle_indices,  # (n_sweeps, N) int64, pre-generated particle indices (0..N-1)
    ):
        """Execute n_sweeps MC sweeps in compiled code (brute-force mode).
        
        For each sweep s:
            For each particle i (in order given by particle_indices[s, :]):
                Propose new_pos = positions[i] + displacements[s, i] (wrapped PBC)
                Compute ΔU using brute-force Numba kernel
                Accept/reject using uniforms[s, i]
                If accepted: update positions[i] = new_pos
        
        Args:
            positions: Particle positions, shape (N, 3), modified in-place
            L: Box length
            rc2: Squared cutoff distance
            beta: Inverse temperature (1/T)
            step: Maximum displacement
            displacements: Pre-generated displacements, shape (n_sweeps, N, 3)
            uniforms: Pre-generated uniform random numbers [0,1), shape (n_sweeps, N)
            particle_indices: Pre-generated particle indices, shape (n_sweeps, N)
        
        Returns:
            (n_accept, n_attempt) where n_attempt = n_sweeps * N
        """
        N = positions.shape[0]
        n_sweeps = displacements.shape[0]
        n_accept = 0
        n_attempt = 0
        
        for s in range(n_sweeps):
            for idx in range(N):
                i = particle_indices[s, idx]
                n_attempt += 1
                
                # Propose new position
                disp = displacements[s, idx]
                new_pos = positions[i] + disp
                # Wrap PBC
                new_pos = new_pos % L
                
                # Compute energy change
                dU = delta_energy_particle_move_numba(positions, i, new_pos, L, rc2)
                
                # Accept/reject
                if dU <= 0.0:
                    # Always accept downhill moves
                    positions[i] = new_pos
                    n_accept += 1
                else:
                    # Metropolis criterion
                    if uniforms[s, idx] < np.exp(-beta * dU):
                        positions[i] = new_pos
                        n_accept += 1
        
        return n_accept, n_attempt

