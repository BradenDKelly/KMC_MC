"""Numba-accelerated neighbor list (cell list) implementation.

This module provides cell-list based neighbor lists for efficient O(N) pair iteration
instead of O(N^2) brute force. Used to accelerate energy and force calculations.

The cell list divides space into cells of size >= (rc + skin), allowing O(1) lookup
of potential neighbors for each particle.
"""

import numpy as np
from .backend import njit, NUMBA_AVAILABLE

if not NUMBA_AVAILABLE:
    def _raise_numba_error():
        raise ImportError(
            "Numba is required for neighbor list kernels. "
            "Install with: pip install numba"
        )
    
    def build_neighbor_list_numba(*args, **kwargs):
        _raise_numba_error()
    
    def total_energy_nl_numba(*args, **kwargs):
        _raise_numba_error()
    
    def virial_pressure_nl_numba(*args, **kwargs):
        _raise_numba_error()
    
    def delta_energy_particle_move_nl_numba(*args, **kwargs):
        _raise_numba_error()
else:
    @njit(cache=True)
    def minimum_image_scalar(dr, L):
        """Minimum image convention for a single vector."""
        half_L = 0.5 * L
        dr = dr - L * np.floor((dr + half_L) / L)
        return dr

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
    def build_neighbor_list_numba(positions, L, rc2):
        """Build neighbor list using cell list method.
        
        Uses a cell list to efficiently find all pairs within cutoff.
        Returns neighbor list in CSR (Compressed Sparse Row) format:
        - neighbor_list: Flat array of neighbor indices
        - neighbor_starts: Starting index in neighbor_list for each particle (length N+1)
        
        Args:
            positions: Particle positions, shape (N, 3), in [0, L)
            L: Box length
            rc2: Squared cutoff distance
            
        Returns:
            neighbor_list: Flat array of neighbor particle indices
            neighbor_starts: Starting indices for each particle, shape (N+1,)
        """
        N = positions.shape[0]
        rc = np.sqrt(rc2)
        
        # Build cell list
        # Cell size should be >= rc to ensure all neighbors are in adjacent cells
        cell_size = max(1, int(np.floor(L / rc)))
        if cell_size < 1:
            cell_size = 1
        n_cells = cell_size * cell_size * cell_size
        cell_width = L / cell_size
        
        # Assign particles to cells
        # cell_particles[cell_idx] will contain list of particle indices in that cell
        # Use a simple approach: count particles per cell, then assign
        cell_counts = np.zeros(n_cells, dtype=np.int32)
        cell_indices = np.zeros(N, dtype=np.int32)
        
        for i in range(N):
            cx = int(np.floor(positions[i, 0] / cell_width))
            cy = int(np.floor(positions[i, 1] / cell_width))
            cz = int(np.floor(positions[i, 2] / cell_width))
            cx = min(cx, cell_size - 1)
            cy = min(cy, cell_size - 1)
            cz = min(cz, cell_size - 1)
            cell_idx = cx + cy * cell_size + cz * cell_size * cell_size
            cell_indices[i] = cell_idx
            cell_counts[cell_idx] += 1
        
        # Build cell_starts (prefix sum)
        cell_starts = np.zeros(n_cells + 1, dtype=np.int32)
        for c in range(n_cells):
            cell_starts[c + 1] = cell_starts[c] + cell_counts[c]
        
        # Sort particles by cell
        sorted_particles = np.zeros(N, dtype=np.int32)
        cell_pos = np.zeros(n_cells, dtype=np.int32)
        for i in range(N):
            cell_idx = cell_indices[i]
            pos = cell_starts[cell_idx] + cell_pos[cell_idx]
            sorted_particles[pos] = i
            cell_pos[cell_idx] += 1
        
        # Count total neighbors (estimate: assume average density)
        # We'll build neighbor list directly
        # Use a conservative estimate: worst case is all pairs
        max_total_neighbors = N * (N - 1) // 2
        neighbor_list = np.zeros(max_total_neighbors, dtype=np.int32)
        neighbor_starts = np.zeros(N + 1, dtype=np.int32)
        neighbor_count = 0
        
        # Build neighbor list by iterating over cells
        for i in range(N):
            pos_i = positions[i]
            neighbor_starts[i] = neighbor_count
            
            # Get cell of particle i
            cx_i = int(np.floor(pos_i[0] / cell_width))
            cy_i = int(np.floor(pos_i[1] / cell_width))
            cz_i = int(np.floor(pos_i[2] / cell_width))
            cx_i = min(cx_i, cell_size - 1)
            cy_i = min(cy_i, cell_size - 1)
            cz_i = min(cz_i, cell_size - 1)
            
            # Check 27 neighboring cells (including self)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        cx = (cx_i + dx) % cell_size
                        cy = (cy_i + dy) % cell_size
                        cz = (cz_i + dz) % cell_size
                        if cx < 0:
                            cx += cell_size
                        if cy < 0:
                            cy += cell_size
                        if cz < 0:
                            cz += cell_size
                        
                        cell_idx = cx + cy * cell_size + cz * cell_size * cell_size
                        
                        # Iterate over particles in this cell
                        start = cell_starts[cell_idx]
                        end = cell_starts[cell_idx + 1]
                        
                        for idx in range(start, end):
                            j = sorted_particles[idx]
                            if j <= i:  # Only consider j > i to avoid double counting
                                continue
                            
                            # Check distance
                            dr = pos_i - positions[j]
                            dr = minimum_image_scalar(dr, L)
                            r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
                            
                            if r2 < rc2:
                                neighbor_list[neighbor_count] = j
                                neighbor_count += 1
        
        neighbor_starts[N] = neighbor_count
        
        # Compact neighbor_list to actual size
        neighbor_list_compact = np.zeros(neighbor_count, dtype=np.int32)
        for i in range(neighbor_count):
            neighbor_list_compact[i] = neighbor_list[i]
        
        return neighbor_list_compact, neighbor_starts

    @njit(cache=True)
    def total_energy_nl_numba(positions, neighbor_list, neighbor_starts, L, rc2):
        """Compute total energy using neighbor list.
        
        Args:
            positions: Particle positions, shape (N, 3)
            neighbor_list: Flat array of neighbor indices
            neighbor_starts: Starting index for each particle, shape (N+1,)
            L: Box length
            rc2: Squared cutoff distance
            
        Returns:
            Total potential energy
        """
        N = positions.shape[0]
        U = 0.0
        
        for i in range(N):
            pos_i = positions[i]
            start = neighbor_starts[i]
            end = neighbor_starts[i + 1]
            
            for idx in range(start, end):
                j = neighbor_list[idx]
                dr = pos_i - positions[j]
                dr = minimum_image_scalar(dr, L)
                r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
                
                if r2 < rc2:
                    U += lj_shifted_energy_numba(r2, rc2)
        
        return U

    @njit(cache=True)
    def virial_pressure_nl_numba(positions, neighbor_list, neighbor_starts, L, rc, T, rc2):
        """Compute virial pressure using neighbor list.
        
        Args:
            positions: Particle positions, shape (N, 3)
            neighbor_list: Flat array of neighbor indices
            neighbor_starts: Starting index for each particle, shape (N+1,)
            L: Box length
            rc: Cutoff distance
            T: Temperature
            rc2: Squared cutoff distance
            
        Returns:
            Pressure in reduced units
        """
        N = positions.shape[0]
        V = L**3
        rho = N / V
        
        # Ideal gas contribution
        P_ideal = rho * T
        
        # Virial contribution
        W = 0.0
        
        for i in range(N):
            pos_i = positions[i]
            start = neighbor_starts[i]
            end = neighbor_starts[i + 1]
            
            for idx in range(start, end):
                j = neighbor_list[idx]
                dr = pos_i - positions[j]
                dr = minimum_image_scalar(dr, L)
                r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
                
                if r2 < rc2 and r2 > 0:
                    r = np.sqrt(r2)
                    F = lj_shifted_force_magnitude_numba(r2, rc2)
                    W += r * F
        
        P_virial = W / (3.0 * V)
        P_total = P_ideal + P_virial
        
        return P_total

    @njit(cache=True)
    def delta_energy_particle_move_nl_numba(positions, i, new_pos, neighbor_list, neighbor_starts, L, rc2):
        """Compute energy change for moving particle i using neighbor list.
        
        Note: This function assumes the neighbor list includes all pairs (i,j) with j > i.
        For a move of particle i, we need to check:
        1. Pairs where i is the first particle (in neighbor_list for i)
        2. Pairs where i is the second particle (need to find particles j that have i as neighbor)
        
        Args:
            positions: Current particle positions, shape (N, 3)
            i: Index of particle to move
            new_pos: New position of particle i, shape (3,)
            neighbor_list: Flat array of neighbor indices
            neighbor_starts: Starting index for each particle, shape (N+1,)
            L: Box length
            rc2: Squared cutoff distance
            
        Returns:
            Energy change: U(new) - U(old)
        """
        N = positions.shape[0]
        old_pos = positions[i]
        dU = 0.0
        
        # Old interactions where i is the first particle (j > i)
        start = neighbor_starts[i]
        end = neighbor_starts[i + 1]
        
        for idx in range(start, end):
            j = neighbor_list[idx]
            dr_old = old_pos - positions[j]
            dr_old = minimum_image_scalar(dr_old, L)
            r2_old = dr_old[0]*dr_old[0] + dr_old[1]*dr_old[1] + dr_old[2]*dr_old[2]
            
            if r2_old < rc2:
                dU -= lj_shifted_energy_numba(r2_old, rc2)
        
        # Old interactions where i is the second particle (j < i, and j has i as neighbor)
        # We need to scan all particles j < i and check if i is in their neighbor list
        for j in range(i):
            start_j = neighbor_starts[j]
            end_j = neighbor_starts[j + 1]
            
            # Check if i is in j's neighbor list
            for idx in range(start_j, end_j):
                if neighbor_list[idx] == i:
                    # Found pair (j, i)
                    dr_old = old_pos - positions[j]
                    dr_old = minimum_image_scalar(dr_old, L)
                    r2_old = dr_old[0]*dr_old[0] + dr_old[1]*dr_old[1] + dr_old[2]*dr_old[2]
                    
                    if r2_old < rc2:
                        dU -= lj_shifted_energy_numba(r2_old, rc2)
                    break  # i appears at most once in j's list
        
        # New interactions where i is the first particle
        for idx in range(start, end):
            j = neighbor_list[idx]
            dr_new = new_pos - positions[j]
            dr_new = minimum_image_scalar(dr_new, L)
            r2_new = dr_new[0]*dr_new[0] + dr_new[1]*dr_new[1] + dr_new[2]*dr_new[2]
            
            if r2_new < rc2:
                dU += lj_shifted_energy_numba(r2_new, rc2)
        
        # New interactions where i is the second particle
        for j in range(i):
            start_j = neighbor_starts[j]
            end_j = neighbor_starts[j + 1]
            
            for idx in range(start_j, end_j):
                if neighbor_list[idx] == i:
                    dr_new = new_pos - positions[j]
                    dr_new = minimum_image_scalar(dr_new, L)
                    r2_new = dr_new[0]*dr_new[0] + dr_new[1]*dr_new[1] + dr_new[2]*dr_new[2]
                    
                    if r2_new < rc2:
                        dU += lj_shifted_energy_numba(r2_new, rc2)
                    break
        
        return dU
