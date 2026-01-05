"""Neighbor list wrapper with rebuild tracking.

This module provides a Python interface to the numba-accelerated neighbor list,
with automatic tracking of particle displacements and rebuild criteria.
"""

from dataclasses import dataclass
import numpy as np
from .backend import require_numba, NUMBA_AVAILABLE
from .utils import minimum_image

if NUMBA_AVAILABLE:
    from .neighborlist_numba import (
        build_neighbor_list_numba,
        total_energy_nl_numba,
        virial_pressure_nl_numba,
        delta_energy_particle_move_nl_numba,
    )


@dataclass(frozen=True)
class NeighborListConfig:
    """Configuration for neighbor list usage in MC simulations.
    
    Args:
        skin: Skin distance for rebuild criterion (rebuild when max displacement > skin/2)
    """
    skin: float


class NeighborList:
    """Neighbor list with automatic rebuild tracking.
    
    Tracks particle positions and rebuilds the neighbor list when particles
    have moved more than skin/2 since the last rebuild.
    """
    
    def __init__(self, positions, L, rc, skin=0.2):
        """Initialize neighbor list.
        
        Args:
            positions: Initial particle positions, shape (N, 3)
            L: Box length
            rc: Cutoff distance
            skin: Skin distance for rebuild criterion (rebuild when max displacement > skin/2)
        """
        require_numba("Neighbor list")
        
        self.L = L
        self.rc = rc
        self.skin = skin
        self.rc2 = rc * rc
        self.cutoff2 = (rc + skin) ** 2  # Neighbor list uses rc + skin as cutoff
        
        # Store reference positions for displacement tracking
        self.N = positions.shape[0]
        self.positions_ref = positions.copy()
        
        # Build initial neighbor list
        self.neighbor_list, self.neighbor_starts = build_neighbor_list_numba(
            positions, L, self.cutoff2
        )
    
    def needs_rebuild(self, positions):
        """Check if neighbor list needs rebuilding.
        
        Args:
            positions: Current particle positions, shape (N, 3)
            
        Returns:
            True if max displacement > skin/2, False otherwise
        """
        # Compute maximum displacement since last rebuild using minimum image convention
        dr = positions - self.positions_ref
        dr = minimum_image(dr, self.L)
        
        # Compute displacement magnitudes
        disp = np.sqrt(np.sum(dr * dr, axis=1))
        max_disp = np.max(disp)
        
        return max_disp > 0.5 * self.skin
    
    def rebuild(self, positions):
        """Rebuild neighbor list from current positions.
        
        Args:
            positions: Current particle positions, shape (N, 3)
        """
        self.positions_ref = positions.copy()
        self.neighbor_list, self.neighbor_starts = build_neighbor_list_numba(
            positions, self.L, self.cutoff2
        )
    
    def update(self, positions, force_rebuild=False):
        """Update neighbor list if needed.
        
        Args:
            positions: Current particle positions, shape (N, 3)
            force_rebuild: If True, force rebuild regardless of displacement
        """
        if force_rebuild or self.needs_rebuild(positions):
            self.rebuild(positions)
    
    def total_energy(self, positions, L, rc):
        """Compute total energy using neighbor list.
        
        Args:
            positions: Particle positions, shape (N, 3)
            L: Box length
            rc: Cutoff distance (must match self.rc)
            
        Returns:
            Total potential energy
        """
        if abs(rc - self.rc) > 1e-10:
            raise ValueError(f"rc mismatch: {rc} != {self.rc}")
        
        return total_energy_nl_numba(
            positions, self.neighbor_list, self.neighbor_starts, L, self.rc2
        )
    
    def virial_pressure(self, positions, L, rc, T):
        """Compute virial pressure using neighbor list.
        
        Args:
            positions: Particle positions, shape (N, 3)
            L: Box length
            rc: Cutoff distance (must match self.rc)
            T: Temperature
            
        Returns:
            Pressure in reduced units
        """
        if abs(rc - self.rc) > 1e-10:
            raise ValueError(f"rc mismatch: {rc} != {self.rc}")
        
        return virial_pressure_nl_numba(
            positions, self.neighbor_list, self.neighbor_starts, L, rc, T, self.rc2
        )
    
    def delta_energy_particle_move(self, positions, i, new_pos, L, rc):
        """Compute energy change for moving particle i.
        
        Args:
            positions: Current particle positions, shape (N, 3)
            i: Index of particle to move
            new_pos: New position of particle i, shape (3,)
            L: Box length
            rc: Cutoff distance (must match self.rc)
            
        Returns:
            Energy change: U(new) - U(old)
        """
        if abs(rc - self.rc) > 1e-10:
            raise ValueError(f"rc mismatch: {rc} != {self.rc}")
        
        return delta_energy_particle_move_nl_numba(
            positions, i, new_pos, self.neighbor_list, self.neighbor_starts, L, self.rc2
        )



