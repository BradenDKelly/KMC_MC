"""Utilities for periodic boundary conditions and helper functions."""

import numpy as np


def minimum_image(dr, L):
    """Apply minimum image convention for periodic boundary conditions.
    
    Args:
        dr: Displacement vector(s), shape (3,) or (N, 3)
        L: Box length
        
    Returns:
        Minimum image displacement vector(s)
    """
    return dr - L * np.round(dr / L)


def init_lattice(N, L, rng):
    """Initialize particles on a simple cubic lattice with small random displacement.
    
    Args:
        N: Number of particles
        L: Box length
        rng: Random number generator (numpy.random.Generator)
        
    Returns:
        Array of particle positions, shape (N, 3), wrapped into [0, L)
    """
    n3 = int(np.ceil(N ** (1/3)))
    grid = np.linspace(0, L, n3, endpoint=False)
    pts = np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T
    x = pts[:N].copy()
    x += 0.01 * (rng.random(x.shape) - 0.5)
    return x % L

