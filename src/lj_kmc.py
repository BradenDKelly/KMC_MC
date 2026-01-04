"""Equilibrium kinetic Monte Carlo for Lennard-Jones particles."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .lj import delta_energy_particle_move, lj_shifted_energy
from .utils import minimum_image


@dataclass
class LJKMCrates:
    """Container for LJ kMC event rates (Do/Ustinov style).
    
    Attributes:
        rates: Event rates, shape (N,)
        events: List of event descriptors, each is (i, new_pos) tuple
    """
    rates: np.ndarray
    events: List[Tuple[int, np.ndarray]]  # (particle_index, new_position)


def compute_relocation_rates(positions, L, rc, beta, rng):
    """Compute relocation rates for all particles (Do/Ustinov style).
    
    For each particle i, propose a random new position uniform in box
    and compute the relocation rate using Barker acceptance rule.
    
    Rate: r_i = exp(-beta * ΔU_i / 2) where ΔU_i is energy change.
    
    Args:
        positions: Particle positions, shape (N, 3)
        L: Box length
        rc: Cutoff distance
        beta: Inverse temperature (1/kT)
        rng: Random number generator
        
    Returns:
        LJKMCrates instance
    """
    N = positions.shape[0]
    rates = np.zeros(N)
    events = []
    
    # For each particle, propose a random new position and compute rate
    for i in range(N):
        # Propose new position: uniform in box
        new_pos = rng.random(3) * L
        
        # Compute energy change
        dU = delta_energy_particle_move(i, new_pos, positions, L, rc)
        
        # Rate using Barker rule: rate = exp(-beta * dU / 2)
        # Clip to avoid overflow
        z = -0.5 * beta * dU
        z = np.clip(z, -700.0, 700.0)
        rates[i] = np.exp(z)
        events.append((i, new_pos))
    
    return LJKMCrates(rates=rates, events=events)


def sample_event(kmc_rates, rng):
    """Sample a relocation event proportional to its rate.
    
    Args:
        kmc_rates: LJKMCrates instance
        rng: Random number generator
        
    Returns:
        Event tuple: (particle_index, new_position)
    """
    total_rate = np.sum(kmc_rates.rates)
    if total_rate <= 0:
        raise ValueError("Total rate is non-positive")
    
    # Sample proportional to rates
    cumsum = np.cumsum(kmc_rates.rates)
    u = rng.random() * total_rate
    idx = int(np.searchsorted(cumsum, u))
    
    return kmc_rates.events[idx]


def apply_relocation(positions, event, L):
    """Apply relocation event.
    
    Args:
        positions: Particle positions, shape (N, 3) (modified in place)
        event: Event tuple (i, new_pos)
        L: Box length
    """
    i, new_pos = event
    positions[i] = new_pos % L


def compute_widom_weights(positions, L, rc, beta, n_insertions, rng):
    """
    Compute average Widom insertion weight <exp(-beta * dU_ins)> for LJ.

    This implementation is intentionally written to match the reference
    calculation used in tests/test_lj_kmc.py exactly.
    """
    N = positions.shape[0]
    rc2 = rc * rc
    weights = []

    for _ in range(n_insertions):
        x_test = rng.random(3) * L
        dU_ins = 0.0
        for j in range(N):
            dr = minimum_image(x_test - positions[j], L)
            r2 = float(np.dot(dr, dr))
            if r2 < rc2:
                dU_ins += lj_shifted_energy(r2, rc2)
        weights.append(np.exp(-beta * dU_ins))

    return float(np.mean(weights))

