"""Metropolis NVT Monte Carlo simulation with Widom excess chemical potential."""

from typing import Optional
import numpy as np
from .utils import init_lattice, minimum_image
from .lj import total_energy, delta_energy_particle_move, lj_shifted_energy
from .neighborlist import NeighborListConfig, NeighborList
from .backend import require_numba, NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    from .lj_numba import delta_energy_particle_move_numba


def advance_mc_sweeps(positions, L, rc, T, max_disp, n_sweeps, nl, rng):
    """Advance MC simulation by n_sweeps sweeps (optimized stepping function).
    
    This is the core MC stepping logic extracted for reuse. Uses Numba-accelerated
    delta_energy_particle_move_numba in brute-force mode for performance.
    
    Args:
        positions: Particle positions, shape (N, 3) (modified in place)
        L: Box length
        rc: Cutoff distance
        T: Temperature
        max_disp: Maximum displacement
        n_sweeps: Number of sweeps to advance
        nl: NeighborList instance or None
        rng: Random number generator
        
    Returns:
        dict with keys: attempts, accepts, acceptance
    """
    require_numba("MC stepping")
    N = positions.shape[0]
    beta = 1.0 / T
    rc2 = rc * rc
    attempts = 0
    accepts = 0
    
    for _ in range(n_sweeps):
        for _ in range(N):
            attempts += 1
            i = rng.integers(N)
            disp = (rng.random(3) * 2 - 1) * max_disp
            new_pos = (positions[i] + disp) % L
            
            # Compute energy change
            if nl is not None:
                dU = nl.delta_energy_particle_move(positions, i, new_pos, L, rc)
            else:
                # Use Numba-accelerated version for brute-force mode
                dU = delta_energy_particle_move_numba(positions, i, new_pos, L, rc2)
            
            if dU <= 0.0 or rng.random() < np.exp(-beta * dU):
                positions[i] = new_pos
                accepts += 1
                if nl is not None:
                    nl.update(positions, force_rebuild=False)
    
    return {
        "attempts": attempts,
        "accepts": accepts,
        "acceptance": accepts / max(attempts, 1),
    }


def run_metropolis_mc(N=256, rho=0.8, T=1.0, rc=2.5, max_disp=0.12,
                      n_equil=20_000, n_prod=50_000, sample_every=200,
                      widom_inserts=500, seed=123, neighborlist: Optional[NeighborListConfig] = None):
    """Run Metropolis Monte Carlo NVT simulation with Widom excess chemical potential.
    
    Args:
        N: Number of particles
        rho: Number density
        T: Temperature
        rc: Cutoff distance
        max_disp: Maximum displacement for trial moves
        n_equil: Number of equilibration steps
        n_prod: Number of production steps
        sample_every: Sample energy and mu_ex every N steps
        widom_inserts: Number of test particle insertions per Widom estimate
        seed: Random seed
        neighborlist: NeighborListConfig to enable neighbor list mode, or None for brute-force
        
    Returns:
        Dictionary with results:
        - L: Box length
        - acceptance: Acceptance ratio
        - U_per_particle_mean: Mean energy per particle
        - mu_ex_mean: Mean excess chemical potential
        - mu_ex_std: Standard deviation of excess chemical potential
    """
    rng = np.random.default_rng(seed)
    beta = 1.0 / T
    V = N / rho
    L = V ** (1/3)

    x = init_lattice(N, L, rng)
    U = total_energy(x, L, rc)

    # Initialize neighbor list if requested
    nl = None
    if neighborlist is not None:
        try:
            nl = NeighborList(x, L, rc, skin=neighborlist.skin)
        except ImportError as e:
            raise ImportError(
                "Neighbor list mode requested but neighbor list backend is unavailable. "
                "Install numba: pip install numba"
            ) from e

    attempts = 0
    accepts = 0
    mu_samples = []
    U_samples = []

    def widom_mu_ex():
        """Compute Widom excess chemical potential via test particle insertion."""
        rc2 = rc*rc
        vals = []
        for _ in range(widom_inserts):
            xt = rng.random(3) * L
            dU = 0.0
            for j in range(N):
                dr = minimum_image(xt - x[j], L)
                r2 = np.dot(dr, dr)
                if r2 < rc2:
                    dU += lj_shifted_energy(r2, rc2)
            vals.append(np.exp(-beta * dU))
        avg = max(float(np.mean(vals)), 1e-300)
        return -T * np.log(avg)

    # Equilibration
    equil_result = advance_mc_sweeps(x, L, rc, T, max_disp, n_equil, nl, rng)
    attempts += equil_result["attempts"]
    accepts += equil_result["accepts"]
    # Recompute U after equilibration (for production sampling)
    U = total_energy(x, L, rc)

    # Production
    for step in range(1, n_prod+1):
        # Advance one step at a time to maintain sample_every cadence
        step_result = advance_mc_sweeps(x, L, rc, T, max_disp, 1, nl, rng)
        attempts += step_result["attempts"]
        accepts += step_result["accepts"]
        # Update U incrementally (approximate, but faster than full recompute)
        # For exact U, we'd need to track dU, but for sampling purposes this is fine
        # since we only use U_samples for mean, not exact values
        
        if step % sample_every == 0:
            U = total_energy(x, L, rc)  # Exact recompute for sampling
            U_samples.append(U / N)
            mu_samples.append(widom_mu_ex())

    return {
        "L": L,
        "acceptance": accepts / max(attempts, 1),
        "U_per_particle_mean": float(np.mean(U_samples)),
        "mu_ex_mean": float(np.mean(mu_samples)),
        "mu_ex_std": float(np.std(mu_samples, ddof=1)),
    }

