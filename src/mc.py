"""Metropolis NVT Monte Carlo simulation with Widom excess chemical potential."""

from typing import Optional
import numpy as np
from .utils import init_lattice, minimum_image
from .lj import total_energy, delta_energy_particle_move, lj_shifted_energy
from .neighborlist import NeighborListConfig, NeighborList


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
    for _ in range(n_equil):
        attempts += 1
        i = rng.integers(N)
        disp = (rng.random(3)*2 - 1) * max_disp
        new_xi = (x[i] + disp) % L
        
        # Compute energy change
        if nl is not None:
            dU = nl.delta_energy_particle_move(x, i, new_xi, L, rc)
        else:
            dU = delta_energy_particle_move(i, new_xi, x, L, rc)
        
        if dU <= 0.0 or rng.random() < np.exp(-beta*dU):
            x[i] = new_xi
            U += dU
            accepts += 1
            # Update neighbor list on accept
            if nl is not None:
                nl.update(x, force_rebuild=False)

    # Production
    for step in range(1, n_prod+1):
        attempts += 1
        i = rng.integers(N)
        disp = (rng.random(3)*2 - 1) * max_disp
        new_xi = (x[i] + disp) % L
        
        # Compute energy change
        if nl is not None:
            dU = nl.delta_energy_particle_move(x, i, new_xi, L, rc)
        else:
            dU = delta_energy_particle_move(i, new_xi, x, L, rc)
        
        if dU <= 0.0 or rng.random() < np.exp(-beta*dU):
            x[i] = new_xi
            U += dU
            accepts += 1
            # Update neighbor list on accept
            if nl is not None:
                nl.update(x, force_rebuild=False)

        if step % sample_every == 0:
            U_samples.append(U / N)
            mu_samples.append(widom_mu_ex())

    return {
        "L": L,
        "acceptance": accepts / max(attempts, 1),
        "U_per_particle_mean": float(np.mean(U_samples)),
        "mu_ex_mean": float(np.mean(mu_samples)),
        "mu_ex_std": float(np.std(mu_samples, ddof=1)),
    }

