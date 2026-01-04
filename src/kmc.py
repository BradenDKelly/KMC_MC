"""Rejection-free equilibrium kMC sampler using continuous-time Barker rates."""

import numpy as np
from .utils import init_lattice
from .lj import total_energy, delta_energy_particle_move


def run_equilibrium_kmc(N=128, rho=0.8, T=1.0, rc=2.5, dmax=0.12, K=6,
                        n_steps=10_000, record_every=200, seed=123):
    """Run rejection-free equilibrium kMC sampler.
    
    Uses continuous-time Barker rates: rate r = exp(-beta*ΔU/2)
    Chooses one event proportional to rate, advances algorithmic time
    dt = -ln(u)/sum(r).
    
    Args:
        N: Number of particles
        rho: Number density
        T: Temperature
        rc: Cutoff distance
        dmax: Maximum displacement for trial moves
        K: Number of trial moves per particle per step
        n_steps: Number of kMC steps
        record_every: Record energy every N steps
        seed: Random seed
        
    Returns:
        Dictionary with results:
        - L: Box length
        - U_per_particle_mean: Mean energy per particle
        - t_last: Final algorithmic time
    """
    rng = np.random.default_rng(seed)
    beta = 1.0 / T
    V = N / rho
    L = V ** (1/3)

    x = init_lattice(N, L, rng)
    U = total_energy(x, L, rc)
    t = 0.0

    U_samples = []
    t_samples = []

    rates = np.empty((N, K), dtype=float)
    trial_pos = np.empty((N, K, 3), dtype=float)

    for step in range(1, n_steps+1):
        disps = (rng.random((N, K, 3))*2 - 1) * dmax
        trial_pos[:] = (x[:, None, :] + disps) % L

        for i in range(N):
            for k in range(K):
                dU = delta_energy_particle_move(i, trial_pos[i, k], x, L, rc)
                z = -0.5 * beta * dU
                z = np.clip(z, -700.0, 700.0)
                rates[i, k] = np.exp(z)

        Rtot = float(np.sum(rates))
        if not np.isfinite(Rtot) or Rtot <= 0:
            raise RuntimeError("Invalid total rate; reduce density/dmax or improve init.")

        dt = -np.log(rng.random()) / Rtot
        t += dt

        flat = rates.ravel()
        thresh = rng.random() * Rtot
        idx = int(np.searchsorted(np.cumsum(flat), thresh))
        i = idx // K
        k = idx % K

        new_xi = trial_pos[i, k].copy()
        dU = delta_energy_particle_move(i, new_xi, x, L, rc)
        x[i] = new_xi
        U += dU

        if step % record_every == 0:
            U_samples.append(U / N)
            t_samples.append(t)

    return {
        "L": L,
        "U_per_particle_mean": float(np.mean(U_samples)),
        "t_last": float(t_samples[-1]) if t_samples else t,
    }

