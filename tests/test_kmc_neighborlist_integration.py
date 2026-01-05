"""Integration tests for Do/Ustinov kMC relocation with neighbor list support."""

import numpy as np
import pytest
from src.utils import init_lattice
from src.backend import require_numba
from src.neighborlist import NeighborListConfig, NeighborList
from src.lj_kmc import (
    compute_relocation_rates,
    sample_event,
    apply_relocation,
)
from src.lj_numba import (
    total_energy_numba as total_energy_fast,
    virial_pressure_numba as virial_pressure_fast,
)


def run_kmc_relocation_brute(positions, L, rc, beta, n_sweeps, rng):
    """Do/Ustinov kMC relocation sampler (brute-force mode).
    
    Args:
        positions: Particle positions, shape (N, 3) (modified in place)
        L: Box length
        rc: Cutoff distance
        beta: Inverse temperature
        n_sweeps: Number of sweeps (N events per sweep)
        rng: Random number generator
        
    Returns:
        List of U/N values, one per sweep
    """
    require_numba("kMC relocation sampler")
    N = positions.shape[0]
    U_samples = []
    
    for sweep in range(n_sweeps):
        # Compute rates ONCE per sweep and reuse for all N events
        kmc_rates = compute_relocation_rates(positions, L, rc, beta, rng, nl=None)
        
        # Apply N events using the same rates list
        for _ in range(N):
            event = sample_event(kmc_rates, rng)
            apply_relocation(positions, event, L)
        
        # Measure U/N after sweep
        U = total_energy_fast(positions, L, rc)
        U_samples.append(U / N)
    
    return U_samples


def run_kmc_relocation_nl(positions, L, rc, beta, n_sweeps, rng, nl_config):
    """Do/Ustinov kMC relocation sampler (neighbor list mode).
    
    Args:
        positions: Particle positions, shape (N, 3) (modified in place)
        L: Box length
        rc: Cutoff distance
        beta: Inverse temperature
        n_sweeps: Number of sweeps (N events per sweep)
        rng: Random number generator
        nl_config: NeighborListConfig instance
        
    Returns:
        List of U/N values, one per sweep
    """
    require_numba("kMC relocation sampler")
    N = positions.shape[0]
    U_samples = []
    
    # Initialize neighbor list
    nl = NeighborList(positions, L, rc, skin=nl_config.skin)
    
    for sweep in range(n_sweeps):
        # Compute rates ONCE per sweep and reuse for all N events
        kmc_rates = compute_relocation_rates(positions, L, rc, beta, rng, nl=nl)
        
        # Apply N events using the same rates list
        for _ in range(N):
            event = sample_event(kmc_rates, rng)
            apply_relocation(positions, event, L)
            # Force rebuild after every relocation (relocations are large)
            nl.rebuild(positions)
        
        # Measure U/N after sweep
        U = total_energy_fast(positions, L, rc)
        U_samples.append(U / N)
    
    return U_samples


def test_kmc_runs_without_neighborlist():
    """Test that kMC relocation runs in brute-force mode without errors."""
    rng = np.random.default_rng(42)
    N = 32
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    T = 1.0
    beta = 1.0 / T
    
    positions = init_lattice(N, L, rng)
    positions += rng.random((N, 3)) * 0.1
    positions = positions % L
    
    # Run brute-force mode
    U_samples = run_kmc_relocation_brute(
        positions.copy(), L, rc, beta, n_sweeps=10, rng=rng
    )
    
    # Check that it completes without errors
    assert len(U_samples) == 10
    assert all(np.isfinite(u) for u in U_samples), "All energies should be finite"


@pytest.mark.slow
def test_kmc_neighborlist_statistical_equivalence_smallN():
    """Test that NL mode produces statistically equivalent results to brute-force.
    
    Uses fixed seed RNG and compares mean energy within statistical tolerance.
    """
    seed = 12345
    rng_brute = np.random.default_rng(seed)
    rng_nl = np.random.default_rng(seed)
    
    N = 32
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    T = 1.0
    beta = 1.0 / T
    skin = 0.2
    n_sweeps = 50  # Small enough for tests, enough for statistics
    
    # Initial positions (same for both runs)
    positions_initial = init_lattice(N, L, rng_brute)
    positions_initial += rng_brute.random((N, 3)) * 0.1
    positions_initial = positions_initial % L
    
    # Run brute-force mode
    positions_brute = positions_initial.copy()
    U_samples_brute = run_kmc_relocation_brute(
        positions_brute, L, rc, beta, n_sweeps, rng_brute
    )
    
    # Run neighbor list mode
    positions_nl = positions_initial.copy()
    nl_config = NeighborListConfig(skin=skin)
    U_samples_nl = run_kmc_relocation_nl(
        positions_nl, L, rc, beta, n_sweeps, rng_nl, nl_config
    )
    
    # Compute means
    mean_U_brute = np.mean(U_samples_brute)
    mean_U_nl = np.mean(U_samples_nl)
    
    # Compute standard errors (simple std/sqrt(n), not blocked)
    std_U_brute = np.std(U_samples_brute, ddof=1)
    std_U_nl = np.std(U_samples_nl, ddof=1)
    se_U_brute = std_U_brute / np.sqrt(len(U_samples_brute))
    se_U_nl = std_U_nl / np.sqrt(len(U_samples_nl))
    
    # Combined standard error (conservative)
    combined_se = np.sqrt(se_U_brute**2 + se_U_nl**2)
    
    # Difference should be within ~4σ (reasonable tolerance for statistical test)
    diff = abs(mean_U_brute - mean_U_nl)
    tolerance = 4.0 * combined_se
    
    assert diff < tolerance, (
        f"Energy mismatch: brute={mean_U_brute:.10f}, nl={mean_U_nl:.10f}, "
        f"diff={diff:.10f}, tolerance={tolerance:.10f} (4σ)"
    )

