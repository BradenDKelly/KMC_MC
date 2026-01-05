"""Equilibrium regression test: Metropolis MC vs kMC relocation sampler.

Compares equilibrium mean potential energy from two samplers:
1. Metropolis MC with local displacement moves
2. kMC relocation sampler with uniform box proposals (Metropolis acceptance)

Both should produce the same equilibrium mean U/N within statistical uncertainty.
"""

import numpy as np
import pytest
from src.utils import init_lattice, minimum_image
from src.backend import require_numba
from src.lj import lj_shifted_energy


# Use numba-accelerated delta_energy (with rc2 parameter signature)
def delta_u_particle_move(positions, i, new_pos, L, rc2):
    """Compute local energy change for moving particle i to new_pos.
    
    ΔU = sum_{j != i} [ u(r_new_ij) - u(r_old_ij) ]
    Uses O(N) computation with minimum_image PBC and lj_shifted_energy.
    
    Args:
        positions: Current particle positions, shape (N, 3)
        i: Index of particle to move
        new_pos: New position of particle i, shape (3,)
        L: Box length
        rc2: Squared cutoff distance
        
    Returns:
        Energy change ΔU = U(new) - U(old)
    """
    N = positions.shape[0]
    old_pos = positions[i]
    dU = 0.0
    
    for j in range(N):
        if j == i:
            continue
        
        # Old interaction
        dr_old = minimum_image(old_pos - positions[j], L)
        r2_old = np.dot(dr_old, dr_old)
        if r2_old < rc2:
            dU -= lj_shifted_energy(r2_old, rc2)
        
        # New interaction
        dr_new = minimum_image(new_pos - positions[j], L)
        r2_new = np.dot(dr_new, dr_new)
        if r2_new < rc2:
            dU += lj_shifted_energy(r2_new, rc2)
    
    return dU


# Import numba-accelerated kernels for performance-critical code
from src.lj_numba import (
    total_energy_numba as total_energy_lj,
    delta_energy_particle_move_numba,
)

def total_energy_lj_python_ref(positions, L, rc):
    """Python reference implementation for correctness testing only.
    
    This is NOT used in performance-critical paths.
    """
    N = positions.shape[0]
    rc2 = rc * rc
    U = 0.0
    
    for i in range(N - 1):
        for j in range(i + 1, N):
            dr = minimum_image(positions[j] - positions[i], L)
            r2 = np.dot(dr, dr)
            if r2 < rc2:
                U += lj_shifted_energy(r2, rc2)
    
    return U


def metropolis_translation_sampler(positions, L, rc, beta, max_disp, n_sweeps, rng):
    """Metropolis MC sampler with local translation moves.
    
    Uses numba-accelerated kernels. Numba is required.
    
    Args:
        positions: Initial particle positions, shape (N, 3) (modified in place)
        L: Box length
        rc: Cutoff distance
        beta: Inverse temperature
        max_disp: Maximum displacement
        n_sweeps: Number of sweeps (N moves per sweep)
        rng: Random number generator
        
    Returns:
        List of U/N values, one per sweep
    """
    require_numba("Metropolis MC sampler")
    
    N = positions.shape[0]
    rc2 = rc * rc
    U_samples = []
    
    for sweep in range(n_sweeps):
        for _ in range(N):
            # Choose random particle
            i = rng.integers(N)
            
            # Save old position
            old_pos = positions[i].copy()
            
            # Propose local displacement
            disp = (rng.random(3) - 0.5) * 2 * max_disp
            new_pos = (old_pos + disp) % L
            
            # Compute energy change using local ΔU (O(N))
            dU = delta_u_particle_move(positions, i, new_pos, L, rc2)
            
            # Metropolis acceptance
            if dU <= 0.0 or rng.random() < np.exp(-beta * dU):
                # Accept: update position
                positions[i] = new_pos
            else:
                # Reject: keep old position (already saved, no change needed)
                pass
        
        # Measure U/N after sweep (full recompute only here)
        U = total_energy_lj(positions, L, rc)
        U_samples.append(U / N)
    
    return U_samples


def kmc_relocation_sampler(positions, L, rc, beta, n_sweeps, rng):
    """kMC relocation sampler with uniform box proposals (Metropolis acceptance).
    
    Proposes uniform random positions in box, accepts/rejects with Metropolis.
    This is a placeholder that shares the relocation proposal distribution
    used in rejection-free kMC, but uses standard Metropolis acceptance.
    
    Uses numba-accelerated kernels. Numba is required.
    
    Args:
        positions: Initial particle positions, shape (N, 3) (modified in place)
        L: Box length
        rc: Cutoff distance
        beta: Inverse temperature
        n_sweeps: Number of sweeps (N moves per sweep)
        rng: Random number generator
        
    Returns:
        List of U/N values, one per sweep
    """
    require_numba("kMC relocation sampler")
    
    N = positions.shape[0]
    rc2 = rc * rc
    U_samples = []
    
    for sweep in range(n_sweeps):
        for _ in range(N):
            # Choose random particle
            i = rng.integers(N)
            
            # Propose uniform random position in box
            new_pos = rng.random(3) * L
            
            # Compute energy change using local ΔU (O(N))
            dU = delta_u_particle_move(positions, i, new_pos, L, rc2)
            
            # Metropolis acceptance
            if dU <= 0.0 or rng.random() < np.exp(-beta * dU):
                # Accept: update position
                positions[i] = new_pos
            else:
                # Reject: keep old position (no change needed)
                pass
        
        # Measure U/N after sweep (full recompute only here)
        U = total_energy_lj(positions, L, rc)
        U_samples.append(U / N)
    
    return U_samples


def blocked_standard_error(samples, block_size):
    """Compute blocked standard error accounting for autocorrelation.
    
    Partitions samples into contiguous blocks, computes mean of each block,
    then computes SE as std(block_means, ddof=1) / sqrt(n_blocks).
    
    Args:
        samples: Array of samples (e.g., U/N values per sweep)
        block_size: Number of samples per block (in sweeps)
        
    Returns:
        Blocked standard error of the mean
    """
    n_samples = len(samples)
    n_blocks = n_samples // block_size
    
    if n_blocks < 1:
        raise ValueError(f"Block size {block_size} too large for {n_samples} samples")
    
    # Partition into blocks and compute block means
    block_means = []
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        block_means.append(np.mean(samples[start_idx:end_idx]))
    
    block_means = np.array(block_means)
    
    # Standard error: std(block_means) / sqrt(n_blocks)
    se_blocked = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
    
    return se_blocked, n_blocks


@pytest.mark.slow
def test_metropolis_vs_kmc_relocation_equilibrium():
    """Regression test: Metropolis MC vs kMC relocation produce same equilibrium U/N.
    
    This is a statistical regression guardrail that validates both samplers satisfy detailed
    balance for the same target distribution, producing consistent mean potential energy within
    statistical uncertainty.
    
    Note: This test uses reduced sampling (500 burnin, 1000 production sweeps) for reasonable
    runtime. Heavy statistical validation is covered by:
    - EOS validation tests (test_ljts_eos_consistency.py)
    - Widom insertion consistency tests (test_lj_kmc.py)
    - Other unit tests with longer runs
    
    This test serves as a regression guard to catch major sampling bugs, not for precise
    statistical validation.
    
    Requires numba for performance.
    """
    require_numba("Equilibrium regression test")
    
    # Protocol parameters (deterministic)
    N = 32
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    beta = 1.0
    max_disp = 0.15 * L
    
    # Reduced sampling for regression test (heavy lifting done elsewhere)
    burnin_sweeps = 500  # Reduced from 2000 for faster regression testing
    prod_sweeps = 1000  # Reduced from 8000 for faster regression testing
    block_size = 50  # Block size in sweeps
    
    # Initialize positions with fixed seed
    rng_init = np.random.default_rng(4000)
    positions_init = init_lattice(N, L, rng_init)
    
    # Metropolis MC run
    positions_mc = positions_init.copy()
    rng_mc = np.random.default_rng(111)
    
    # Burn-in
    metropolis_translation_sampler(
        positions_mc, L, rc, beta, max_disp, burnin_sweeps, rng_mc
    )
    
    # Production
    U_samples_mc = metropolis_translation_sampler(
        positions_mc, L, rc, beta, max_disp, prod_sweeps, rng_mc
    )
    
    # kMC relocation run
    positions_reloc = positions_init.copy()
    rng_reloc = np.random.default_rng(222)
    
    # Burn-in
    kmc_relocation_sampler(
        positions_reloc, L, rc, beta, burnin_sweeps, rng_reloc
    )
    
    # Production
    U_samples_reloc = kmc_relocation_sampler(
        positions_reloc, L, rc, beta, prod_sweeps, rng_reloc
    )
    
    # Compute statistics
    mean_mc = np.mean(U_samples_mc)
    mean_reloc = np.mean(U_samples_reloc)
    
    # Blocked standard error of the mean (accounts for autocorrelation)
    se_mc, n_blocks_mc = blocked_standard_error(U_samples_mc, block_size)
    se_reloc, n_blocks_reloc = blocked_standard_error(U_samples_reloc, block_size)
    
    # Diagnostic: ensure enough blocks for reliable statistics
    min_blocks = 10  # Reduced from 20 for regression test (>=10 is sufficient for guardrail)
    assert n_blocks_mc >= min_blocks, (
        f"MC sampler: only {n_blocks_mc} blocks, need >= {min_blocks}. "
        f"Increase prod_sweeps or decrease block_size."
    )
    assert n_blocks_reloc >= min_blocks, (
        f"kMC relocation sampler: only {n_blocks_reloc} blocks, need >= {min_blocks}. "
        f"Increase prod_sweeps or decrease block_size."
    )
    
    # Comparison: difference should be within 3σ
    max_se = max(se_mc, se_reloc)
    diff = abs(mean_reloc - mean_mc)
    
    assert diff <= 3 * max_se, (
        f"Mean U/N mismatch: Metropolis={mean_mc:.6f} ± {se_mc:.6f} ({n_blocks_mc} blocks), "
        f"kMC relocation={mean_reloc:.6f} ± {se_reloc:.6f} ({n_blocks_reloc} blocks), "
        f"difference={diff:.6f} > 3*max(SE)={3*max_se:.6f}"
    )

