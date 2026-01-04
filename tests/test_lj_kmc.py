"""Tests for equilibrium kMC for Lennard-Jones particles."""

import numpy as np
import pytest
from src.lj_kmc import (
    LJKMCrates,
    compute_relocation_rates,
    sample_event,
    apply_relocation,
    compute_widom_weights,
)
from src.utils import init_lattice
from src.lj import total_energy


def test_compute_relocation_rates():
    """Test that relocation rates are computed correctly."""
    rng = np.random.default_rng(1000)
    
    N = 32
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    beta = 1.0
    
    # Initialize positions
    positions = init_lattice(N, L, rng)
    
    # Compute rates
    kmc_rates = compute_relocation_rates(positions, L, rc, beta, rng)
    
    # Check structure
    assert len(kmc_rates.rates) == N, "Should have N rates"
    assert len(kmc_rates.events) == N, "Should have N events"
    assert np.all(kmc_rates.rates > 0), "All rates should be positive"
    assert np.all(np.isfinite(kmc_rates.rates)), "All rates should be finite"
    
    # Check events
    for i, event in enumerate(kmc_rates.events):
        idx, new_pos = event
        assert idx == i, f"Event {i} should have particle index {i}"
        assert new_pos.shape == (3,), "New position should have shape (3,)"
        assert np.all(new_pos >= 0), "New position should be >= 0"
        assert np.all(new_pos < L), "New position should be < L"


def test_sample_event():
    """Test that events are sampled proportional to rates."""
    rng = np.random.default_rng(2000)
    
    # Create simple rate list
    N = 10
    rates = np.ones(N) * np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])
    events = [(i, np.array([0.0, 0.0, 0.0])) for i in range(N)]
    kmc_rates = LJKMCrates(rates=rates, events=events)
    
    # Sample many times
    n_samples = 10000
    counts = np.zeros(N)
    
    for _ in range(n_samples):
        event = sample_event(kmc_rates, rng)
        idx, _ = event
        counts[idx] += 1
    
    # Check proportions (within statistical tolerance)
    total = np.sum(counts)
    expected_probs = rates / np.sum(rates)
    observed_probs = counts / total
    
    for i in range(N):
        # Allow 5% relative error
        assert abs(observed_probs[i] - expected_probs[i]) < 0.05, \
            f"Event {i} probability mismatch: expected {expected_probs[i]}, got {observed_probs[i]}"


def test_apply_relocation():
    """Test that relocation is applied correctly."""
    rng = np.random.default_rng(3000)
    
    N = 20
    L = 10.0
    positions = init_lattice(N, L, rng)
    positions_orig = positions.copy()
    
    # Apply relocation to particle 0
    i = 0
    new_pos = np.array([5.0, 5.0, 5.0])
    event = (i, new_pos)
    
    apply_relocation(positions, event, L)
    
    # Check that particle i was moved
    np.testing.assert_allclose(positions[i], new_pos % L, rtol=1e-10, atol=1e-10)
    
    # Check that other particles are unchanged
    for j in range(N):
        if j != i:
            np.testing.assert_array_equal(positions[j], positions_orig[j])


def test_widom_weights_consistency():
    """Test that Widom weights from kMC configuration match direct Widom.

    For the same LJ configuration, compute Widom weights using
    compute_widom_weights and compare with direct Widom insertion.

    IMPORTANT: Use independent RNG streams for insertion points so that both
    methods see identical x_test samples, regardless of other RNG consumption.
    """
    rng = np.random.default_rng(4000)

    # Create test configuration
    N = 32
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    beta = 1.0

    # Initialize positions (uses rng; that's fine)
    positions = init_lattice(N, L, rng)

    # Use separate, identically-seeded RNGs for insertion points
    n_insertions = 500
    rng_kmc = np.random.default_rng(12345)
    rng_direct = np.random.default_rng(12345)

    # Compute Widom weights using kMC function
    avg_weight_kmc = compute_widom_weights(
        positions, L, rc, beta, n_insertions, rng_kmc
    )

    # Compute Widom weights directly (same calculation, same insertion points)
    rc2 = rc * rc
    weights_direct = []

    from src.utils import minimum_image
    from src.lj import lj_shifted_energy

    for _ in range(n_insertions):
        x_test = rng_direct.random(3) * L
        dU_ins = 0.0
        for j in range(N):
            dr = minimum_image(x_test - positions[j], L)
            r2 = np.dot(dr, dr)
            if r2 < rc2:
                dU_ins += lj_shifted_energy(r2, rc2)
        weights_direct.append(np.exp(-beta * dU_ins))

    avg_weight_direct = float(np.mean(weights_direct))

    # They should match (same insertion points, same energy calculation)
    np.testing.assert_allclose(
        avg_weight_kmc, avg_weight_direct, rtol=1e-10, atol=1e-10,
        err_msg=f"Widom weights mismatch: kMC={avg_weight_kmc}, direct={avg_weight_direct}"
    )



def test_kMC_relocation_preserves_energy_structure():
    """Test that kMC relocation events preserve energy structure."""
    rng = np.random.default_rng(5000)
    
    N = 20
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    beta = 1.0
    
    # Initialize positions
    positions = init_lattice(N, L, rng)
    
    # Compute initial energy
    U_initial = total_energy(positions, L, rc)
    
    # Build event list
    kmc_rates = compute_relocation_rates(positions, L, rc, beta, rng)
    
    # Sample and apply an event
    event = sample_event(kmc_rates, rng)
    apply_relocation(positions, event, L)
    
    # Compute new energy
    U_new = total_energy(positions, L, rc)
    
    # Energy should be finite
    assert np.isfinite(U_new), "Energy should be finite after relocation"
    
    # Energy should have changed (random move)
    assert abs(U_new - U_initial) > 1e-10, "Energy should change after relocation"


def test_rates_positive_finite():
    """Test that all rates are positive and finite."""
    rng = np.random.default_rng(6000)
    
    N = 32
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    beta = 1.0
    
    positions = init_lattice(N, L, rng)
    kmc_rates = compute_relocation_rates(positions, L, rc, beta, rng)
    
    assert np.all(kmc_rates.rates > 0), "All rates should be positive"
    assert np.all(np.isfinite(kmc_rates.rates)), "All rates should be finite"
    assert np.sum(kmc_rates.rates) > 0, "Total rate should be positive"

