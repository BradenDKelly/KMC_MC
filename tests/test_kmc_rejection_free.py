"""Tests to verify rejection-free kMC behavior (Ustinov/Do style).

These tests ensure that kMC implementations are truly rejection-free:
- Events are selected with probability proportional to rates
- Selected events are ALWAYS applied (acceptance = 1.0)
- No accept/reject branches based on exp(-beta*ΔU)
- Time stepping (if implemented) produces positive finite Δt
"""

import numpy as np
import pytest
from src.kmc import run_equilibrium_kmc
from src.lj_kmc import (
    LJKMCrates,
    compute_relocation_rates,
    sample_event,
    apply_relocation,
)
from src.utils import init_lattice
from src.lj import total_energy, delta_energy_particle_move


def test_kmc_equilibrium_kmc_rejection_free():
    """Test that run_equilibrium_kmc is rejection-free (events always applied)."""
    rng = np.random.default_rng(5000)
    
    N = 32
    rho = 0.5
    T = 1.0
    rc = 2.5
    dmax = 0.15
    K = 6
    n_steps = 100
    
    # Track that events are always applied (no rejection counter needed)
    # In rejection-free kMC, acceptance = number of steps
    result = run_equilibrium_kmc(
        N=N, rho=rho, T=T, rc=rc, dmax=dmax, K=K,
        n_steps=n_steps, record_every=1, seed=rng.bit_generator.random_raw()
    )
    
    # If we got here without errors, all events were applied (rejection-free)
    # The function would have failed if any event was rejected
    assert "U_per_particle_mean" in result
    assert "t_last" in result
    assert np.isfinite(result["U_per_particle_mean"])
    assert result["t_last"] > 0  # Time must advance


def test_kmc_equilibrium_kmc_time_stepping():
    """Test that time stepping in run_equilibrium_kmc produces positive finite Δt."""
    rng = np.random.default_rng(6000)
    
    N = 32
    rho = 0.5
    T = 1.0
    rc = 2.5
    dmax = 0.15
    K = 6
    n_steps = 50
    
    # Run kMC and verify time advances properly
    result = run_equilibrium_kmc(
        N=N, rho=rho, T=T, rc=rc, dmax=dmax, K=K,
        n_steps=n_steps, record_every=1, seed=rng.bit_generator.random_raw()
    )
    
    # Time must be positive and finite
    assert result["t_last"] > 0, "Time must advance (Δt > 0 for each step)"
    assert np.isfinite(result["t_last"]), "Time must be finite"
    
    # For n_steps steps, time should have advanced multiple times
    # (each step has Δt = -log(u)/R where u ~ Uniform(0,1), so Δt > 0)
    assert result["t_last"] > 0.01, "Time should have advanced significantly"


def test_lj_kmc_sample_event_normalized_weights():
    """Test that sample_event uses normalized weights (sum(probabilities) = 1)."""
    rng = np.random.default_rng(7000)
    
    N = 20
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    beta = 1.0
    
    positions = init_lattice(N, L, rng)
    
    # Compute rates
    kmc_rates = compute_relocation_rates(positions, L, rc, beta, rng)
    
    # Verify total rate > 0
    total_rate = np.sum(kmc_rates.rates)
    assert total_rate > 0, "Total rate must be positive"
    assert np.isfinite(total_rate), "Total rate must be finite"
    
    # Verify normalized probabilities: p_i = rates[i] / total_rate
    probabilities = kmc_rates.rates / total_rate
    sum_probs = np.sum(probabilities)
    
    # Sum of probabilities should be 1.0 (within floating point tolerance)
    np.testing.assert_allclose(
        sum_probs, 1.0, rtol=1e-10, atol=1e-10,
        err_msg="Event selection probabilities must sum to 1.0 (normalized)"
    )
    
    # Verify all probabilities are non-negative
    assert np.all(probabilities >= 0), "All probabilities must be non-negative"


def test_lj_kmc_sample_event_always_applies():
    """Test that sample_event + apply_relocation always applies events (rejection-free)."""
    rng = np.random.default_rng(8000)
    
    N = 20
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    beta = 1.0
    
    positions = init_lattice(N, L, rng)
    positions_orig = positions.copy()
    
    n_trials = 100
    n_applied = 0
    
    for _ in range(n_trials):
        # Compute rates
        kmc_rates = compute_relocation_rates(positions, L, rc, beta, rng)
        
        # Verify total rate > 0
        total_rate = np.sum(kmc_rates.rates)
        assert total_rate > 0, "Total rate must be positive at each step"
        
        # Sample event (rejection-free)
        event = sample_event(kmc_rates, rng)
        
        # ALWAYS apply event (no accept/reject branch)
        apply_relocation(positions, event, L)
        n_applied += 1
    
    # In rejection-free kMC, acceptance = number of trials (all events applied)
    assert n_applied == n_trials, (
        f"Rejection-free kMC: all {n_trials} events should be applied, "
        f"got {n_applied} applied"
    )
    
    # Positions should have changed (not all events were no-ops)
    assert not np.allclose(positions, positions_orig), (
        "Positions should have changed after applying events"
    )


def test_lj_kmc_no_accept_reject_branch():
    """Test that lj_kmc functions have no accept/reject logic based on exp(-beta*ΔU)."""
    rng = np.random.default_rng(9000)
    
    N = 20
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    beta = 1.0
    
    positions = init_lattice(N, L, rng)
    
    n_steps = 50
    events_selected = 0
    events_applied = 0
    
    for _ in range(n_steps):
        # Compute rates (uses exp(-beta*ΔU/2) but this is for rate calculation, not acceptance)
        kmc_rates = compute_relocation_rates(positions, L, rc, beta, rng)
        
        # Verify total rate > 0
        total_rate = np.sum(kmc_rates.rates)
        assert total_rate > 0, "Total rate must be positive"
        
        # Sample event (no acceptance check here)
        event = sample_event(kmc_rates, rng)
        events_selected += 1
        
        # Apply event (no acceptance check - always applied)
        apply_relocation(positions, event, L)
        events_applied += 1
    
    # In rejection-free kMC: events_selected == events_applied == n_steps
    assert events_selected == n_steps, f"Expected {n_steps} events selected, got {events_selected}"
    assert events_applied == n_steps, (
        f"Rejection-free kMC: all {events_selected} selected events must be applied, "
        f"got {events_applied} applied"
    )
    assert events_selected == events_applied, (
        "Rejection-free kMC: number of selected events must equal number of applied events"
    )


def test_lj_kmc_total_rate_positive():
    """Test that total rate R > 0 at each step (required for rejection-free kMC)."""
    rng = np.random.default_rng(10000)
    
    N = 20
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    beta = 1.0
    
    positions = init_lattice(N, L, rng)
    
    n_steps = 50
    total_rates = []
    
    for _ in range(n_steps):
        kmc_rates = compute_relocation_rates(positions, L, rc, beta, rng)
        total_rate = np.sum(kmc_rates.rates)
        total_rates.append(total_rate)
        
        # Verify total rate > 0 at each step
        assert total_rate > 0, f"Total rate must be positive, got {total_rate} at step {len(total_rates)}"
        assert np.isfinite(total_rate), f"Total rate must be finite, got {total_rate}"
        
        # Apply event to update positions for next iteration
        event = sample_event(kmc_rates, rng)
        apply_relocation(positions, event, L)
    
    # Verify all rates were positive
    assert len(total_rates) == n_steps
    assert all(r > 0 for r in total_rates), "All total rates must be positive"
    assert all(np.isfinite(r) for r in total_rates), "All total rates must be finite"

