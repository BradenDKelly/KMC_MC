"""Tests for Metropolis Monte Carlo and Widom excess chemical potential."""

import numpy as np
import pytest
from src.mc import run_metropolis_mc


def test_widom_mu_ex_reasonable_at_low_density():
    """Test that Widom mu_ex is reasonable at low density.
    
    At low density, mu_ex should be close to 0 (ideal gas limit).
    """
    # Very low density system
    N = 32
    rho = 0.01  # Very low density
    T = 1.0
    
    # Run shorter simulation for testing
    result = run_metropolis_mc(
        N=N,
        rho=rho,
        T=T,
        n_equil=1000,
        n_prod=2000,
        sample_every=100,
        widom_inserts=200,
        seed=789
    )
    
    mu_ex = result["mu_ex_mean"]
    
    # At low density, mu_ex should be close to 0 (within a reasonable range)
    # For ideal gas, mu_ex = 0. For a real gas at very low density, 
    # mu_ex should be small and negative (attractive interactions)
    # or small and positive (repulsive at short range dominates)
    # With LJ at low density, we expect small negative values
    assert mu_ex < 1.0, f"mu_ex ({mu_ex}) should be < 1.0 at low density"
    assert mu_ex > -5.0, f"mu_ex ({mu_ex}) should be > -5.0 at low density"


def test_widom_mu_ex_converges():
    """Test that Widom mu_ex produces finite results."""
    N = 64
    rho = 0.3
    T = 1.0
    
    result = run_metropolis_mc(
        N=N,
        rho=rho,
        T=T,
        n_equil=500,
        n_prod=1000,
        sample_every=100,
        widom_inserts=100,
        seed=999
    )
    
    # Check that mu_ex is finite
    assert np.isfinite(result["mu_ex_mean"]), "mu_ex_mean should be finite"
    assert np.isfinite(result["mu_ex_std"]), "mu_ex_std should be finite"
    assert result["mu_ex_std"] >= 0, "mu_ex_std should be non-negative"


def test_metropolis_mc_returns_reasonable_values():
    """Test that Metropolis MC returns reasonable structure."""
    N = 32
    rho = 0.5
    T = 1.0
    
    result = run_metropolis_mc(
        N=N,
        rho=rho,
        T=T,
        n_equil=500,
        n_prod=1000,
        sample_every=100,
        widom_inserts=100,
        seed=111
    )
    
    # Check all expected keys are present
    assert "L" in result
    assert "acceptance" in result
    assert "U_per_particle_mean" in result
    assert "mu_ex_mean" in result
    assert "mu_ex_std" in result
    
    # Check reasonable values
    assert result["L"] > 0, "Box length should be positive"
    assert 0 <= result["acceptance"] <= 1, "Acceptance should be in [0, 1]"
    assert np.isfinite(result["U_per_particle_mean"]), "Energy should be finite"
    assert np.isfinite(result["mu_ex_mean"]), "mu_ex should be finite"

