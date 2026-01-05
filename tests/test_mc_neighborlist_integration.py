"""Integration tests for Metropolis MC with neighbor list support."""

import numpy as np
import pytest
from src.mc import run_metropolis_mc
from src.neighborlist import NeighborListConfig


def test_mc_runs_without_neighborlist():
    """Test that MC runs in brute-force mode without errors."""
    N = 32
    rho = 0.5
    T = 1.0
    rc = 2.5
    max_disp = 0.1
    
    # Run brute-force mode (neighborlist=None)
    result = run_metropolis_mc(
        N=N,
        rho=rho,
        T=T,
        rc=rc,
        max_disp=max_disp,
        n_equil=100,
        n_prod=200,
        sample_every=50,
        widom_inserts=50,
        seed=42,
        neighborlist=None  # Explicitly test brute-force mode
    )
    
    # Check that it completes without errors
    assert "L" in result
    assert "acceptance" in result
    assert "U_per_particle_mean" in result
    
    # Sanity check: energy should be finite
    assert np.isfinite(result["U_per_particle_mean"]), "Energy should be finite"
    assert 0 <= result["acceptance"] <= 1, "Acceptance should be in [0, 1]"


def test_mc_neighborlist_matches_bruteforce_when_no_rebuild():
    """Test that NL mode produces identical results to brute-force when no rebuild occurs.
    
    Uses large skin and small step size to avoid rebuilds, and same RNG seed
    to ensure identical random number sequences.
    """
    N = 32
    rho = 0.5
    T = 1.0
    rc = 2.5
    max_disp = 0.02  # Very small step size
    skin = 1.0  # Very large skin (much larger than max_disp)
    
    # Use same seed for both runs
    seed = 12345
    
    # Run brute-force mode
    result_brute = run_metropolis_mc(
        N=N,
        rho=rho,
        T=T,
        rc=rc,
        max_disp=max_disp,
        n_equil=50,
        n_prod=100,
        sample_every=50,
        widom_inserts=20,
        seed=seed,
        neighborlist=None
    )
    
    # Run neighbor list mode with same parameters
    nl_config = NeighborListConfig(skin=skin)
    result_nl = run_metropolis_mc(
        N=N,
        rho=rho,
        T=T,
        rc=rc,
        max_disp=max_disp,
        n_equil=50,
        n_prod=100,
        sample_every=50,
        widom_inserts=20,
        seed=seed,
        neighborlist=nl_config
    )
    
    # Check that acceptance counts match (should be identical with same RNG seed)
    assert result_brute["acceptance"] == result_nl["acceptance"], (
        f"Acceptance should match: brute={result_brute['acceptance']:.10f}, "
        f"nl={result_nl['acceptance']:.10f}"
    )
    
    # Check that energies match tightly (within numerical precision)
    np.testing.assert_allclose(
        result_brute["U_per_particle_mean"],
        result_nl["U_per_particle_mean"],
        rtol=1e-10,
        atol=1e-10,
        err_msg="Energy should match between brute-force and NL modes"
    )
    
    # Check that mu_ex matches (within reasonable tolerance)
    np.testing.assert_allclose(
        result_brute["mu_ex_mean"],
        result_nl["mu_ex_mean"],
        rtol=1e-8,
        atol=1e-8,
        err_msg="Widom mu_ex should match between brute-force and NL modes"
    )

