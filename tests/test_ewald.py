"""Tests for Ewald summation electrostatics."""

import numpy as np
import pytest
from src.ewald import (
    EwaldParams,
    EwaldCache,
    ewald_energy_total,
    k_vectors,
    build_cache,
    delta_energy_move,
    apply_move,
)
from src.utils import minimum_image


def test_delta_matches_total_recompute():
    """Test that delta energy matches brute-force recomputation.
    
    For random neutral charges and random move of one particle,
    verify U(new) - U(old) equals delta computed by recomputing totals.
    """
    rng = np.random.default_rng(42)
    
    # System parameters
    N = 20
    L = 10.0
    alpha = 0.35
    rcut = 4.5
    kmax = 5
    
    params = EwaldParams(L=L, alpha=alpha, rcut=rcut, kmax=kmax, ke=1.0)
    
    # Generate random neutral charges
    charges = rng.normal(size=N)
    charges = charges - np.mean(charges)  # Make neutral
    # Ensure exactly neutral (within tolerance)
    charges = charges - np.sum(charges) / N
    
    # Generate random positions
    positions = rng.random((N, 3)) * L
    
    # Compute initial energy
    U_old = ewald_energy_total(positions, charges, params)
    
    # Move one particle randomly
    i = rng.integers(N)
    disp = (rng.random(3) * 2 - 1) * 0.5
    positions_new = positions.copy()
    positions_new[i] = (positions[i] + disp) % L
    
    # Compute new energy
    U_new = ewald_energy_total(positions_new, charges, params)
    
    # Delta energy from recomputation
    delta_recompute = U_new - U_old
    
    # Verify delta is finite and reasonable
    assert np.isfinite(delta_recompute), "Delta energy should be finite"
    
    # For this test, we're just verifying the recomputation works
    # (incremental delta will come later)
    # The delta should be non-zero for a random move
    assert abs(delta_recompute) > 1e-10, "Delta energy should be non-zero for random move"


def test_charge_scaling_lambda_squared():
    """Test that energy scales as λ² when charges are scaled by λ."""
    rng = np.random.default_rng(123)
    
    # System parameters
    N = 20
    L = 10.0
    alpha = 0.35
    rcut = 4.5
    kmax = 5
    
    params = EwaldParams(L=L, alpha=alpha, rcut=rcut, kmax=kmax, ke=1.0)
    
    # Generate random neutral charges
    charges_base = rng.normal(size=N)
    charges_base = charges_base - np.mean(charges_base)  # Make neutral
    charges_base = charges_base - np.sum(charges_base) / N
    
    # Generate random positions
    positions = rng.random((N, 3)) * L
    
    # Compute energy for base charges
    E_base = ewald_energy_total(positions, charges_base, params)
    
    # Test multiple scaling factors
    lambdas = [0.5, 1.0, 1.5, 2.0]
    
    for lam in lambdas:
        charges_scaled = lam * charges_base
        E_scaled = ewald_energy_total(positions, charges_scaled, params)
        
        # Energy should scale as λ²
        expected_E = lam**2 * E_base
        
        # Check scaling (allow for numerical precision)
        np.testing.assert_allclose(
            E_scaled, expected_E, rtol=1e-8, atol=1e-8,
            err_msg=f"Energy scaling failed for λ={lam}: "
                    f"E_scaled={E_scaled}, expected={expected_E}"
        )


def test_translation_invariance():
    """Test that energy is translationally invariant.
    
    Shifting all positions by the same vector (mod L) should not change energy.
    """
    rng = np.random.default_rng(456)
    
    # System parameters
    N = 20
    L = 10.0
    alpha = 0.35
    rcut = 4.5
    kmax = 5
    
    params = EwaldParams(L=L, alpha=alpha, rcut=rcut, kmax=kmax, ke=1.0)
    
    # Generate random neutral charges
    charges = rng.normal(size=N)
    charges = charges - np.mean(charges)  # Make neutral
    charges = charges - np.sum(charges) / N
    
    # Generate random positions
    positions = rng.random((N, 3)) * L
    
    # Compute initial energy
    E_ref = ewald_energy_total(positions, charges, params)
    
    # Test multiple translations
    for _ in range(5):
        # Random translation vector
        translation = rng.random(3) * L
        positions_translated = (positions + translation) % L
        
        # Compute energy after translation
        E_translated = ewald_energy_total(positions_translated, charges, params)
        
        # Energy should be unchanged (within numerical precision)
        np.testing.assert_allclose(
            E_translated, E_ref, rtol=1e-10, atol=1e-10,
            err_msg=f"Energy not translationally invariant: "
                    f"E_ref={E_ref}, E_translated={E_translated}"
        )


def test_charge_neutrality_enforced():
    """Test that non-neutral systems raise ValueError."""
    rng = np.random.default_rng(789)
    
    # System parameters
    N = 20
    L = 10.0
    alpha = 0.35
    rcut = 4.5
    kmax = 5
    
    params = EwaldParams(L=L, alpha=alpha, rcut=rcut, kmax=kmax, ke=1.0)
    
    # Generate random positions
    positions = rng.random((N, 3)) * L
    
    # Non-neutral charges
    charges = rng.normal(size=N)
    # Make it clearly non-neutral
    charges = charges + 1.0
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="not net-neutral"):
        ewald_energy_total(positions, charges, params)


def test_k_vectors():
    """Test that k_vectors generates correct vectors."""
    L = 10.0
    kmax = 3
    
    n_vecs, k_vecs = k_vectors(L, kmax)
    
    # Check that n=0 is excluded
    assert not np.any(np.all(n_vecs == 0, axis=1)), "n=0 should be excluded"
    
    # Check that all |n| <= kmax
    n_mags = np.linalg.norm(n_vecs, axis=1)
    assert np.all(n_mags > 0), "All n should be non-zero"
    assert np.all(n_mags <= kmax), f"All |n| should be <= {kmax}"
    
    # Check k-vectors: k = 2*pi/L * n
    expected_k = (2.0 * np.pi / L) * n_vecs
    np.testing.assert_allclose(k_vecs, expected_k, rtol=1e-10, atol=1e-10)


def test_ewald_energy_components():
    """Test that Ewald energy components are reasonable."""
    rng = np.random.default_rng(999)
    
    # System parameters
    N = 20
    L = 10.0
    alpha = 0.35
    rcut = 4.5
    kmax = 5
    
    params = EwaldParams(L=L, alpha=alpha, rcut=rcut, kmax=kmax, ke=1.0)
    
    # Generate random neutral charges
    charges = rng.normal(size=N)
    charges = charges - np.mean(charges)  # Make neutral
    charges = charges - np.sum(charges) / N
    
    # Generate random positions
    positions = rng.random((N, 3)) * L
    
    # Compute energy
    E_total = ewald_energy_total(positions, charges, params)
    
    # Energy should be finite
    assert np.isfinite(E_total), "Total energy should be finite"
    
    # For neutral system, energy should be reasonable
    # (not extremely large or small)
    assert abs(E_total) < 1e6, "Energy should be reasonable in magnitude"


def test_incremental_delta_matches_total():
    """Test that incremental delta energy matches total recomputation.
    
    For many random single-particle moves, verify ΔU_incremental ≈ U_total(new)-U_total(old)
    within tight tolerance.
    """
    rng = np.random.default_rng(1000)
    
    # System parameters
    N = 20
    L = 10.0
    alpha = 0.35
    rcut = 4.5
    kmax = 5
    
    params = EwaldParams(L=L, alpha=alpha, rcut=rcut, kmax=kmax, ke=1.0)
    
    # Generate random neutral charges
    charges = rng.normal(size=N)
    charges = charges - np.mean(charges)  # Make neutral
    charges = charges - np.sum(charges) / N
    
    # Generate random positions
    positions = rng.random((N, 3)) * L
    
    # Build cache
    cache = build_cache(positions, charges, params)
    
    # Test many random moves
    n_moves = 50
    for move_idx in range(n_moves):
        # Store original positions for recomputation
        positions_old = positions.copy()
        
        # Choose random particle and move it
        i = rng.integers(N)
        disp = (rng.random(3) * 2 - 1) * 0.5
        r_new = (positions[i] + disp) % L
        
        # Compute incremental delta
        dU_incremental = delta_energy_move(cache, i, r_new, positions, charges)
        
        # Compute total energy before move
        U_old = ewald_energy_total(positions_old, charges, params)
        
        # Apply move
        apply_move(cache, i, r_new, positions, charges)
        
        # Compute total energy after move
        U_new = ewald_energy_total(positions, charges, params)
        
        # Delta from total recomputation
        dU_total = U_new - U_old
        
        # Verify they match (tight tolerance)
        np.testing.assert_allclose(
            dU_incremental, dU_total, rtol=1e-10, atol=1e-10,
            err_msg=f"Move {move_idx}: Incremental delta ({dU_incremental}) does not match "
                    f"total recomputation ({dU_total})"
        )


def test_cached_structure_factor_matches_recomputed():
    """Test that cached structure factor S matches recomputed S after moves.
    
    After applying move via cache update, verify cached S matches recomputed S
    occasionally (spot-check).
    """
    rng = np.random.default_rng(2000)
    
    # System parameters
    N = 20
    L = 10.0
    alpha = 0.35
    rcut = 4.5
    kmax = 5
    
    params = EwaldParams(L=L, alpha=alpha, rcut=rcut, kmax=kmax, ke=1.0)
    
    # Generate random neutral charges
    charges = rng.normal(size=N)
    charges = charges - np.mean(charges)  # Make neutral
    charges = charges - np.sum(charges) / N
    
    # Generate random positions
    positions = rng.random((N, 3)) * L
    
    # Build cache
    cache = build_cache(positions, charges, params)
    
    # Test multiple moves with spot-checks
    n_moves = 30
    check_every = 5  # Check every 5 moves
    
    for move_idx in range(n_moves):
        # Choose random particle and move it
        i = rng.integers(N)
        disp = (rng.random(3) * 2 - 1) * 0.5
        r_new = (positions[i] + disp) % L
        
        # Apply move via cache
        apply_move(cache, i, r_new, positions, charges)
        
        # Spot-check: recompute structure factors and compare
        if move_idx % check_every == 0:
            # Recompute structure factors from scratch
            _, k_vecs = k_vectors(L, kmax)
            S_recomputed = np.zeros(k_vecs.shape[0], dtype=complex)
            
            for k_idx in range(k_vecs.shape[0]):
                k = k_vecs[k_idx]
                k_dot_r = np.dot(positions, k)  # shape (N,)
                S_recomputed[k_idx] = np.sum(charges * np.exp(1j * k_dot_r))
            
            # Compare cached S with recomputed S
            np.testing.assert_allclose(
                cache.S, S_recomputed, rtol=1e-10, atol=1e-10,
                err_msg=f"After move {move_idx}: Cached S does not match recomputed S"
            )


def test_build_cache():
    """Test that build_cache creates correct cache."""
    rng = np.random.default_rng(3000)
    
    # System parameters
    N = 20
    L = 10.0
    alpha = 0.35
    rcut = 4.5
    kmax = 5
    
    params = EwaldParams(L=L, alpha=alpha, rcut=rcut, kmax=kmax, ke=1.0)
    
    # Generate random neutral charges
    charges = rng.normal(size=N)
    charges = charges - np.mean(charges)  # Make neutral
    charges = charges - np.sum(charges) / N
    
    # Generate random positions
    positions = rng.random((N, 3)) * L
    
    # Build cache
    cache = build_cache(positions, charges, params)
    
    # Check cache structure
    assert cache.params == params, "Cache params should match"
    assert cache.kvecs.shape[1] == 3, "kvecs should have shape (M, 3)"
    assert cache.k2.shape[0] == cache.kvecs.shape[0], "k2 should match kvecs length"
    assert cache.c_k.shape[0] == cache.kvecs.shape[0], "c_k should match kvecs length"
    assert cache.S.shape[0] == cache.kvecs.shape[0], "S should match kvecs length"
    assert cache.S.dtype == complex, "S should be complex"
    
    # Verify structure factors match direct computation
    _, k_vecs = k_vectors(L, kmax)
    for k_idx in range(k_vecs.shape[0]):
        k = k_vecs[k_idx]
        k_dot_r = np.dot(positions, k)
        S_expected = np.sum(charges * np.exp(1j * k_dot_r))
        np.testing.assert_allclose(
            cache.S[k_idx], S_expected, rtol=1e-10, atol=1e-10,
            err_msg=f"Structure factor mismatch for k-vector {k_idx}"
        )

