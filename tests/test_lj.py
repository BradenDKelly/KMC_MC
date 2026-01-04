"""Tests for Lennard-Jones energy calculations."""

import numpy as np
import pytest
from src.lj import total_energy, delta_energy_particle_move
from src.utils import init_lattice


def test_delta_energy_matches_brute_force():
    """Test that delta_energy matches brute-force total energy differences."""
    # Setup: small system for fast computation
    N = 32
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    rng = np.random.default_rng(42)
    
    # Initialize positions
    x = init_lattice(N, L, rng)
    
    # Test several random moves
    for _ in range(5):
        # Choose random particle and new position
        i = rng.integers(N)
        disp = (rng.random(3) * 2 - 1) * 0.1
        new_xi = (x[i] + disp) % L
        
        # Compute delta energy using delta_energy_particle_move
        dU_delta = delta_energy_particle_move(i, new_xi, x, L, rc)
        
        # Compute using brute force: U(new) - U(old)
        U_old = total_energy(x, L, rc)
        x_new = x.copy()
        x_new[i] = new_xi
        U_new = total_energy(x_new, L, rc)
        dU_brute_force = U_new - U_old
        
        # They should match (within numerical precision)
        np.testing.assert_allclose(dU_delta, dU_brute_force, rtol=1e-10, atol=1e-10,
                                   err_msg=f"delta_energy ({dU_delta}) does not match "
                                           f"brute-force difference ({dU_brute_force})")


def test_translational_invariance():
    """Test that energy is translationally invariant (energy doesn't change with translation)."""
    N = 32
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    rng = np.random.default_rng(123)
    
    # Initialize positions
    x = init_lattice(N, L, rng)
    
    # Compute reference energy
    U_ref = total_energy(x, L, rc)
    
    # Translate all particles by a random vector (wrap around)
    translation = rng.random(3) * L
    x_translated = (x + translation) % L
    
    # Energy should be the same (within numerical precision)
    U_translated = total_energy(x_translated, L, rc)
    
    np.testing.assert_allclose(U_ref, U_translated, rtol=1e-10, atol=1e-10,
                               err_msg=f"Energy not translationally invariant: "
                                       f"U_ref={U_ref}, U_translated={U_translated}")


def test_multiple_translational_invariance():
    """Test translational invariance with multiple translations."""
    N = 32
    rho = 0.5
    L = (N / rho) ** (1/3)
    rc = 2.5
    rng = np.random.default_rng(456)
    
    # Initialize positions
    x = init_lattice(N, L, rng)
    U_ref = total_energy(x, L, rc)
    
    # Test multiple translations
    for _ in range(5):
        translation = rng.random(3) * L
        x_translated = (x + translation) % L
        U_translated = total_energy(x_translated, L, rc)
        np.testing.assert_allclose(U_ref, U_translated, rtol=1e-10, atol=1e-10)

