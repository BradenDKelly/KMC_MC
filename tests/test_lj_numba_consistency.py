"""Test consistency between numba-accelerated and Python reference implementations.

These tests verify that numba kernels produce the same results as Python reference
implementations. Numba is required to run these tests.
"""

import numpy as np
import pytest
from src.utils import init_lattice
from src.backend import require_numba, NUMBA_AVAILABLE
from src.lj import total_energy, virial_pressure, delta_energy_particle_move
from src.lj_numba import (
    total_energy_numba,
    virial_pressure_numba,
    delta_energy_particle_move_numba,
)


@pytest.fixture(autouse=True)
def require_numba_fixture():
    """Require numba for all tests in this module."""
    require_numba("Numba consistency tests")
def test_total_energy_consistency():
    """Test that numba total_energy matches Python reference."""
    rng = np.random.default_rng(42)
    N = 64
    L = 5.0
    rc = 2.5
    
    positions = init_lattice(N, L, rng)
    # Add some random perturbations
    positions += rng.random((N, 3)) * 0.1
    
    U_python = total_energy(positions, L, rc)
    U_numba = total_energy_numba(positions, L, rc)
    
    assert np.allclose(U_python, U_numba, rtol=1e-12, atol=1e-12), (
        f"Total energy mismatch: Python={U_python:.10f}, Numba={U_numba:.10f}"
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")
def test_virial_pressure_consistency():
    """Test that numba virial_pressure matches Python reference."""
    rng = np.random.default_rng(43)
    N = 64
    L = 5.0
    rc = 2.5
    T = 1.3
    
    positions = init_lattice(N, L, rng)
    # Add some random perturbations
    positions += rng.random((N, 3)) * 0.1
    
    P_python = virial_pressure(positions, L, rc, T)
    P_numba = virial_pressure_numba(positions, L, rc, T)
    
    assert np.allclose(P_python, P_numba, rtol=1e-12, atol=1e-12), (
        f"Virial pressure mismatch: Python={P_python:.10f}, Numba={P_numba:.10f}"
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")
def test_delta_energy_particle_move_consistency():
    """Test that numba delta_energy_particle_move matches Python reference."""
    rng = np.random.default_rng(44)
    N = 64
    L = 5.0
    rc = 2.5
    rc2 = rc * rc
    
    positions = init_lattice(N, L, rng)
    # Add some random perturbations
    positions += rng.random((N, 3)) * 0.1
    
    # Test several random moves
    for _ in range(10):
        i = rng.integers(N)
        new_pos = rng.random(3) * L
        
        dU_python = delta_energy_particle_move(i, new_pos, positions, L, rc)
        dU_numba = delta_energy_particle_move_numba(positions, i, new_pos, L, rc * rc)
        
        assert np.allclose(dU_python, dU_numba, rtol=1e-12, atol=1e-12), (
            f"Delta energy mismatch for particle {i}: Python={dU_python:.10f}, Numba={dU_numba:.10f}"
        )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")
def test_numba_kernels_multiple_configs():
    """Test consistency across multiple random configurations."""
    rng = np.random.default_rng(45)
    N = 32
    L = 4.0
    rc = 2.5
    T = 1.3
    rc2 = rc * rc
    
    for config_idx in range(5):
        positions = rng.random((N, 3)) * L
        
        # Total energy
        U_python = total_energy(positions, L, rc)
        U_numba = total_energy_numba(positions, L, rc)
        assert np.allclose(U_python, U_numba, rtol=1e-12, atol=1e-12), (
            f"Config {config_idx}: Total energy mismatch"
        )
        
        # Virial pressure
        P_python = virial_pressure(positions, L, rc, T)
        P_numba = virial_pressure_numba(positions, L, rc, T)
        assert np.allclose(P_python, P_numba, rtol=1e-12, atol=1e-12), (
            f"Config {config_idx}: Virial pressure mismatch"
        )
        
        # Delta energy for a few moves
        for _ in range(3):
            i = rng.integers(N)
            new_pos = rng.random(3) * L
            dU_python = delta_energy_particle_move(i, new_pos, positions, L, rc)
            dU_numba = delta_energy_particle_move_numba(positions, i, new_pos, L, rc * rc)
            assert np.allclose(dU_python, dU_numba, rtol=1e-12, atol=1e-12), (
                f"Config {config_idx}: Delta energy mismatch"
            )

