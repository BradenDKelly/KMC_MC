"""Pytest configuration and fixtures."""

import numpy as np
import pytest
from src.backend import NUMBA_AVAILABLE


def pytest_addoption(parser):
    """Add command-line options for pytest."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command-line options."""
    if config.getoption("--runslow"):
        # If --runslow is given, run all tests including slow ones
        return
    
    # Skip slow tests unless --runslow is given
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session", autouse=True)
def warmup_numba_jit():
    """Warm up numba JIT compilation before running tests.
    
    This fixture runs once per test session and pre-compiles numba kernels
    on a small dummy configuration to avoid JIT compilation overhead during
    actual test execution. This stabilizes test runtimes.
    """
    if not NUMBA_AVAILABLE:
        # Skip warmup if numba is not available
        return
    
    try:
        from src.lj_numba import (
            total_energy_numba,
            virial_pressure_numba,
            delta_energy_particle_move_numba,
        )
        
        # Create a small deterministic configuration for warmup
        N = 4
        L = 5.0
        rc = 2.5
        T = 1.0
        rc2 = rc * rc
        
        # Initialize small system
        rng = np.random.default_rng(42)  # Fixed seed for determinism
        positions = rng.random((N, 3)) * L
        
        # Warm up each kernel once
        _ = total_energy_numba(positions, L, rc)
        _ = virial_pressure_numba(positions, L, rc, T)
        
        # Warm up delta_energy with a small move
        i = 0
        new_pos = positions[i].copy()
        new_pos[0] += 0.1
        _ = delta_energy_particle_move_numba(positions, i, new_pos, L, rc2)
        
    except (ImportError, AttributeError):
        # If kernels can't be imported or fail, skip warmup
        # This can happen if numba is not properly installed
        pass

