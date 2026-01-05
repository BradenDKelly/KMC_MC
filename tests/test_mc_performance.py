"""Performance tests for MC brute-force mode."""

import pytest
import numpy as np
import time
from src.mc import advance_mc_sweeps
from src.utils import init_lattice


@pytest.mark.slow
def test_mc_bruteforce_performance():
    """Sanity check: brute-force MC should achieve reasonable moves/s.
    
    This test guards against performance regressions but uses conservative
    thresholds to avoid CI flakiness. It primarily logs performance metrics.
    """
    N = 256
    rho = 0.75
    T = 1.0
    rc = 2.5
    step = 0.1
    n_sweeps = 50
    
    # Initialize system
    rng = np.random.default_rng(42)
    L = (N / rho) ** (1/3)
    positions = init_lattice(N, L, rng)
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    
    # Time the MC stepping
    t_start = time.perf_counter()
    result = advance_mc_sweeps(positions, L, rc, T, step, n_sweeps, None, rng)
    t_end = time.perf_counter()
    
    wall_time = t_end - t_start
    n_events = result["attempts"]
    moves_per_second = n_events / wall_time if wall_time > 0 else 0.0
    
    # Log performance (for manual inspection)
    print(f"\nMC brute-force performance (N={N}, sweeps={n_sweeps}):")
    print(f"  Wall time: {wall_time:.3f} s")
    print(f"  Events: {n_events}")
    print(f"  Moves/s: {moves_per_second:.1f}")
    print(f"  Acceptance: {result['acceptance']:.3f}")
    
    # Conservative threshold: should be > 2000 moves/s on typical desktop
    # This is well below the expected ~20k moves/s, so it guards against major regressions
    # but won't fail on slower CI machines
    assert moves_per_second > 2000, (
        f"MC brute-force performance too low: {moves_per_second:.1f} moves/s "
        f"(expected > 2000 moves/s). This may indicate a regression."
    )
    
    # Verify physics is correct (acceptance should be reasonable)
    assert 0.0 < result["acceptance"] < 1.0, "Acceptance rate should be in (0, 1)"
    assert result["attempts"] == n_sweeps * N, "Should have exactly n_sweeps * N attempts"

