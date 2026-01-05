
"""Test correctness of neighbor list implementation.

These tests verify that neighbor-list-based calculations are numerically identical
to brute-force reference implementations. Tests cover:
- Pair coverage (all pairs with r < rc appear, no pairs beyond rc + skin)
- Total energy equivalence
- Virial/pressure equivalence
- ΔU equivalence for trial moves
- Rebuild logic (displacements < skin/2 → no rebuild, > skin/2 → rebuild)

Numba is required to run these tests.
"""

import numpy as np
import pytest
from src.utils import init_lattice, minimum_image
from src.backend import require_numba, NUMBA_AVAILABLE
from src.lj import total_energy, virial_pressure, delta_energy_particle_move, lj_shifted_energy
from src.lj_numba import (
    total_energy_numba,
    virial_pressure_numba,
    delta_energy_particle_move_numba,
)
from src.neighborlist import NeighborList
from src.neighborlist_numba import (
    build_neighbor_list_numba,
    total_energy_nl_numba,
    virial_pressure_nl_numba,
    delta_energy_particle_move_nl_numba,
)


@pytest.fixture(autouse=True)
def require_numba_fixture():
    """Require numba for all tests in this module."""
    require_numba("Neighbor list correctness tests")


def extract_pairs_from_neighbor_list(neighbor_list, neighbor_starts, N):
    """Extract all (i, j) pairs from neighbor list.
    
    The neighbor list stores pairs where j > i. Returns a set of (i, j) tuples.
    """
    pairs = set()
    for i in range(N):
        start = neighbor_starts[i]
        end = neighbor_starts[i + 1]
        for idx in range(start, end):
            j = neighbor_list[idx]
            if j > i:  # Neighbor list only stores j > i
                pairs.add((i, j))
    return pairs


def brute_force_pairs_within_cutoff(positions, L, rc2):
    """Find all pairs (i, j) with i < j and r_ij^2 < rc2 using brute force.
    
    Returns:
        Set of (i, j) tuples where i < j and r_ij^2 < rc2
    """
    N = positions.shape[0]
    pairs = set()
    for i in range(N - 1):
        for j in range(i + 1, N):
            dr = positions[j] - positions[i]
            dr = minimum_image(dr, L)
            r2 = np.dot(dr, dr)
            if r2 < rc2 and r2 > 0:
                pairs.add((i, j))
    return pairs


def brute_force_pairs_within_cutoff_extended(positions, L, rc2_extended):
    """Find all pairs (i, j) with i < j and r_ij^2 < rc2_extended using brute force.
    
    Used to check that no pairs beyond rc + skin appear in neighbor list.
    """
    N = positions.shape[0]
    pairs = set()
    for i in range(N - 1):
        for j in range(i + 1, N):
            dr = positions[j] - positions[i]
            dr = minimum_image(dr, L)
            r2 = np.dot(dr, dr)
            if r2 < rc2_extended and r2 > 0:
                pairs.add((i, j))
    return pairs


def test_pair_coverage():
    """Test that neighbor list includes all pairs with r < rc and no pairs beyond rc + skin."""
    rng = np.random.default_rng(42)
    N = 32
    L = 5.0
    rc = 2.5
    skin = 0.2
    rc2 = rc * rc
    cutoff2 = (rc + skin) ** 2
    
    # Test multiple configurations
    for config_idx in range(5):
        positions = init_lattice(N, L, rng)
        positions += rng.random((N, 3)) * 0.1
        positions = positions % L  # Wrap to [0, L)
        
        # Build neighbor list
        neighbor_list, neighbor_starts = build_neighbor_list_numba(positions, L, cutoff2)
        nl_pairs = extract_pairs_from_neighbor_list(neighbor_list, neighbor_starts, N)
        
        # Brute force: pairs with r < rc
        brute_pairs_rc = brute_force_pairs_within_cutoff(positions, L, rc2)
        
        # Brute force: pairs with r < rc + skin
        brute_pairs_extended = brute_force_pairs_within_cutoff_extended(positions, L, cutoff2)
        
        # Check: all pairs with r < rc must be in neighbor list
        missing_pairs = brute_pairs_rc - nl_pairs
        assert len(missing_pairs) == 0, (
            f"Config {config_idx}: Missing pairs with r < rc: {missing_pairs}"
        )
        
        # Check: no pairs beyond rc + skin should be in neighbor list
        extra_pairs = nl_pairs - brute_pairs_extended
        assert len(extra_pairs) == 0, (
            f"Config {config_idx}: Extra pairs beyond rc + skin: {extra_pairs}"
        )


def test_total_energy_equivalence():
    """Test that neighbor-list-based total energy matches brute force."""
    rng = np.random.default_rng(43)
    N = 32
    L = 5.0
    rc = 2.5
    skin = 0.2
    rc2 = rc * rc
    cutoff2 = (rc + skin) ** 2
    
    # Test multiple configurations
    for config_idx in range(5):
        positions = init_lattice(N, L, rng)
        positions += rng.random((N, 3)) * 0.1
        positions = positions % L
        
        # Brute force reference (Python)
        U_brute_python = total_energy(positions, L, rc)
        
        # Brute force reference (Numba)
        U_brute_numba = total_energy_numba(positions, L, rc)
        
        # Build neighbor list with cutoff2 = (rc + skin)^2
        neighbor_list, neighbor_starts = build_neighbor_list_numba(positions, L, cutoff2)
        
        # Neighbor list energy (uses rc2 = rc^2 for energy calculation)
        U_nl = total_energy_nl_numba(positions, neighbor_list, neighbor_starts, L, rc2)
        
        # All should match
        np.testing.assert_allclose(U_brute_python, U_brute_numba, rtol=0, atol=1e-12, err_msg=(
            f"Config {config_idx}: Brute force mismatch (Python vs Numba)"
        ))
        np.testing.assert_allclose(U_brute_python, U_nl, rtol=0, atol=1e-10, err_msg=(
            f"Config {config_idx}: Neighbor list energy mismatch. "
            f"Brute force={U_brute_python:.12f}, NL={U_nl:.12f}"
        ))


def test_virial_pressure_equivalence():
    """Test that neighbor-list-based virial pressure matches brute force."""
    rng = np.random.default_rng(44)
    N = 32
    L = 5.0
    rc = 2.5
    skin = 0.2
    rc2 = rc * rc
    cutoff2 = (rc + skin) ** 2
    T = 1.0
    
    # Test multiple configurations
    for config_idx in range(5):
        positions = init_lattice(N, L, rng)
        positions += rng.random((N, 3)) * 0.1
        positions = positions % L
        
        # Brute force reference (Python)
        P_brute_python = virial_pressure(positions, L, rc, T)
        
        # Brute force reference (Numba)
        P_brute_numba = virial_pressure_numba(positions, L, rc, T)
        
        # Build neighbor list with cutoff2 = (rc + skin)^2
        neighbor_list, neighbor_starts = build_neighbor_list_numba(positions, L, cutoff2)
        
        # Neighbor list pressure (uses rc2 = rc^2 for pressure calculation)
        P_nl = virial_pressure_nl_numba(positions, neighbor_list, neighbor_starts, L, rc, T, rc2)
        
        # All should match
        np.testing.assert_allclose(P_brute_python, P_brute_numba, rtol=0, atol=1e-12, err_msg=(
            f"Config {config_idx}: Brute force pressure mismatch (Python vs Numba)"
        ))
        np.testing.assert_allclose(P_brute_python, P_nl, rtol=0, atol=1e-10, err_msg=(
            f"Config {config_idx}: Neighbor list pressure mismatch. "
            f"Brute force={P_brute_python:.12f}, NL={P_nl:.12f}"
        ))


def test_delta_energy_equivalence():
    """Test that neighbor-list-based ΔU matches brute force for particle moves."""
    rng = np.random.default_rng(45)
    N = 32
    L = 5.0
    rc = 2.5
    skin = 0.2
    rc2 = rc * rc
    cutoff2 = (rc + skin) ** 2
    
    # Test multiple configurations and moves
    for config_idx in range(5):
        positions = init_lattice(N, L, rng)
        positions += rng.random((N, 3)) * 0.1
        positions = positions % L
        
        # Build neighbor list with cutoff2 = (rc + skin)^2
        neighbor_list, neighbor_starts = build_neighbor_list_numba(positions, L, cutoff2)
        
        # Test several random moves
        for move_idx in range(10):
            i = rng.integers(N)
            # Small displacement
            displacement = rng.random(3) * 0.1
            new_pos = (positions[i] + displacement) % L
            
            # Brute force reference (Python)
            dU_brute_python = delta_energy_particle_move(i, new_pos, positions, L, rc)
            
            # Brute force reference (Numba)
            dU_brute_numba = delta_energy_particle_move_numba(positions, i, new_pos, L, rc2)
            
            # Neighbor list (uses rc2 = rc^2 for energy calculation)
            dU_nl = delta_energy_particle_move_nl_numba(
                positions, i, new_pos, neighbor_list, neighbor_starts, L, rc2
            )
            
            # All should match
            np.testing.assert_allclose(dU_brute_python, dU_brute_numba, rtol=0, atol=1e-12, err_msg=(
                f"Config {config_idx}, move {move_idx}: Brute force ΔU mismatch (Python vs Numba)"
            ))
            np.testing.assert_allclose(dU_brute_python, dU_nl, rtol=0, atol=1e-10, err_msg=(
                f"Config {config_idx}, move {move_idx}: Neighbor list ΔU mismatch. "
                f"Brute force={dU_brute_python:.12f}, NL={dU_nl:.12f}"
            ))


def test_rebuild_logic_no_rebuild():
    """Test that neighbor list does NOT rebuild when displacements < skin/2."""
    rng = np.random.default_rng(46)
    N = 32
    L = 5.0
    rc = 2.5
    skin = 0.2
    
    positions = init_lattice(N, L, rng)
    positions += rng.random((N, 3)) * 0.1
    positions = positions % L
    
    # Build initial neighbor list
    nl = NeighborList(positions, L, rc, skin)
    initial_list = nl.neighbor_list.copy()
    initial_starts = nl.neighbor_starts.copy()
    
    # Make small displacements (< skin/2)
    max_displacement = skin / 4.0  # Well below threshold
    displacements = rng.random((N, 3)) * max_displacement
    new_positions = positions + displacements
    new_positions = new_positions % L  # Wrap
    
    # Update (should NOT rebuild)
    nl.update(new_positions, force_rebuild=False)
    
    # Check that neighbor list arrays are unchanged
    assert np.array_equal(nl.neighbor_list, initial_list), (
        "Neighbor list was rebuilt when it shouldn't have been (displacement < skin/2)"
    )
    assert np.array_equal(nl.neighbor_starts, initial_starts), (
        "Neighbor list starts were changed when rebuild shouldn't have occurred"
    )


def test_rebuild_logic_force_rebuild():
    """Test that neighbor list DOES rebuild when displacements > skin/2."""
    rng = np.random.default_rng(47)
    N = 32
    L = 5.0
    rc = 2.5
    skin = 0.2
    
    positions = init_lattice(N, L, rng)
    positions += rng.random((N, 3)) * 0.1
    positions = positions % L
    
    # Build initial neighbor list
    nl = NeighborList(positions, L, rc, skin)
    initial_list = nl.neighbor_list.copy()
    initial_starts = nl.neighbor_starts.copy()
    
    # Make large displacements (> skin/2)
    min_displacement = skin  # Well above threshold
    max_displacement = skin * 2.0
    # Create displacements that ensure at least one particle moves > skin/2
    displacements = np.zeros((N, 3))
    # First particle gets a large displacement
    displacements[0] = rng.random(3) * max_displacement
    # Ensure it's large enough
    displacements[0] = displacements[0] / np.linalg.norm(displacements[0]) * min_displacement * 1.1
    
    new_positions = positions + displacements
    new_positions = new_positions % L  # Wrap

    # Verify needs_rebuild returns True BEFORE update/rebuild
    assert nl.needs_rebuild(new_positions), (
        "needs_rebuild should return True for displacements > skin/2"
    )

    # Update (should rebuild)
    nl.update(new_positions, force_rebuild=False)

    # After rebuild, needs_rebuild should return False
    assert not nl.needs_rebuild(new_positions), (
        "After rebuild, needs_rebuild should return False"
    )


def test_rebuild_logic_force_rebuild_flag():
    """Test that force_rebuild=True always rebuilds regardless of displacement."""
    rng = np.random.default_rng(48)
    N = 32
    L = 5.0
    rc = 2.5
    skin = 0.2
    
    positions = init_lattice(N, L, rng)
    positions += rng.random((N, 3)) * 0.1
    positions = positions % L
    
    # Build initial neighbor list
    nl = NeighborList(positions, L, rc, skin)
    initial_list = nl.neighbor_list.copy()
    initial_starts = nl.neighbor_starts.copy()
    
    # Make tiny displacements (< skin/2)
    max_displacement = skin / 10.0  # Very small
    displacements = rng.random((N, 3)) * max_displacement
    new_positions = positions + displacements
    new_positions = new_positions % L
    
    # Verify needs_rebuild returns False
    assert not nl.needs_rebuild(new_positions), (
        "needs_rebuild should return False for tiny displacements"
    )
    
    # But force_rebuild=True should still rebuild
    nl.update(new_positions, force_rebuild=True)
    
    # After forced rebuild, needs_rebuild should return False
    assert not nl.needs_rebuild(new_positions), (
        "After forced rebuild, needs_rebuild should return False"
    )


def test_neighbor_list_energy_after_rebuild():
    """Test that energy calculation is correct after rebuild."""
    rng = np.random.default_rng(49)
    N = 32
    L = 5.0
    rc = 2.5
    skin = 0.2
    
    # Initial configuration
    positions1 = init_lattice(N, L, rng)
    positions1 += rng.random((N, 3)) * 0.1
    positions1 = positions1 % L
    
    # Build neighbor list
    nl = NeighborList(positions1, L, rc, skin)
    
    # Move particles significantly
    positions2 = positions1 + rng.random((N, 3)) * skin * 1.5
    positions2 = positions2 % L
    
    # Rebuild
    nl.rebuild(positions2)
    
    # Energy should match brute force
    U_brute = total_energy_numba(positions2, L, rc)
    U_nl = nl.total_energy(positions2, L, rc)
    
    assert np.allclose(U_brute, U_nl, rtol=1e-12, atol=1e-12), (
        f"Energy mismatch after rebuild: Brute force={U_brute:.12f}, NL={U_nl:.12f}"
    )


def test_neighbor_list_edge_cases():
    """Test edge cases: very small system, large cutoff, etc."""
    rng = np.random.default_rng(50)
    
    # Small system
    N_small = 8
    L_small = 3.0
    rc_small = 1.5
    skin_small = 0.1
    
    positions_small = rng.random((N_small, 3)) * L_small
    nl_small = NeighborList(positions_small, L_small, rc_small, skin_small)
    U_small_brute = total_energy_numba(positions_small, L_small, rc_small)
    U_small_nl = nl_small.total_energy(positions_small, L_small, rc_small)
    assert np.allclose(U_small_brute, U_small_nl, rtol=1e-12, atol=1e-12)
    
    # Large cutoff (most pairs included)
    N_large = 16
    L_large = 4.0
    rc_large = 3.5  # Large cutoff
    skin_large = 0.2
    
    positions_large = rng.random((N_large, 3)) * L_large
    nl_large = NeighborList(positions_large, L_large, rc_large, skin_large)
    U_large_brute = total_energy_numba(positions_large, L_large, rc_large)
    U_large_nl = nl_large.total_energy(positions_large, L_large, rc_large)
    assert np.allclose(U_large_brute, U_large_nl, rtol=1e-12, atol=1e-12)
