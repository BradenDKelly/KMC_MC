"""Tests for water box with rigid SPC/E water molecules."""

import numpy as np
import pytest
from copy import deepcopy
from src.waterbox import (
    WaterBox,
    water_sites_lab,
    flatten_sites,
    oxygen_positions,
    total_energy_waterbox,
    delta_energy_rigid_move,
    apply_rigid_move,
)
from src.water import make_spce_water
from src.ewald import EwaldParams, build_cache, ewald_energy_total
from src.rigid import (
    uniform_random_orientation,
    rigid_body_move_proposal,
    quaternion_normalize,
)
from src.lj import lj_shifted_energy
from src.utils import minimum_image


def create_test_waterbox(N=8, seed=42):
    """Create a test water box with random positions and orientations."""
    rng = np.random.default_rng(seed)
    
    # Box parameters
    L = 10.0  # nm
    
    # Generate random COM positions
    R = rng.random((N, 3)) * L
    
    # Generate random orientations
    q = np.zeros((N, 4))
    for i in range(N):
        q[i] = uniform_random_orientation(rng)
    
    # Create water molecule
    water = make_spce_water()
    
    # Ewald parameters
    alpha = 0.35
    rcut = 4.5
    kmax = 5
    ewald_params = EwaldParams(L=L, alpha=alpha, rcut=rcut, kmax=kmax, ke=1.0)
    
    # Build Ewald cache
    flat_pos, flat_charges, _ = flatten_sites(R, q, water, L)
    ewald_cache = build_cache(flat_pos, flat_charges, ewald_params)
    
    return WaterBox(
        L=L,
        R=R,
        q=q,
        water=water,
        ewald_params=ewald_params,
        ewald_cache=ewald_cache
    )


def test_water_sites_lab():
    """Test water_sites_lab returns correct shape."""
    water = make_spce_water()
    rng = np.random.default_rng(123)
    
    R = np.array([5.0, 5.0, 5.0])
    q = uniform_random_orientation(rng)
    L = 10.0
    
    sites = water_sites_lab(R, q, water, L)
    
    assert sites.shape == (3, 3), "sites should have shape (3, 3)"
    assert np.all(sites >= 0), "sites should be >= 0"
    assert np.all(sites < L), "sites should be < L"


def test_flatten_sites():
    """Test flatten_sites returns correct arrays."""
    water = make_spce_water()
    rng = np.random.default_rng(456)
    
    N = 5
    L = 10.0
    R = rng.random((N, 3)) * L
    q = np.zeros((N, 4))
    for i in range(N):
        q[i] = uniform_random_orientation(rng)
    
    positions, charges, mol_to_sites = flatten_sites(R, q, water, L)
    
    assert positions.shape == (3*N, 3), f"positions should have shape ({3*N}, 3)"
    assert charges.shape == (3*N,), f"charges should have shape ({3*N},)"
    assert len(mol_to_sites) == N, "mol_to_sites should have length N"
    
    # Check net charge is zero
    net_charge = np.sum(charges)
    np.testing.assert_allclose(net_charge, 0.0, rtol=1e-10, atol=1e-10)


def test_oxygen_positions():
    """Test oxygen_positions returns correct shape."""
    water = make_spce_water()
    rng = np.random.default_rng(789)
    
    N = 5
    L = 10.0
    R = rng.random((N, 3)) * L
    q = np.zeros((N, 4))
    for i in range(N):
        q[i] = uniform_random_orientation(rng)
    
    O_pos = oxygen_positions(R, q, water, L)
    
    assert O_pos.shape == (N, 3), f"O_pos should have shape ({N}, 3)"


def test_total_energy_waterbox():
    """Test that total_energy_waterbox returns finite energy."""
    box = create_test_waterbox(N=8, seed=1000)
    
    U_total = total_energy_waterbox(box)
    
    assert np.isfinite(U_total), "Total energy should be finite"


def test_delta_energy_matches_total_recompute():
    """Test that delta energy matches total recomputation for rigid moves.
    
    For multiple random rigid moves, verify ΔU_incremental ≈ U_total(new)-U_total(old).
    """
    rng = np.random.default_rng(2000)
    
    # Create water box
    box = create_test_waterbox(N=8, seed=2000)
    
    # Test multiple moves
    n_moves = 20
    max_disp = 0.2  # nm
    max_angle = 0.1  # radians
    
    for move_idx in range(n_moves):
        # Choose random molecule
        m = rng.integers(box.R.shape[0])
        
        # Propose move
        R_old = box.R[m].copy()
        q_old = box.q[m].copy()
        
        # Translation
        disp = (rng.random(3) * 2 - 1) * max_disp
        R_new = (R_old + disp) % box.L
        
        # Rotation: small random rotation
        # Generate random axis and angle
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = rng.random() * max_angle
        half_angle = 0.5 * angle
        from src.rigid import quaternion_multiply
        q_rot = np.array([
            np.cos(half_angle),
            np.sin(half_angle) * axis[0],
            np.sin(half_angle) * axis[1],
            np.sin(half_angle) * axis[2]
        ])
        q_new = quaternion_multiply(q_rot, q_old)
        q_new = quaternion_normalize(q_new)
        
        # Compute incremental delta
        dU_incremental = delta_energy_rigid_move(box, m, R_new, q_new)
        
        # Compute total energy before move
        U_old = total_energy_waterbox(box)
        
        # Make a copy and apply move
        box_copy = deepcopy(box)
        apply_rigid_move(box_copy, m, R_new, q_new)
        
        # Compute total energy after move (recompute everything)
        # Rebuild cache for accurate comparison
        flat_pos, flat_charges, _ = flatten_sites(
            box_copy.R, box_copy.q, box_copy.water, box_copy.L
        )
        box_copy.ewald_cache = build_cache(flat_pos, flat_charges, box_copy.ewald_params)
        
        U_new = total_energy_waterbox(box_copy)
        
        # Delta from total recomputation
        dU_total = U_new - U_old
        
        # Verify they match (tolerance ~1e-6 to 1e-4)
        np.testing.assert_allclose(
            dU_incremental, dU_total, rtol=1e-4, atol=1e-4,
            err_msg=f"Move {move_idx}: Incremental delta ({dU_incremental}) does not match "
                    f"total recomputation ({dU_total})"
        )
        
        # Apply move to original box for next iteration
        apply_rigid_move(box, m, R_new, q_new)


def test_cache_after_apply_move():
    """Test that cache S matches recomputed S after apply_rigid_move.
    
    Spot-check: after applying move via cache update, verify cached S matches
    recomputed S occasionally.
    """
    rng = np.random.default_rng(3000)
    
    # Create water box
    box = create_test_waterbox(N=8, seed=3000)
    
    # Test multiple moves with spot-checks
    n_moves = 15
    check_every = 3  # Check every 3 moves
    max_disp = 0.2
    max_angle = 0.1
    
    for move_idx in range(n_moves):
        # Choose random molecule
        m = rng.integers(box.R.shape[0])
        
        # Propose move
        R_old = box.R[m].copy()
        q_old = box.q[m].copy()
        
        disp = (rng.random(3) * 2 - 1) * max_disp
        R_new = (R_old + disp) % box.L
        
        # Rotation: small random rotation
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = rng.random() * max_angle
        half_angle = 0.5 * angle
        from src.rigid import quaternion_multiply
        q_rot = np.array([
            np.cos(half_angle),
            np.sin(half_angle) * axis[0],
            np.sin(half_angle) * axis[1],
            np.sin(half_angle) * axis[2]
        ])
        q_new = quaternion_multiply(q_rot, q_old)
        q_new = quaternion_normalize(q_new)
        
        # Apply move
        apply_rigid_move(box, m, R_new, q_new)
        
        # Spot-check: recompute cache and compare
        if move_idx % check_every == 0:
            flat_pos, flat_charges, _ = flatten_sites(box.R, box.q, box.water, box.L)
            cache_recomputed = build_cache(flat_pos, flat_charges, box.ewald_params)
            
            # Compare cached S with recomputed S
            np.testing.assert_allclose(
                box.ewald_cache.S, cache_recomputed.S, rtol=1e-10, atol=1e-10,
                err_msg=f"After move {move_idx}: Cached S does not match recomputed S"
            )


def test_waterbox_creation():
    """Test that WaterBox can be created and has correct structure."""
    box = create_test_waterbox(N=8, seed=4000)
    
    assert box.R.shape[0] == 8, "Should have 8 molecules"
    assert box.R.shape[1] == 3, "R should have 3 components"
    assert box.q.shape[0] == 8, "Should have 8 quaternions"
    assert box.q.shape[1] == 4, "q should have 4 components"
    assert box.L > 0, "Box length should be positive"
    
    # Check quaternions are normalized
    for i in range(box.q.shape[0]):
        q_norm = np.linalg.norm(box.q[i])
        np.testing.assert_allclose(q_norm, 1.0, rtol=1e-10, atol=1e-10)
    
    # Check positions are in [0, L)
    assert np.all(box.R >= 0), "All positions should be >= 0"
    assert np.all(box.R < box.L), "All positions should be < L"

