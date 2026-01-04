"""Tests for SPC/E water molecule factory."""

import numpy as np
import pytest
from src.water import make_spce_water


def test_oh_distances_match_target():
    """Test that O-H distances match target within tight tolerance.
    
    SPC/E O-H bond length: 0.1 nm
    """
    water = make_spce_water()
    
    # Get positions (O is index 0, H1 is index 1, H2 is index 2)
    O_pos = water.s_body[0]
    H1_pos = water.s_body[1]
    H2_pos = water.s_body[2]
    
    # Compute O-H distances
    OH1_dist = np.linalg.norm(H1_pos - O_pos)
    OH2_dist = np.linalg.norm(H2_pos - O_pos)
    
    # Target bond length: 0.1 nm
    target_bond_length = 0.1  # nm
    
    # Check both distances match target (tight tolerance)
    np.testing.assert_allclose(
        OH1_dist, target_bond_length, rtol=1e-10, atol=1e-10,
        err_msg=f"O-H1 distance ({OH1_dist} nm) does not match target ({target_bond_length} nm)"
    )
    np.testing.assert_allclose(
        OH2_dist, target_bond_length, rtol=1e-10, atol=1e-10,
        err_msg=f"O-H2 distance ({OH2_dist} nm) does not match target ({target_bond_length} nm)"
    )


def test_hoh_angle_matches_target():
    """Test that H-O-H angle matches target within tight tolerance.
    
    SPC/E H-O-H angle: 109.47° (tetrahedral angle)
    """
    water = make_spce_water()
    
    # Get positions
    O_pos = water.s_body[0]
    H1_pos = water.s_body[1]
    H2_pos = water.s_body[2]
    
    # Compute vectors from O to H atoms
    OH1_vec = H1_pos - O_pos
    OH2_vec = H2_pos - O_pos
    
    # Normalize vectors
    OH1_vec_norm = OH1_vec / np.linalg.norm(OH1_vec)
    OH2_vec_norm = OH2_vec / np.linalg.norm(OH2_vec)
    
    # Compute angle using dot product: cos(θ) = (v1 · v2) / (|v1| |v2|)
    cos_angle = np.dot(OH1_vec_norm, OH2_vec_norm)
    # Clamp to avoid numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.rad2deg(angle_rad)
    
    # Target angle: 109.47°
    target_angle_deg = 109.47
    
    # Check angle matches target (tight tolerance)
    np.testing.assert_allclose(
        angle_deg, target_angle_deg, rtol=1e-6, atol=1e-6,
        err_msg=f"H-O-H angle ({angle_deg}°) does not match target ({target_angle_deg}°)"
    )


def test_net_charge_is_zero():
    """Test that net charge is ~0."""
    water = make_spce_water()
    
    # Compute net charge
    net_charge = np.sum(water.charge)
    
    # Net charge should be approximately zero
    np.testing.assert_allclose(
        net_charge, 0.0, rtol=1e-10, atol=1e-10,
        err_msg=f"Net charge ({net_charge} e) is not zero"
    )


def test_center_on_com_mass_weighted_com_is_zero():
    """Test that after center_on_com(), mass-weighted COM is ~0."""
    water = make_spce_water()
    
    # Compute mass-weighted COM
    com = water.com_body()
    
    # COM should be approximately zero (tight tolerance)
    np.testing.assert_allclose(
        com, np.zeros(3), rtol=1e-12, atol=1e-12,
        err_msg=f"Mass-weighted COM ({com}) is not zero after centering"
    )


def test_spce_water_parameters():
    """Test that SPC/E water has correct parameters."""
    water = make_spce_water()
    
    # Check charges
    q_O = water.charge[0]
    q_H1 = water.charge[1]
    q_H2 = water.charge[2]
    
    np.testing.assert_allclose(q_O, -0.8476, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q_H1, 0.4238, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q_H2, 0.4238, rtol=1e-10, atol=1e-10)
    
    # Check LJ parameters (only on O)
    sigma_O = water.sigma[0]
    epsilon_O = water.epsilon[0]
    sigma_H1 = water.sigma[1]
    epsilon_H1 = water.epsilon[1]
    sigma_H2 = water.sigma[2]
    epsilon_H2 = water.epsilon[2]
    
    np.testing.assert_allclose(sigma_O, 0.316555789, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(epsilon_O, 0.650, rtol=1e-6, atol=1e-6)
    assert sigma_H1 == 0.0, "H1 sigma should be zero"
    assert epsilon_H1 == 0.0, "H1 epsilon should be zero"
    assert sigma_H2 == 0.0, "H2 sigma should be zero"
    assert epsilon_H2 == 0.0, "H2 epsilon should be zero"
    
    # Check masses
    mass_O = water.mass[0]
    mass_H1 = water.mass[1]
    mass_H2 = water.mass[2]
    
    np.testing.assert_allclose(mass_O, 15.9994, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(mass_H1, 1.008, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(mass_H2, 1.008, rtol=1e-6, atol=1e-6)


def test_spce_water_shape():
    """Test that SPC/E water has correct shape."""
    water = make_spce_water()
    
    # Should have 3 sites
    assert water.s_body.shape == (3, 3), "s_body should have shape (3, 3)"
    assert water.mass.shape == (3,), "mass should have shape (3,)"
    assert water.charge.shape == (3,), "charge should have shape (3,)"
    assert water.sigma.shape == (3,), "sigma should have shape (3,)"
    assert water.epsilon.shape == (3,), "epsilon should have shape (3,)"

