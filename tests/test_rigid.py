"""Tests for rigid-body utilities."""

import numpy as np
import pytest
from src.rigid import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_rotate_vector,
    uniform_random_orientation,
    apply_rigid_transform,
    rigid_body_move_proposal,
)


def test_rotation_preserves_vector_length():
    """Test that rotation preserves vector length."""
    rng = np.random.default_rng(42)
    
    # Test with multiple random vectors and rotations
    for _ in range(10):
        # Random vector
        v = rng.normal(size=3)
        v_len = np.linalg.norm(v)
        
        # Random unit quaternion
        q = uniform_random_orientation(rng)
        
        # Rotate vector
        v_rotated = quaternion_rotate_vector(q, v)
        v_rotated_len = np.linalg.norm(v_rotated)
        
        # Length should be preserved
        np.testing.assert_allclose(
            v_len, v_rotated_len, rtol=1e-10, atol=1e-10,
            err_msg=f"Vector length not preserved: {v_len} vs {v_rotated_len}"
        )


def test_quaternion_normalization_stays_one():
    """Test that quaternion normalization stays ~1."""
    rng = np.random.default_rng(123)
    
    # Test with multiple random quaternions
    for _ in range(20):
        # Random quaternion (not necessarily normalized)
        q = rng.normal(size=4)
        
        # Normalize
        q_norm = quaternion_normalize(q)
        
        # Check norm is 1
        norm = np.linalg.norm(q_norm)
        np.testing.assert_allclose(
            norm, 1.0, rtol=1e-10, atol=1e-10,
            err_msg=f"Normalized quaternion norm is not 1: {norm}"
        )


def test_small_rotation_does_not_change_com_translation():
    """Test that small rotation does not change COM translation.
    
    This verifies that rotation alone (without translation) doesn't affect
    the center of mass position. We apply a rotation to a body and check
    that the COM remains at the same location.
    """
    rng = np.random.default_rng(456)
    
    # Initial COM and orientation
    R0 = np.array([5.0, 5.0, 5.0])
    q0 = uniform_random_orientation(rng)
    
    # Body-frame sites (relative to COM)
    s_body = np.array([
        [0.1, 0.0, 0.0],
        [-0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
    ])
    s_body = s_body - np.mean(s_body, axis=0)

    # Apply small rotation (no translation)
    max_angle = 0.01  # Very small angle
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    angle = rng.random() * max_angle
    
    half_angle = 0.5 * angle
    q_rot = np.array([
        np.cos(half_angle),
        np.sin(half_angle) * axis[0],
        np.sin(half_angle) * axis[1],
        np.sin(half_angle) * axis[2]
    ])
    q1 = quaternion_normalize(quaternion_multiply(q_rot, q0))
    
    # Compute lab coordinates before and after rotation
    L = 10.0
    r0 = apply_rigid_transform(R0, q0, s_body, L=None)
    r1 = apply_rigid_transform(R0, q1, s_body, L=None)

    
    # COM should be the same (mean of lab coordinates)
    com0 = np.mean(r0, axis=0)
    com1 = np.mean(r1, axis=0)
    
    # Account for periodic wrapping
    dr_com = com1 - com0
    dr_com = dr_com - L * np.round(dr_com / L)
    
    # COM should be unchanged (within numerical precision)
    np.testing.assert_allclose(
        dr_com, np.zeros(3), rtol=1e-10, atol=1e-6,
        err_msg=f"COM changed after small rotation: {dr_com}"
    )


def test_uniform_orientation_sampler_sanity():
    """Test uniform orientation sampler: mean of rotated unit vector components ~0.
    
    For a uniform distribution on SO(3), if we rotate a fixed unit vector
    by many random orientations, the mean of the rotated vector components
    should be approximately zero (within tolerance).
    """
    rng = np.random.default_rng(789)
    
    # Fixed unit vector (e.g., z-axis)
    v_fixed = np.array([0.0, 0.0, 1.0])
    
    # Sample many orientations
    n_samples = 10000
    rotated_vectors = np.zeros((n_samples, 3))
    
    for i in range(n_samples):
        q = uniform_random_orientation(rng)
        rotated_vectors[i] = quaternion_rotate_vector(q, v_fixed)
    
    # Compute mean of each component
    mean_components = np.mean(rotated_vectors, axis=0)
    
    # Mean should be approximately zero (within tolerance)
    # For uniform distribution, we expect mean ~ 0 with std ~ 1/sqrt(n_samples)
    tolerance = 3.0 / np.sqrt(n_samples)  # 3-sigma tolerance
    
    np.testing.assert_allclose(
        mean_components, np.zeros(3), rtol=0, atol=tolerance,
        err_msg=f"Mean of rotated unit vector components not ~0: {mean_components} "
                f"(tolerance: {tolerance})"
    )


def test_quaternion_multiply_identity():
    """Test that quaternion multiplication with identity works."""
    rng = np.random.default_rng(111)
    
    # Identity quaternion (no rotation)
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    
    for _ in range(5):
        q = uniform_random_orientation(rng)
        
        # q * identity = q
        q1 = quaternion_multiply(q, q_identity)
        np.testing.assert_allclose(q, q1, rtol=1e-10, atol=1e-10)
        
        # identity * q = q
        q2 = quaternion_multiply(q_identity, q)
        np.testing.assert_allclose(q, q2, rtol=1e-10, atol=1e-10)


def test_quaternion_conjugate_inverse():
    """Test that quaternion conjugate acts as inverse for unit quaternions."""
    rng = np.random.default_rng(222)
    
    for _ in range(5):
        q = uniform_random_orientation(rng)
        q_conj = quaternion_conjugate(q)
        
        # q * q* should give identity (for unit quaternions)
        q_product = quaternion_multiply(q, q_conj)
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        
        np.testing.assert_allclose(
            q_product, q_identity, rtol=1e-10, atol=1e-10,
            err_msg="q * q* should equal identity for unit quaternions"
        )


def test_apply_rigid_transform_periodic_wrap():
    """Test that apply_rigid_transform properly wraps coordinates."""
    rng = np.random.default_rng(333)
    
    L = 10.0
    R = np.array([9.5, 9.5, 9.5])  # Near boundary
    q = uniform_random_orientation(rng)
    
    # Body-frame sites
    s_body = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    
    r_lab = apply_rigid_transform(R, q, s_body, L)
    
    # All coordinates should be in [0, L)
    assert np.all(r_lab >= 0), "Coordinates should be >= 0"
    assert np.all(r_lab < L), "Coordinates should be < L"


def test_rigid_body_move_proposal():
    """Test that rigid_body_move_proposal produces valid moves."""
    rng = np.random.default_rng(444)
    
    R0 = np.array([5.0, 5.0, 5.0])
    q0 = uniform_random_orientation(rng)
    max_disp = 0.1
    max_angle = 0.1
    
    R_new, q_new = rigid_body_move_proposal(R0, q0, max_disp, max_angle, rng)
    
    # Check that quaternion is normalized
    q_norm = np.linalg.norm(q_new)
    np.testing.assert_allclose(q_norm, 1.0, rtol=1e-10, atol=1e-10)
    
    # Check that displacement is within bounds
    disp = R_new - R0
    max_actual_disp = np.max(np.abs(disp))
    assert max_actual_disp <= max_disp * np.sqrt(3), "Displacement exceeds max_disp"

