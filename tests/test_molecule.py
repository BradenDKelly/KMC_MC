"""Tests for RigidMolecule dataclass."""

import numpy as np
import pytest
from src.molecule import RigidMolecule
from src.rigid import uniform_random_orientation


def test_center_on_com_yields_com_zero():
    """Test that center_on_com() yields COM ≈ 0 (tight tolerance)."""
    rng = np.random.default_rng(42)
    
    # Create a molecule with non-zero COM
    n_sites = 5
    s_body = rng.normal(size=(n_sites, 3)) * 0.5
    mass = rng.random(n_sites) + 0.1  # Ensure positive
    charge = rng.normal(size=n_sites)
    sigma = rng.random(n_sites)
    epsilon = rng.random(n_sites)
    
    mol = RigidMolecule(
        s_body=s_body,
        mass=mass,
        charge=charge,
        sigma=sigma,
        epsilon=epsilon
    )
    
    # Center on COM
    mol_centered = mol.center_on_com()
    
    # Check COM is approximately zero
    com = mol_centered.com_body()
    np.testing.assert_allclose(
        com, np.zeros(3), rtol=1e-12, atol=1e-12,
        err_msg=f"COM should be zero after centering, got {com}"
    )


def test_site_positions_translation_invariance():
    """Test that site_positions() translation invariance: shifting R shifts all sites by same vector (no wrap, L=None)."""
    rng = np.random.default_rng(123)
    
    # Create a molecule
    n_sites = 4
    s_body = rng.normal(size=(n_sites, 3)) * 0.3
    mass = np.ones(n_sites)
    charge = np.zeros(n_sites)
    sigma = np.ones(n_sites)
    epsilon = np.ones(n_sites)
    
    mol = RigidMolecule(
        s_body=s_body,
        mass=mass,
        charge=charge,
        sigma=sigma,
        epsilon=epsilon
    )
    
    # Initial position and orientation
    R0 = np.array([5.0, 5.0, 5.0])
    q = uniform_random_orientation(rng)
    
    # Get initial positions
    r0 = mol.site_positions(R0, q, L=None)
    
    # Translate COM
    translation = rng.normal(size=3) * 2.0
    R1 = R0 + translation
    
    # Get new positions
    r1 = mol.site_positions(R1, q, L=None)
    
    # All sites should be shifted by the same translation vector
    dr = r1 - r0
    expected_dr = np.broadcast_to(translation, (n_sites, 3))
    
    np.testing.assert_allclose(
        dr, expected_dr, rtol=1e-10, atol=1e-10,
        err_msg="All sites should shift by the same translation vector"
    )


def test_site_positions_rotation_invariance_distances():
    """Test rotation invariance: distances between sites unchanged after rotation (L=None)."""
    rng = np.random.default_rng(456)
    
    # Create a molecule
    n_sites = 6
    s_body = rng.normal(size=(n_sites, 3)) * 0.4
    mass = np.ones(n_sites)
    charge = np.zeros(n_sites)
    sigma = np.ones(n_sites)
    epsilon = np.ones(n_sites)
    
    mol = RigidMolecule(
        s_body=s_body,
        mass=mass,
        charge=charge,
        sigma=sigma,
        epsilon=epsilon
    )
    
    # Initial position and orientation
    R = np.array([5.0, 5.0, 5.0])
    q0 = uniform_random_orientation(rng)
    
    # Get initial positions
    r0 = mol.site_positions(R, q0, L=None)
    
    # Compute pairwise distances
    distances0 = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            dist = np.linalg.norm(r0[i] - r0[j])
            distances0.append(dist)
    distances0 = np.array(distances0)
    
    # Rotate molecule
    q1 = uniform_random_orientation(rng)
    r1 = mol.site_positions(R, q1, L=None)
    
    # Compute pairwise distances after rotation
    distances1 = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            dist = np.linalg.norm(r1[i] - r1[j])
            distances1.append(dist)
    distances1 = np.array(distances1)
    
    # Distances should be unchanged (rotation preserves distances)
    np.testing.assert_allclose(
        distances0, distances1, rtol=1e-10, atol=1e-10,
        err_msg="Pairwise distances should be unchanged after rotation"
    )


def test_validation_wrong_shapes_raise_valueerror():
    """Test that validation raises ValueError for wrong shapes."""
    rng = np.random.default_rng(789)
    
    # Valid base molecule
    n_sites = 3
    s_body = rng.normal(size=(n_sites, 3))
    mass = np.ones(n_sites)
    charge = np.zeros(n_sites)
    sigma = np.ones(n_sites)
    epsilon = np.ones(n_sites)
    
    # Test wrong s_body shape (wrong number of dimensions)
    with pytest.raises(ValueError, match="s_body must have shape"):
        RigidMolecule(
            s_body=s_body.ravel(),  # Flattened, wrong shape
            mass=mass,
            charge=charge,
            sigma=sigma,
            epsilon=epsilon
        )
    
    # Test wrong s_body shape (wrong second dimension)
    with pytest.raises(ValueError, match="s_body must have shape"):
        RigidMolecule(
            s_body=s_body[:, :2],  # Only 2D, wrong shape
            mass=mass,
            charge=charge,
            sigma=sigma,
            epsilon=epsilon
        )
    
    # Test wrong mass shape
    with pytest.raises(ValueError, match="mass must have shape"):
        RigidMolecule(
            s_body=s_body,
            mass=mass[:2],  # Wrong size
            charge=charge,
            sigma=sigma,
            epsilon=epsilon
        )
    
    # Test wrong charge shape
    with pytest.raises(ValueError, match="charge must have shape"):
        RigidMolecule(
            s_body=s_body,
            mass=mass,
            charge=charge[:2],  # Wrong size
            sigma=sigma,
            epsilon=epsilon
        )
    
    # Test wrong sigma shape
    with pytest.raises(ValueError, match="sigma must have shape"):
        RigidMolecule(
            s_body=s_body,
            mass=mass,
            charge=charge,
            sigma=sigma[:2],  # Wrong size
            epsilon=epsilon
        )
    
    # Test wrong epsilon shape
    with pytest.raises(ValueError, match="epsilon must have shape"):
        RigidMolecule(
            s_body=s_body,
            mass=mass,
            charge=charge,
            sigma=sigma,
            epsilon=epsilon[:2]  # Wrong size
        )


def test_validation_mass_positive():
    """Test that validation raises ValueError for non-positive masses."""
    rng = np.random.default_rng(999)
    
    n_sites = 3
    s_body = rng.normal(size=(n_sites, 3))
    mass = np.array([1.0, 0.0, 1.0])  # Zero mass
    charge = np.zeros(n_sites)
    sigma = np.ones(n_sites)
    epsilon = np.ones(n_sites)
    
    with pytest.raises(ValueError, match="All masses must be > 0"):
        RigidMolecule(
            s_body=s_body,
            mass=mass,
            charge=charge,
            sigma=sigma,
            epsilon=epsilon
        )
    
    # Test negative mass
    mass = np.array([1.0, -0.1, 1.0])
    with pytest.raises(ValueError, match="All masses must be > 0"):
        RigidMolecule(
            s_body=s_body,
            mass=mass,
            charge=charge,
            sigma=sigma,
            epsilon=epsilon
        )


def test_com_body_mass_weighted():
    """Test that com_body() correctly computes mass-weighted COM."""
    # Simple test case: two sites with different masses
    s_body = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    mass = np.array([1.0, 3.0])  # Heavier mass at second site
    charge = np.zeros(2)
    sigma = np.ones(2)
    epsilon = np.ones(2)
    
    mol = RigidMolecule(
        s_body=s_body,
        mass=mass,
        charge=charge,
        sigma=sigma,
        epsilon=epsilon
    )
    
    com = mol.com_body()
    # Expected: (1*[0,0,0] + 3*[1,0,0]) / 4 = [0.75, 0, 0]
    expected_com = np.array([0.75, 0.0, 0.0])
    
    np.testing.assert_allclose(
        com, expected_com, rtol=1e-10, atol=1e-10,
        err_msg="COM should be mass-weighted"
    )


def test_center_on_com_preserves_other_fields():
    """Test that center_on_com() preserves mass, charge, sigma, epsilon."""
    rng = np.random.default_rng(111)
    
    n_sites = 4
    s_body = rng.normal(size=(n_sites, 3))
    mass = rng.random(n_sites) + 0.1
    charge = rng.normal(size=n_sites)
    sigma = rng.random(n_sites)
    epsilon = rng.random(n_sites)
    
    mol = RigidMolecule(
        s_body=s_body,
        mass=mass,
        charge=charge,
        sigma=sigma,
        epsilon=epsilon
    )
    
    mol_centered = mol.center_on_com()
    
    # Check that other fields are preserved
    np.testing.assert_array_equal(mol_centered.mass, mass)
    np.testing.assert_array_equal(mol_centered.charge, charge)
    np.testing.assert_array_equal(mol_centered.sigma, sigma)
    np.testing.assert_array_equal(mol_centered.epsilon, epsilon)
    
    # Check that s_body is different (centered)
    assert not np.allclose(mol_centered.s_body, s_body), "s_body should be different after centering"

