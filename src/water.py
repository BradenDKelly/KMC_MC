"""SPC/E rigid water molecule factory."""

import numpy as np
from .molecule import RigidMolecule


def make_spce_water():
    """Create an SPC/E rigid water molecule as a RigidMolecule instance.
    
    Returns a RigidMolecule with 3 sites (O, H1, H2) in body frame.
    The molecule is centered on its mass-weighted center of mass.
    
    Geometry (body frame):
    - O-H bond length: 0.1 nm
    - H-O-H angle: 109.47° (tetrahedral angle)
    - O initially at origin, H atoms in x-z plane symmetrically
    - Final positions are centered on COM
    
    Parameters:
    - Charges: O = -0.8476 e, H = +0.4238 e each (net zero)
    - LJ: only on oxygen (σ = 0.316555789 nm, ε = 0.650 kJ/mol)
    - Masses: O = 15.9994 amu, H = 1.008 amu each
    
    Units:
    - Distances: nanometers (nm)
    - Charges: elementary charge units (e)
    - LJ σ: nanometers (nm)
    - LJ ε: kJ/mol
    - Masses: atomic mass units (amu)
    
    Returns:
        RigidMolecule: SPC/E water molecule with COM at origin
    """
    # SPC/E geometry parameters (in nm)
    bond_length = 0.1  # O-H bond length in nm
    angle_deg = 109.47  # H-O-H angle in degrees
    angle_rad = np.deg2rad(angle_deg)  # Convert to radians
    half_angle = angle_rad / 2.0
    
    # Place O at origin
    O_pos = np.array([0.0, 0.0, 0.0])
    
    # Place H atoms in x-z plane symmetrically
    # H1: at angle +half_angle from z-axis in x-z plane
    # H2: at angle -half_angle from z-axis in x-z plane
    H1_pos = np.array([
        bond_length * np.sin(half_angle),
        0.0,
        bond_length * np.cos(half_angle)
    ])
    H2_pos = np.array([
    -bond_length * np.sin(half_angle),
    0.0,
    bond_length * np.cos(half_angle)
    ])

    
    # Combine positions
    s_body = np.array([O_pos, H1_pos, H2_pos])
    
    # SPC/E charges (in elementary charge units)
    q_O = -0.8476
    q_H = 0.4238
    charge = np.array([q_O, q_H, q_H])
    
    # Atomic masses (in amu)
    mass_O = 15.9994
    mass_H = 1.008
    mass = np.array([mass_O, mass_H, mass_H])
    
    # SPC/E LJ parameters (only on oxygen)
    # σ in nm, ε in kJ/mol
    sigma_O = 0.316555789  # nm
    epsilon_O = 0.650  # kJ/mol (converted from ε/k_B = 78.19743111 K)
    sigma = np.array([sigma_O, 0.0, 0.0])  # H sites have zero LJ
    epsilon = np.array([epsilon_O, 0.0, 0.0])  # H sites have zero LJ
    
    # Create molecule
    mol = RigidMolecule(
        s_body=s_body,
        mass=mass,
        charge=charge,
        sigma=sigma,
        epsilon=epsilon
    )
    
    # Center on COM
    mol = mol.center_on_com()
    
    return mol

