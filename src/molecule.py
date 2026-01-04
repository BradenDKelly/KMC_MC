"""Rigid molecule container for Monte Carlo simulations."""

from dataclasses import dataclass
import numpy as np
from .rigid import apply_rigid_transform


@dataclass
class RigidMolecule:
    """Container for a rigid molecule with multiple interaction sites.
    
    Attributes:
        s_body: Site positions in body frame, shape (n, 3)
        mass: Masses for each site, shape (n,)
        charge: Charges for each site, shape (n,)
        sigma: Lennard-Jones sigma parameters for each site, shape (n,)
        epsilon: Lennard-Jones epsilon parameters for each site, shape (n,)
    """
    s_body: np.ndarray
    mass: np.ndarray
    charge: np.ndarray
    sigma: np.ndarray
    epsilon: np.ndarray
    
    def __post_init__(self):
        """Validate shapes, dtypes, and constraints."""
        # Convert to numpy arrays if not already
        self.s_body = np.asarray(self.s_body, dtype=np.float64)
        self.mass = np.asarray(self.mass, dtype=np.float64)
        self.charge = np.asarray(self.charge, dtype=np.float64)
        self.sigma = np.asarray(self.sigma, dtype=np.float64)
        self.epsilon = np.asarray(self.epsilon, dtype=np.float64)
        
        # Get number of sites
        n = self.s_body.shape[0]
        
        # Validate shapes
        if self.s_body.ndim != 2 or self.s_body.shape[1] != 3:
            raise ValueError(f"s_body must have shape (n, 3), got {self.s_body.shape}")
        
        expected_shape = (n,)
        for name, arr in [("mass", self.mass), ("charge", self.charge), 
                          ("sigma", self.sigma), ("epsilon", self.epsilon)]:
            if arr.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
        
        # Validate mass > 0
        if np.any(self.mass <= 0):
            raise ValueError("All masses must be > 0")
    
    def com_body(self):
        """Compute mass-weighted center of mass in body frame.
        
        Returns:
            Center of mass position, shape (3,)
        """
        total_mass = np.sum(self.mass)
        if total_mass == 0:
            raise ValueError("Total mass is zero")
        return np.sum(self.mass[:, None] * self.s_body, axis=0) / total_mass
    
    def center_on_com(self):
        """Return a new RigidMolecule with s_body shifted so COM = 0.
        
        Returns:
            New RigidMolecule instance with centered body-frame coordinates
        """
        com = self.com_body()
        s_body_centered = self.s_body - com
        return RigidMolecule(
            s_body=s_body_centered,
            mass=self.mass.copy(),
            charge=self.charge.copy(),
            sigma=self.sigma.copy(),
            epsilon=self.epsilon.copy()
        )
    
    def site_positions(self, R, q, L=None):
        """Compute lab-frame positions of all sites.
        
        Args:
            R: Center of mass position in lab frame, shape (3,)
            q: Orientation quaternion, shape (4,)
            L: Box length for periodic wrapping (None for no wrapping)
            
        Returns:
            Lab-frame site positions, shape (n, 3)
        """
        return apply_rigid_transform(R, q, self.s_body, L)

