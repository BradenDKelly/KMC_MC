"""Rigid SPC/E water box with LJ and Ewald electrostatics."""

from dataclasses import dataclass
import numpy as np
from math import erfc, sqrt
from .rigid import apply_rigid_transform, quaternion_normalize
from .lj import lj_shifted_energy
from .ewald import EwaldParams, EwaldCache, ewald_energy_total, build_cache
from .water import make_spce_water
from .utils import minimum_image


@dataclass
class WaterBox:
    """Container for a box of rigid SPC/E water molecules.
    
    Attributes:
        L: Box length
        R: COM positions, shape (N, 3), in [0, L)
        q: Quaternions, shape (N, 4), normalized
        water: RigidMolecule instance (SPC/E water)
        ewald_params: EwaldParams instance
        ewald_cache: EwaldCache instance
    """
    L: float
    R: np.ndarray
    q: np.ndarray
    water: object  # RigidMolecule
    ewald_params: EwaldParams
    ewald_cache: EwaldCache


def water_sites_lab(Ri, qi, water, L):
    """Get lab-frame positions of all sites for one water molecule.
    
    Args:
        Ri: COM position, shape (3,)
        qi: Orientation quaternion, shape (4,)
        water: RigidMolecule instance
        L: Box length
        
    Returns:
        Lab-frame site positions, shape (3, 3)
    """
    return apply_rigid_transform(Ri, qi, water.s_body, L)


def flatten_sites(R, q, water, L):
    """Flatten all water molecules into site arrays.
    
    Args:
        R: COM positions, shape (N, 3)
        q: Quaternions, shape (N, 4)
        water: RigidMolecule instance
        L: Box length
        
    Returns:
        Tuple (positions, charges, mol_to_sites):
        - positions: All site positions, shape (3N, 3)
        - charges: All site charges, shape (3N,)
        - mol_to_sites: List of tuples (start, end) for each molecule
    """
    N = R.shape[0]
    positions = []
    charges = []
    mol_to_sites = []
    
    for m in range(N):
        sites = water_sites_lab(R[m], q[m], water, L)
        positions.append(sites)
        charges.append(water.charge)
        mol_to_sites.append((3*m, 3*m + 3))
    
    positions = np.vstack(positions)
    charges = np.concatenate(charges)
    
    return positions, charges, mol_to_sites


def oxygen_positions(R, q, water, L):
    """Get oxygen positions for all water molecules.
    
    Args:
        R: COM positions, shape (N, 3)
        q: Quaternions, shape (N, 4)
        water: RigidMolecule instance
        L: Box length
        
    Returns:
        Oxygen positions, shape (N, 3)
    """
    N = R.shape[0]
    O_pos = np.zeros((N, 3))
    
    for m in range(N):
        sites = water_sites_lab(R[m], q[m], water, L)
        O_pos[m] = sites[0]  # Site 0 is oxygen
    
    return O_pos


def total_energy_waterbox(box):
    """Compute total potential energy of water box.
    
    Energy = U_LJ (O-O only) + U_coul (Ewald)
    
    Args:
        box: WaterBox instance
        
    Returns:
        Total potential energy
    """
    N = box.R.shape[0]
    L = box.L
    water = box.water
    
    # Get oxygen positions
    O_pos = oxygen_positions(box.R, box.q, water, L)
    
    # LJ energy: O-O interactions only
    # Use same cutoff as in existing LJ module (need to choose rc)
    # For SPC/E, typical cutoff is around 1.0-1.2 nm
    rc = 1.0  # nm, cutoff for LJ
    rc2 = rc * rc
    sigma_O = water.sigma[0]
    epsilon_O = water.epsilon[0]
    
    # Convert sigma/epsilon to LJ units (assuming reduced units or consistent scaling)
    # For now, use direct LJ with sigma and epsilon
    # Note: lj_shifted_energy expects r2 and rc2, and returns 4*epsilon*((sigma/r)^12 - (sigma/r)^6) - shift
    # But our sigma/epsilon are in different units. We need to use them correctly.
    # Actually, lj_shifted_energy uses reduced units where sigma=1, epsilon=1
    # We need to scale distances by sigma and energies by epsilon
    
    U_LJ = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            dr = minimum_image(O_pos[j] - O_pos[i], L)
            r2 = np.dot(dr, dr)
            r = sqrt(r2)
            
            if r <= rc:
                # Scale by sigma for reduced units
                r_reduced = r / sigma_O
                r2_reduced = r_reduced * r_reduced
                rc_reduced = rc / sigma_O
                rc2_reduced = rc_reduced * rc_reduced
                
                # Compute LJ energy and scale by epsilon
                u_reduced = lj_shifted_energy(r2_reduced, rc2_reduced)
                U_LJ += epsilon_O * u_reduced
    
    # Coulomb energy: Ewald summation
    flat_pos, flat_charges, _ = flatten_sites(box.R, box.q, box.water, box.L)
    U_coul = ewald_energy_total(flat_pos, flat_charges, box.ewald_params)
    
    return U_LJ + U_coul


def delta_energy_rigid_move(box, m, R_new, q_new):
    """Compute energy change for rigid move of molecule m.
    
    Args:
        box: WaterBox instance
        m: Molecule index
        R_new: New COM position, shape (3,)
        q_new: New quaternion, shape (4,)
        
    Returns:
        Energy change ΔU = U(new) - U(old)
    """
    N = box.R.shape[0]
    L = box.L
    water = box.water
    ewald_params = box.ewald_params
    cache = box.ewald_cache
    
    # Normalize and wrap
    q_new = quaternion_normalize(q_new)
    R_new = R_new % L
    R_old = box.R[m] % L
    q_old = box.q[m]
    
    # Get old and new site positions
    sites_old = water_sites_lab(R_old, q_old, water, L)
    sites_new = water_sites_lab(R_new, q_new, water, L)
    
    # Wrap sites to [0, L) for consistency
    sites_old = sites_old % L
    sites_new = sites_new % L
    
    # Get oxygen positions
    O_old = sites_old[0]
    O_new = sites_new[0]
    
    # LJ ΔU: O-O interactions only
    sigma_O = water.sigma[0]
    epsilon_O = water.epsilon[0]
    rc = 1.0  # nm, same as in total_energy
    rc2 = rc * rc
    
    dU_LJ = 0.0
    O_pos = oxygen_positions(box.R, box.q, water, L)
    
    for j in range(N):
        if j == m:
            continue
        
        O_j = O_pos[j] % L
        
        # Old interaction
        dr_old = minimum_image(O_old - O_j, L)
        r2_old = np.dot(dr_old, dr_old)
        r_old = sqrt(r2_old)
        if r_old <= rc:
            r_reduced_old = r_old / sigma_O
            r2_reduced_old = r_reduced_old * r_reduced_old
            rc_reduced = rc / sigma_O
            rc2_reduced = rc_reduced * rc_reduced
            u_reduced_old = lj_shifted_energy(r2_reduced_old, rc2_reduced)
            dU_LJ -= epsilon_O * u_reduced_old
        
        # New interaction
        dr_new = minimum_image(O_new - O_j, L)
        r2_new = np.dot(dr_new, dr_new)
        r_new = sqrt(r2_new)
        if r_new <= rc:
            r_reduced_new = r_new / sigma_O
            r2_reduced_new = r_reduced_new * r_reduced_new
            rc_reduced = rc / sigma_O
            rc2_reduced = rc_reduced * rc_reduced
            u_reduced_new = lj_shifted_energy(r2_reduced_new, rc2_reduced)
            dU_LJ += epsilon_O * u_reduced_new
    
    # Coulomb ΔU: real-space + reciprocal
    alpha = ewald_params.alpha
    rcut = ewald_params.rcut
    ke = ewald_params.ke
    
    # Real-space: 3 moved sites vs all other sites (3N-3)
    dU_coul_real = 0.0
    flat_pos, flat_charges, mol_to_sites = flatten_sites(box.R, box.q, water, L)
    
    for a in range(3):  # Sites of molecule m
        site_idx_old = 3*m + a
        q_a = water.charge[a]
        r_old_a = sites_old[a]
        r_new_a = sites_new[a]
        
        for site_idx_other in range(3*N):
            # Skip sites of molecule m
            mol_other = site_idx_other // 3
            if mol_other == m:
                continue
            
            q_other = flat_charges[site_idx_other]
            r_other = flat_pos[site_idx_other] % L
            
            # Old interaction
            dr_old = minimum_image(r_old_a - r_other, L)
            r_old = sqrt(np.dot(dr_old, dr_old))
            if r_old <= rcut:
                dU_coul_real -= ke * q_a * q_other * erfc(alpha * r_old) / r_old
            
            # New interaction
            dr_new = minimum_image(r_new_a - r_other, L)
            r_new = sqrt(np.dot(dr_new, dr_new))
            if r_new <= rcut:
                dU_coul_real += ke * q_a * q_other * erfc(alpha * r_new) / r_new
    
    # Reciprocal-space: update S(k) virtually
    dU_coul_recip = 0.0
    
    for k_idx in range(cache.kvecs.shape[0]):
        k = cache.kvecs[k_idx]
        
        # Compute dS = Σ_{a in sites of m} q_a (exp(i k·r_new_a) - exp(i k·r_old_a))
        dS = 0.0 + 0.0j
        for a in range(3):
            q_a = water.charge[a]
            k_dot_r_old = np.dot(k, sites_old[a])
            k_dot_r_new = np.dot(k, sites_new[a])
            dS += q_a * (np.exp(1j * k_dot_r_new) - np.exp(1j * k_dot_r_old))
        
        # ΔU_recip = ke * c_k * (|S+dS|^2 - |S|^2)
        S_old = cache.S[k_idx]
        S_new = S_old + dS
        S_old_mag2 = np.abs(S_old)**2
        S_new_mag2 = np.abs(S_new)**2
        dU_coul_recip += ke * cache.c_k[k_idx] * (S_new_mag2 - S_old_mag2)
    
    return dU_LJ + dU_coul_real + dU_coul_recip


def apply_rigid_move(box, m, R_new, q_new):
    """Apply rigid move to molecule m and update cache.
    
    Args:
        box: WaterBox instance (modified in place)
        m: Molecule index
        R_new: New COM position, shape (3,)
        q_new: New quaternion, shape (4,)
    """
    L = box.L
    water = box.water
    cache = box.ewald_cache
    
    # Normalize and wrap
    q_new = quaternion_normalize(q_new)
    R_new = R_new % L
    
    # Get old and new site positions (wrapped)
    R_old = box.R[m] % L
    q_old = box.q[m]
    sites_old = water_sites_lab(R_old, q_old, water, L) % L
    sites_new = water_sites_lab(R_new, q_new, water, L) % L
    
    # Update box
    box.R[m] = R_new
    box.q[m] = q_new
    
    # Update Ewald cache: S(k) += dS for all sites of molecule m
    for k_idx in range(cache.kvecs.shape[0]):
        k = cache.kvecs[k_idx]
        
        # Compute dS = Σ_{a in sites of m} q_a (exp(i k·r_new_a) - exp(i k·r_old_a))
        dS = 0.0 + 0.0j
        for a in range(3):
            q_a = water.charge[a]
            k_dot_r_old = np.dot(k, sites_old[a])
            k_dot_r_new = np.dot(k, sites_new[a])
            dS += q_a * (np.exp(1j * k_dot_r_new) - np.exp(1j * k_dot_r_old))
        
        cache.S[k_idx] += dS

