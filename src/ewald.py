"""Ewald summation for electrostatic interactions in periodic systems."""

from dataclasses import dataclass
import numpy as np
from math import erfc, sqrt, pi
from .utils import minimum_image


@dataclass
class EwaldParams:
    """Parameters for Ewald summation.
    
    Attributes:
        L: Box length (cubic box)
        alpha: Ewald splitting parameter
        rcut: Real-space cutoff distance
        kmax: Maximum k-vector magnitude (integer)
        ke: Coulomb constant (default 1.0)
    """
    L: float
    alpha: float
    rcut: float
    kmax: int
    ke: float = 1.0


@dataclass
class EwaldCache:
    """Cache for incremental Ewald updates.
    
    Attributes:
        params: EwaldParams instance
        kvecs: k-vectors, shape (M, 3)
        k2: Squared k-vector magnitudes, shape (M,)
        c_k: Precomputed coefficients c_k = (2*pi/V) * exp(-k2/(4*alpha^2))/k2, shape (M,)
        S: Complex structure factor S(k) = sum_j q_j exp(i k·r_j), shape (M,), dtype complex
    """
    params: EwaldParams
    kvecs: np.ndarray
    k2: np.ndarray
    c_k: np.ndarray
    S: np.ndarray  # complex structure factors


def k_vectors(L, kmax):
    """Generate k-vectors for reciprocal-space Ewald summation.
    
    Returns integer n-vectors for n != 0 with |n| <= kmax,
    and corresponding k = 2*pi/L * n.
    
    Args:
        L: Box length
        kmax: Maximum k-vector magnitude (integer)
        
    Returns:
        Tuple (n_vectors, k_vectors):
        - n_vectors: Integer vectors, shape (N_k, 3)
        - k_vectors: k-vectors, shape (N_k, 3)
    """
    # Generate all integer vectors n with |n| <= kmax, excluding n = (0,0,0)
    n_list = []
    for nx in range(-kmax, kmax + 1):
        for ny in range(-kmax, kmax + 1):
            for nz in range(-kmax, kmax + 1):
                n = np.array([nx, ny, nz])
                n_mag = np.linalg.norm(n)
                if n_mag > 0 and n_mag <= kmax:
                    n_list.append(n)
    
    n_vectors = np.array(n_list)
    
    # Compute k-vectors: k = 2*pi/L * n
    k_vectors = (2.0 * pi / L) * n_vectors
    
    return n_vectors, k_vectors


def ewald_energy_total(positions, charges, params):
    """Compute total electrostatic energy using Ewald summation.
    
    For a cubic periodic box with tin-foil boundary conditions.
    System must be net-neutral (sum of charges must be zero).
    
    Energy components:
    1. Real-space: sum over i<j with r <= rcut using minimum image
    2. Reciprocal-space: structure factor S(k) and sum over k != 0
    3. Self term: -ke * alpha/sqrt(pi) * sum_i q_i^2
    
    Args:
        positions: Particle positions, shape (N, 3)
        charges: Particle charges, shape (N,)
        params: EwaldParams instance
        
    Returns:
        Total electrostatic energy
        
    Raises:
        ValueError: If system is not net-neutral
    """
    # Check charge neutrality
    total_charge = np.sum(charges)
    if abs(total_charge) >= 1e-12:
        raise ValueError(f"System is not net-neutral: total charge = {total_charge}")
    
    N = positions.shape[0]
    L = params.L
    alpha = params.alpha
    rcut = params.rcut
    ke = params.ke
    
    # Real-space term: sum over i<j with r <= rcut
    E_real = 0.0
    rcut2 = rcut * rcut
    
    for i in range(N - 1):
        for j in range(i + 1, N):
            dr = minimum_image(positions[j] - positions[i], L)
            r2 = np.dot(dr, dr)
            r = sqrt(r2)
            
            if r <= rcut:
                E_real += ke * charges[i] * charges[j] * erfc(alpha * r) / r
    
    # Reciprocal-space term
    # Get k-vectors
    _, k_vecs = k_vectors(L, params.kmax)
    N_k = k_vecs.shape[0]
    
    # Compute structure factors S(k) = sum_j q_j exp(i k·r_j)
    E_recip = 0.0
    prefactor = ke * (2.0 * pi) / (L**3)
    
    for k_idx in range(N_k):
        k = k_vecs[k_idx]
        k2 = np.dot(k, k)
        
        # Structure factor: S(k) = sum_j q_j exp(i k·r_j)
        k_dot_r = np.dot(positions, k)  # shape (N,)
        S_real = np.sum(charges * np.cos(k_dot_r))
        S_imag = np.sum(charges * np.sin(k_dot_r))
        S_mag2 = S_real**2 + S_imag**2
        
        # Reciprocal-space contribution
        exp_factor = np.exp(-k2 / (4.0 * alpha**2))
        E_recip += prefactor * exp_factor / k2 * S_mag2
    
    # Self term: -ke * alpha/sqrt(pi) * sum_i q_i^2
    E_self = -ke * alpha / sqrt(pi) * np.sum(charges**2)
    
    # Total energy
    E_total = E_real + E_recip + E_self
    
    return E_total


def build_cache(positions, charges, params):
    # Check charge neutrality
    total_charge = np.sum(charges)
    if abs(total_charge) >= 1e-12:
        raise ValueError(f"System is not net-neutral: total charge = {total_charge}")

    # Canonicalize coordinates for k·r phase
    pos = positions % params.L

    # Get k-vectors
    _, k_vecs = k_vectors(params.L, params.kmax)
    N_k = k_vecs.shape[0]

    # Compute k^2 for each k-vector
    k2 = np.array([np.dot(k, k) for k in k_vecs])

    # Precompute coefficients c_k
    V = params.L**3
    alpha = params.alpha
    c_k = (2.0 * pi / V) * np.exp(-k2 / (4.0 * alpha**2)) / k2

    # Compute structure factors S(k)
    S = np.zeros(N_k, dtype=complex)
    for k_idx in range(N_k):
        k = k_vecs[k_idx]
        k_dot_r = np.dot(pos, k)          # <-- use wrapped pos here
        S[k_idx] = np.sum(charges * np.exp(1j * k_dot_r))

    return EwaldCache(params=params, kvecs=k_vecs, k2=k2, c_k=c_k, S=S)



def delta_energy_move(cache, i, r_new, positions, charges):
    """Compute incremental energy change for moving particle i to r_new.
    
    Computes ΔU_recip + ΔU_real for moving particle i.
    Real-space is computed by direct pair difference.
    Self term cancels for fixed charges.
    
    Args:
        cache: EwaldCache instance
        i: Index of particle to move
        r_new: New position of particle i, shape (3,)
        positions: Current particle positions, shape (N, 3)
        charges: Particle charges, shape (N,)
        
    Returns:
        Energy change ΔU = U(new) - U(old)
    """
    params = cache.params
    L = params.L
    alpha = params.alpha
    rcut = params.rcut
    ke = params.ke
    N = positions.shape[0]
    
    r_old = positions[i] % L
    r_new = r_new % L


    q_i = charges[i]
    
    # Real-space contribution: compute difference
    # Old contributions: sum over j != i with r_old <= rcut
    # New contributions: sum over j != i with r_new <= rcut
    dU_real = 0.0
    
    for j in range(N):
        if j == i:
            continue
        
        # Old interaction
        dr_old = minimum_image(r_old - positions[j], L)
        r_old_mag = sqrt(np.dot(dr_old, dr_old))
        if r_old_mag <= rcut:
            dU_real -= ke * q_i * charges[j] * erfc(alpha * r_old_mag) / r_old_mag
        
        # New interaction
        dr_new = minimum_image(r_new - positions[j], L)
        r_new_mag = sqrt(np.dot(dr_new, dr_new))
        if r_new_mag <= rcut:
            dU_real += ke * q_i * charges[j] * erfc(alpha * r_new_mag) / r_new_mag
    
    # Reciprocal-space contribution: ΔU_recip = ke * sum_k c_k * (|S_new(k)|^2 - |S_old(k)|^2)
    # We can compute this incrementally:
    # S_new(k) = S_old(k) - q_i * exp(i k·r_old) + q_i * exp(i k·r_new)
    #          = S_old(k) + q_i * (exp(i k·r_new) - exp(i k·r_old))
    
    dU_recip = 0.0
    for k_idx in range(cache.kvecs.shape[0]):
        k = cache.kvecs[k_idx]
        
        # Old contribution to S(k)
        k_dot_r_old = np.dot(k, r_old)
        exp_old = np.exp(1j * k_dot_r_old)
        S_old_k = cache.S[k_idx]
        
        # New contribution to S(k)
        k_dot_r_new = np.dot(k, r_new)
        exp_new = np.exp(1j * k_dot_r_new)
        S_new_k = S_old_k + q_i * (exp_new - exp_old)
        
        # Energy change: ke * c_k * (|S_new|^2 - |S_old|^2)
        S_old_mag2 = np.abs(S_old_k)**2
        S_new_mag2 = np.abs(S_new_k)**2
        dU_recip += ke * cache.c_k[k_idx] * (S_new_mag2 - S_old_mag2)
    
    # Self term cancels for fixed charges (q_i^2 is the same before and after)
    
    return dU_real + dU_recip


def apply_move(cache, i, r_new, positions, charges):
    """Update cache and positions after moving particle i to r_new.
    
    Updates positions[i] and updates structure factor S in cache.
    
    Args:
        cache: EwaldCache instance (modified in place)
        i: Index of particle to move
        r_new: New position of particle i, shape (3,)
        positions: Particle positions, shape (N, 3) (modified in place)
        charges: Particle charges, shape (N,)
    """
    L = cache.params.L
    r_old = positions[i] % L
    r_new = r_new % L

    q_i = charges[i]
    
    # Update position
    positions[i] = r_new
    
    # Update structure factors: S(k) += q_i * (exp(i k·r_new) - exp(i k·r_old))
    for k_idx in range(cache.kvecs.shape[0]):
        k = cache.kvecs[k_idx]
        
        k_dot_r_old = np.dot(k, r_old)
        k_dot_r_new = np.dot(k, r_new)
        
        exp_old = np.exp(1j * k_dot_r_old)
        exp_new = np.exp(1j * k_dot_r_new)
        
        cache.S[k_idx] += q_i * (exp_new - exp_old)

