"""Lennard-Jones Truncated and Shifted (LJTS) EOS from Thol et al. 2014.

Reference: Thol et al., "Equation of State for the Lennard-Jones Truncated and 
Shifted Model Fluid" (2014). 

This implementation is for rc = 2.5σ (cutoff radius = 2.5).

All quantities are in reduced units: ε = σ = kB = 1.

Based on FEOS_LJTS_1.pdf reference implementation.
"""

import numpy as np


# Critical constants
TC = 1.086
RHOC = 0.319

# FEOS parameter table (k=0..20)
# Columns: k, TermID, n, t, d, l, eta, beta, gamma, epsilon
FEOS_PARAMS = np.array([
    [1, 0.0156060840, 1.000, 4.0, 0.0,  0.00,   0.00, 0.00, 0.00],  # k=0
    [1, 1.7917527000, 0.304, 1.0, 0.0,  0.00,   0.00, 0.00, 0.00],  # k=1
    [1, -1.9613228000, 0.583, 1.0, 0.0,  0.00,   0.00, 0.00, 0.00],  # k=2
    [1, 1.3045604000, 0.662, 2.0, 0.0,  0.00,   0.00, 0.00, 0.00],  # k=3
    [1, -1.8117673000, 0.870, 2.0, 0.0,  0.00,   0.00, 0.00, 0.00],  # k=4
    [1, 0.1548399700, 0.870, 3.0, 0.0,  0.00,   0.00, 0.00, 0.00],  # k=5
    [2, -0.4751098200, 1.286, 1.0, 1.0,  0.00,   0.00, 0.00, 0.00],  # k=6
    [2, -0.5842280700, 1.960, 1.0, 2.0,  0.00,   0.00, 0.00, 0.00],  # k=7
    [2, -0.5060736400, 2.400, 3.0, 2.0,  0.00,   0.00, 0.00, 0.00],  # k=8
    [2, 0.1163964400, 1.700, 2.0, 1.0,  0.00,   0.00, 0.00, 0.00],  # k=9
    [2, -0.2009241200, 3.000, 2.0, 2.0,  0.00,   0.00, 0.00, 0.00],  # k=10
    [2, -0.0948852040, 1.250, 5.0, 1.0,  0.00,   0.00, 0.00, 0.00],  # k=11
    [3, 0.0094333106, 3.600, 1.0, 0.0,  4.70,  20.0, 1.0, 0.55],  # k=12
    [3, 0.3044462800, 2.080, 1.0, 0.0,  1.92,  0.77, 0.5, 0.70],  # k=13
    [3, -0.0010820946, 5.240, 2.0, 0.0,  2.70,  0.50, 0.8, 2.00],  # k=14
    [3, -0.0996933910, 0.960, 3.0, 0.0,  1.49,  0.80, 1.5, 1.14],  # k=15
    [3, 0.0091193522, 1.360, 3.0, 0.0,  0.65,  0.40, 0.7, 1.20],  # k=16
    [3, 0.1297054300, 1.655, 2.0, 0.0,  1.73,  0.43, 1.6, 1.31],  # k=17
    [3, 0.0230360300, 0.900, 1.0, 0.0,  3.70,  8.00, 1.3, 1.14],  # k=18
    [3, -0.0826710730, 0.860, 2.0, 0.0,  1.90,  3.30, 0.6, 0.53],  # k=19
    [3, -2.2497821000, 3.950, 3.0, 0.0, 13.20, 114.0, 1.3, 0.96],  # k=20
])

# Extract parameter arrays
TERM_ID = FEOS_PARAMS[:, 0].astype(int)  # TermID: 1, 2, or 3
N_PARAM = FEOS_PARAMS[:, 1]  # n coefficient
T_PARAM = FEOS_PARAMS[:, 2]  # t exponent for tau
D_PARAM = FEOS_PARAMS[:, 3]  # d exponent for delta
L_PARAM = FEOS_PARAMS[:, 4]  # l exponent for exp(-delta^l) in TermID=2
ETA_PARAM = FEOS_PARAMS[:, 5]  # eta for gaussian in TermID=3
BETA_PARAM = FEOS_PARAMS[:, 6]  # beta for gaussian in TermID=3
GAMMA_PARAM = FEOS_PARAMS[:, 7]  # gamma for gaussian in TermID=3
EPSILON_PARAM = FEOS_PARAMS[:, 8]  # epsilon for gaussian in TermID=3


def _compute_basis_term_and_derivatives(term_id, n, t, d, l, eta, beta, gamma, epsilon, tau, delta):
    """Compute basis function term and its derivatives.
    
    Returns: (term, dterm_dtau, dterm_ddelta, d2term_dtau2, d2term_dtaudelta, d2term_ddelta2)
    """
    # Common factors
    tau_t = tau ** t
    delta_d = delta ** d
    
    if term_id == 1:
        # Polynomial term: n * tau^t * delta^d
        term = n * tau_t * delta_d
        
        dterm_dtau = n * t * (tau ** (t - 1)) * delta_d if t > 0 else 0.0
        dterm_ddelta = n * tau_t * d * (delta ** (d - 1)) if d > 0 else 0.0
        
        d2term_dtau2 = n * t * (t - 1) * (tau ** (t - 2)) * delta_d if t > 1 else 0.0
        d2term_dtaudelta = n * t * (tau ** (t - 1)) * d * (delta ** (d - 1)) if (t > 0 and d > 0) else 0.0
        d2term_ddelta2 = n * tau_t * d * (d - 1) * (delta ** (d - 2)) if d > 1 else 0.0
        
    elif term_id == 2:
        # Polynomial with exp(-delta^l): n * tau^t * delta^d * exp(-delta^l)
        exp_factor = np.exp(-(delta ** l))
        term = n * tau_t * delta_d * exp_factor
        
        dtau_factor = n * t * (tau ** (t - 1)) if t > 0 else 0.0
        ddelta_factor_base = n * tau_t * d * (delta ** (d - 1)) if d > 0 else 0.0
        ddelta_factor_exp = -n * tau_t * delta_d * l * (delta ** (l - 1)) if l > 0 else 0.0
        
        dterm_dtau = dtau_factor * delta_d * exp_factor
        dterm_ddelta = (ddelta_factor_base + ddelta_factor_exp) * exp_factor
        
        # Second derivatives (simplified, handling edge cases)
        d2term_dtau2 = n * t * (t - 1) * (tau ** (t - 2)) * delta_d * exp_factor if t > 1 else 0.0
        d2term_dtaudelta = (n * t * (tau ** (t - 1)) * (d * (delta ** (d - 1)) - l * delta_d * (delta ** (l - 1)))) * exp_factor if (t > 0) else 0.0
        
        # d2term_ddelta2 is more complex, handle carefully
        if d > 1:
            d2base = n * tau_t * d * (d - 1) * (delta ** (d - 2))
        else:
            d2base = 0.0
        if l > 1:
            d2exp1 = -n * tau_t * l * (l - 1) * delta_d * (delta ** (l - 2))
        else:
            d2exp1 = 0.0
        if d > 0 and l > 0:
            d2exp2 = -n * tau_t * 2 * d * l * (delta ** (d - 1)) * (delta ** (l - 1))
        else:
            d2exp2 = 0.0
        if l > 0:
            d2exp3 = n * tau_t * l * l * delta_d * (delta ** (2 * l - 2))
        else:
            d2exp3 = 0.0
        
        d2term_ddelta2 = (d2base + d2exp1 + d2exp2 + d2exp3) * exp_factor
        
    elif term_id == 3:
        # Gaussian-like: n * tau^t * delta^d * exp(-eta*(delta-epsilon)^2 - beta*(tau-gamma)^2)
        delta_diff = delta - epsilon
        tau_diff = tau - gamma
        exp_arg = -eta * delta_diff**2 - beta * tau_diff**2
        exp_factor = np.exp(exp_arg)
        
        term = n * tau_t * delta_d * exp_factor
        
        # First derivatives
        dterm_dtau = n * (t * (tau ** (t - 1)) * delta_d - 2 * beta * tau_diff * tau_t * delta_d) * exp_factor if t > 0 else -n * 2 * beta * tau_diff * tau_t * delta_d * exp_factor
        dterm_ddelta = n * (d * (delta ** (d - 1)) * tau_t - 2 * eta * delta_diff * tau_t * delta_d) * exp_factor if d > 0 else -n * 2 * eta * delta_diff * tau_t * delta_d * exp_factor
        
        # Second derivatives (simplified for stability)
        d2term_dtau2 = n * exp_factor * (
            (t * (t - 1) * (tau ** (t - 2)) * delta_d if t > 1 else 0.0)
            - 4 * beta * tau_diff * (t * (tau ** (t - 1)) * delta_d if t > 0 else 0.0)
            + 4 * beta**2 * tau_diff**2 * tau_t * delta_d
            - 2 * beta * tau_t * delta_d
        )
        
        d2term_dtaudelta = n * exp_factor * (
            (t * (tau ** (t - 1)) * d * (delta ** (d - 1)) if (t > 0 and d > 0) else 0.0)
            - 2 * beta * tau_diff * (d * (delta ** (d - 1)) * tau_t if d > 0 else 0.0)
            - 2 * eta * delta_diff * (t * (tau ** (t - 1)) * delta_d if t > 0 else 0.0)
            + 4 * eta * beta * delta_diff * tau_diff * tau_t * delta_d
        )
        
        d2term_ddelta2 = n * exp_factor * (
            (d * (d - 1) * (delta ** (d - 2)) * tau_t if d > 1 else 0.0)
            - 4 * eta * delta_diff * (d * (delta ** (d - 1)) * tau_t if d > 0 else 0.0)
            + 4 * eta**2 * delta_diff**2 * tau_t * delta_d
            - 2 * eta * tau_t * delta_d
        )
        
    else:
        raise ValueError(f"Unknown TermID: {term_id}")
    
    return term, dterm_dtau, dterm_ddelta, d2term_dtau2, d2term_dtaudelta, d2term_ddelta2


def _compute_helmholtz_derivatives(T, rho):
    """Compute Helmholtz free energy and its derivatives.
    
    Returns: (A00, A10, A01, A20, A11, A02)
    where indices are derivatives w.r.t. tau and delta.
    """
    tau = TC / T
    delta = rho / RHOC
    
    A00 = 0.0
    A10 = 0.0  # dA/dtau
    A01 = 0.0  # dA/ddelta
    A20 = 0.0  # d^2A/dtau^2
    A11 = 0.0  # d^2A/(dtau*ddelta)
    A02 = 0.0  # d^2A/ddelta^2
    
    for k in range(len(TERM_ID)):
        term_id = TERM_ID[k]
        n = N_PARAM[k]
        t = T_PARAM[k]
        d = D_PARAM[k]
        l = L_PARAM[k]
        eta = ETA_PARAM[k]
        beta = BETA_PARAM[k]
        gamma = GAMMA_PARAM[k]
        epsilon = EPSILON_PARAM[k]
        
        term, dterm_dtau, dterm_ddelta, d2term_dtau2, d2term_dtaudelta, d2term_ddelta2 = \
            _compute_basis_term_and_derivatives(term_id, n, t, d, l, eta, beta, gamma, epsilon, tau, delta)
        
        A00 += term
        A10 += dterm_dtau
        A01 += dterm_ddelta
        A20 += d2term_dtau2
        A11 += d2term_dtaudelta
        A02 += d2term_ddelta2
    
    return A00, A10, A01, A20, A11, A02


def a_res(T, rho):
    """Residual Helmholtz free energy per particle (in units of kT).
    
    a_res = (A_res / N) / (kT) = A00
    
    Args:
        T: Temperature (reduced units, T* = kT/ε)
        rho: Number density (reduced units, ρ* = ρ*σ³)
        
    Returns:
        Residual Helmholtz free energy per particle in units of kT
    """
    A00, _, _, _, _, _ = _compute_helmholtz_derivatives(T, rho)
    return A00


def Z(T, rho):
    """Compressibility factor Z = P/(ρ*kT).
    
    Z = 1 + A01
    
    Args:
        T: Temperature (reduced units)
        rho: Number density (reduced units)
        
    Returns:
        Compressibility factor (dimensionless)
    """
    _, _, A01, _, _, _ = _compute_helmholtz_derivatives(T, rho)
    return 1.0 + A01


def P(T, rho):
    """Pressure (in reduced units: P* = P*σ³/ε).
    
    P = rho * T * Z
    
    Args:
        T: Temperature (reduced units)
        rho: Number density (reduced units)
        
    Returns:
        Pressure in reduced units
    """
    return rho * T * Z(T, rho)


def u_res(T, rho):
    """Residual internal energy per particle (in units of ε).
    
    Thermodynamic identity:
    u_res/(kT) = tau * (∂a_res/∂tau)_delta
    
    where:
    - a_res = A_res/(NkT) is the residual Helmholtz free energy per particle
    - tau = Tc/T
    - (∂a_res/∂tau)_delta = A10 (computed in _compute_helmholtz_derivatives)
    
    Therefore: u_res/(kT) = tau * A10
    In reduced units (kT = T*ε where we set ε=1): u_res/T = tau * A10
    So: u_res = tau * A10 * T = (Tc/T) * A10 * T = Tc * A10
    
    However, the standard convention in reduced units is:
    u_res (in ε units) = u_res/(kT) * T = tau * A10 * T = Tc * A10
    
    But checking the reference implementation and standard practice,
    the formula is: u_res = tau * A10 (in units of ε when multiplied by T implicitly)
    
    Actually, let's be more careful. The standard formula from the reference is:
    u_res/(kT) = tau * A10, so u_res = tau * A10 * T = Tc * A10
    
    Args:
        T: Temperature (reduced units, T* = kT/ε)
        rho: Number density (reduced units, ρ* = ρ*σ³)
        
    Returns:
        Residual internal energy per particle in units of ε.
        Formula: u_res = Tc * A10 where A10 = (∂a_res/∂tau)_delta
    """
    tau = TC / T
    _, A10, _, _, _, _ = _compute_helmholtz_derivatives(T, rho)
    # u_res/(kT) = tau * A10, so u_res (in ε units) = tau * A10 * T = Tc * A10
    return TC * A10


def mu_res(T, rho):
    """Residual chemical potential (in units of kT).
    
    mu_res = A00 + A01
    
    Args:
        T: Temperature (reduced units)
        rho: Number density (reduced units)
        
    Returns:
        Residual chemical potential in units of kT
    """
    A00, _, A01, _, _, _ = _compute_helmholtz_derivatives(T, rho)
    return A00 + A01
