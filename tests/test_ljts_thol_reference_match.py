"""Reference match test for Thol 2014 LJTS EOS implementation.

Validates that our EOS implementation matches a reference implementation
that follows the FEOS_LJTS_1.pdf C++ listing exactly.
"""

import numpy as np
import pytest
from src.eos import a_res, Z, P, mu_res, u_res


# Critical constants (from Thol 2014)
TC_REF = 1.086
RHOC_REF = 0.319

# FEOS parameter table (same as in src/eos/ljts_thol_2014.py)
# Columns: TermID, n, t, d, l, eta, beta, gamma, epsilon
FEOS_PARAMS_REF = np.array([
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


def _compute_basis_term_reference(term_id, n, t, d, l, eta, beta, gamma, epsilon, tau, delta):
    """Compute basis term and derivatives following PDF reference exactly.
    
    Returns: (term, dterm_dtau, dterm_ddelta, d2term_dtau2, d2term_dtaudelta, d2term_ddelta2)
    """
    tau_t = tau ** t if t != 0 else 1.0
    delta_d = delta ** d if d != 0 else 1.0
    
    if term_id == 1:
        # Polynomial: n * tau^t * delta^d
        term = n * tau_t * delta_d
        
        dterm_dtau = n * t * (tau ** (t - 1)) * delta_d if t != 0 else 0.0
        dterm_ddelta = n * d * (delta ** (d - 1)) * tau_t if d != 0 else 0.0
        
        d2term_dtau2 = n * t * (t - 1) * (tau ** (t - 2)) * delta_d if t > 1 else (0.0 if t != 1 else 0.0)
        d2term_dtaudelta = n * t * d * (tau ** (t - 1)) * (delta ** (d - 1)) if (t != 0 and d != 0) else 0.0
        d2term_ddelta2 = n * d * (d - 1) * (delta ** (d - 2)) * tau_t if d > 1 else (0.0 if d != 1 else 0.0)
        
    elif term_id == 2:
        # Polynomial with exp(-delta^l): n * tau^t * delta^d * exp(-delta^l)
        exp_factor = np.exp(-delta ** l)
        term = n * tau_t * delta_d * exp_factor
        
        dterm_dtau = n * t * (tau ** (t - 1)) * delta_d * exp_factor if t != 0 else 0.0
        dterm_ddelta = n * exp_factor * (
            d * (delta ** (d - 1)) * tau_t - l * (delta ** (l + d - 1)) * tau_t
        ) if d != 0 else -n * l * (delta ** (l - 1)) * tau_t * exp_factor
        
        d2term_dtau2 = n * t * (t - 1) * (tau ** (t - 2)) * delta_d * exp_factor if t > 1 else 0.0
        d2term_dtaudelta = n * exp_factor * (
            t * d * (tau ** (t - 1)) * (delta ** (d - 1)) - t * l * (tau ** (t - 1)) * (delta ** (l + d - 1))
        ) if (t != 0 and d != 0) else (-n * t * l * (tau ** (t - 1)) * (delta ** (l - 1)) * exp_factor if t != 0 else 0.0)
        d2term_ddelta2 = n * exp_factor * (
            d * (d - 1) * (delta ** (d - 2)) * tau_t
            - 2 * l * d * (delta ** (l + d - 2)) * tau_t
            + l * (l + d - 1) * (delta ** (l + d - 2)) * tau_t
        ) if d > 1 else (n * exp_factor * (
            -2 * l * (delta ** (l - 1)) * tau_t + l * (l - 1) * (delta ** (l - 2)) * tau_t
        ) if d == 1 else n * l * (l - 1) * (delta ** (l - 2)) * tau_t * exp_factor)
        
    elif term_id == 3:
        # Gaussian: n * tau^t * delta^d * exp(-eta*(delta-epsilon)^2 - beta*(tau-gamma)^2)
        delta_diff = delta - epsilon
        tau_diff = tau - gamma
        exp_arg = -eta * delta_diff**2 - beta * tau_diff**2
        exp_factor = np.exp(exp_arg)
        
        term = n * tau_t * delta_d * exp_factor
        
        dterm_dtau = n * exp_factor * (
            t * (tau ** (t - 1)) * delta_d - 2 * beta * tau_diff * tau_t * delta_d
        ) if t != 0 else -n * 2 * beta * tau_diff * delta_d * exp_factor
        
        dterm_ddelta = n * exp_factor * (
            d * (delta ** (d - 1)) * tau_t - 2 * eta * delta_diff * tau_t * delta_d
        ) if d != 0 else -n * 2 * eta * delta_diff * tau_t * exp_factor
        
        d2term_dtau2 = n * exp_factor * (
            (t * (t - 1) * (tau ** (t - 2)) * delta_d if t > 1 else 0.0)
            - 4 * beta * tau_diff * (t * (tau ** (t - 1)) * delta_d if t != 0 else 0.0)
            + 4 * beta**2 * tau_diff**2 * tau_t * delta_d
            - 2 * beta * tau_t * delta_d
        )
        
        d2term_dtaudelta = n * exp_factor * (
            (t * (tau ** (t - 1)) * d * (delta ** (d - 1)) if (t != 0 and d != 0) else 0.0)
            - 2 * beta * tau_diff * (d * (delta ** (d - 1)) * tau_t if d != 0 else 0.0)
            - 2 * eta * delta_diff * (t * (tau ** (t - 1)) * delta_d if t != 0 else 0.0)
            + 4 * eta * beta * delta_diff * tau_diff * tau_t * delta_d
        )
        
        d2term_ddelta2 = n * exp_factor * (
            (d * (d - 1) * (delta ** (d - 2)) * tau_t if d > 1 else 0.0)
            - 4 * eta * delta_diff * (d * (delta ** (d - 1)) * tau_t if d != 0 else 0.0)
            + 4 * eta**2 * delta_diff**2 * tau_t * delta_d
            - 2 * eta * tau_t * delta_d
        )
        
    else:
        raise ValueError(f"Unknown TermID: {term_id}")
    
    return term, dterm_dtau, dterm_ddelta, d2term_dtau2, d2term_dtaudelta, d2term_ddelta2


def compute_helmholtz_reference(T, rho):
    """Compute Helmholtz derivatives using reference implementation.
    
    Returns: (A00, A10, A01, A20, A11, A02)
    """
    tau = TC_REF / T
    delta = rho / RHOC_REF
    
    A00 = 0.0
    A10 = 0.0
    A01 = 0.0
    A20 = 0.0
    A11 = 0.0
    A02 = 0.0
    
    for row in FEOS_PARAMS_REF:
        term_id = int(row[0])
        n, t, d, l = row[1], row[2], row[3], row[4]
        eta, beta, gamma, epsilon = row[5], row[6], row[7], row[8]
        
        term, dtau, ddelta, d2tau2, d2taudelta, d2delta2 = _compute_basis_term_reference(
            term_id, n, t, d, l, eta, beta, gamma, epsilon, tau, delta
        )
        
        A00 += term
        A10 += dtau
        A01 += ddelta
        A20 += d2tau2
        A11 += d2taudelta
        A02 += d2delta2
    
    return A00, A10, A01, A20, A11, A02


def compute_eos_reference(T, rho):
    """Compute EOS properties using reference implementation.
    
    Returns: (a_res, Z, mu_res, u_res, P)
    """
    A00, A10, A01, A20, A11, A02 = compute_helmholtz_reference(T, rho)
    
    a_res_ref = A00
    Z_ref = 1.0 + A01
    mu_res_ref = A00 + A01
    # Correct formula based on thermodynamic identity: u_res/(kT) = tau * A10
    # So u_res = Tc * A10 (in ε units)
    # Note: The previous formula A10/T was incorrect and has been corrected.
    u_res_ref = TC_REF * A10
    P_ref = rho * T * Z_ref
    
    return a_res_ref, Z_ref, mu_res_ref, u_res_ref, P_ref


@pytest.mark.parametrize("T,rho", [
    (1.3, 0.5),
    (1.0, 0.3),
    (2.0, 0.6),
])
def test_eos_reference_match(T, rho):
    """Test that EOS implementation matches reference at tight tolerance."""
    # Reference values
    a_res_ref, Z_ref, mu_res_ref, u_res_ref, P_ref = compute_eos_reference(T, rho)
    
    # Our implementation
    a_res_impl = a_res(T, rho)
    Z_impl = Z(T, rho)
    mu_res_impl = mu_res(T, rho)
    u_res_impl = u_res(T, rho)
    P_impl = P(T, rho)
    
    # Tight tolerances
    rtol = 1e-12
    atol = 1e-12
    
    # Compare
    np.testing.assert_allclose(a_res_impl, a_res_ref, rtol=rtol, atol=atol,
                                err_msg=f"a_res mismatch at T={T}, rho={rho}")
    np.testing.assert_allclose(Z_impl, Z_ref, rtol=rtol, atol=atol,
                                err_msg=f"Z mismatch at T={T}, rho={rho}")
    np.testing.assert_allclose(mu_res_impl, mu_res_ref, rtol=rtol, atol=atol,
                                err_msg=f"mu_res mismatch at T={T}, rho={rho}")
    np.testing.assert_allclose(u_res_impl, u_res_ref, rtol=rtol, atol=atol,
                                err_msg=f"u_res mismatch at T={T}, rho={rho}")
    np.testing.assert_allclose(P_impl, P_ref, rtol=rtol, atol=atol,
                                err_msg=f"P mismatch at T={T}, rho={rho}")

