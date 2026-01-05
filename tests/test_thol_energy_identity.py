"""Test to validate EOS energy mapping using finite differences.

Validates that u_res(T,rho) from the EOS module correctly represents
the residual internal energy per particle using the thermodynamic identity:
u_res/(kT) = tau * (∂a_res/∂tau)_delta

where a_res = A_res/(NkT) is the residual Helmholtz free energy per particle,
tau = Tc/T, and delta = rho/rhoc.
"""

import numpy as np
import pytest
from src.eos import a_res, u_res
from src.eos.ljts_thol_2014 import TC, RHOC


def finite_difference_u_res(T, rho, dT=1e-6):
    """Compute u_res/(kT) using finite difference of a_res.
    
    Uses the identity: u_res/(kT) = tau * (∂a_res/∂tau)_delta
    
    where tau = Tc/T, so (∂a_res/∂tau) = (∂a_res/∂T) * (dT/dtau)
    and dT/dtau = -Tc/tau^2 = -T^2/Tc
    
    Alternatively, we can compute:
    (∂a_res/∂tau)_delta = -T^2/Tc * (∂a_res/∂T)_rho
    
    For symmetric finite difference:
    (∂a_res/∂T)_rho ≈ [a_res(T+dT, rho) - a_res(T-dT, rho)] / (2*dT)
    
    Then: u_res/(kT) = tau * (∂a_res/∂tau)_delta
                    = (Tc/T) * [-T^2/Tc * (∂a_res/∂T)_rho]
                    = -T * (∂a_res/∂T)_rho
    
    So: u_res/(kT) = -T * [a_res(T+dT, rho) - a_res(T-dT, rho)] / (2*dT)
    
    Args:
        T: Temperature (reduced units)
        rho: Density (reduced units)
        dT: Finite difference step size
        
    Returns:
        u_res_over_kT: Residual internal energy per particle in units of kT
    """
    tau = TC / T
    
    # Symmetric finite difference for (∂a_res/∂T)_rho
    a_plus = a_res(T + dT, rho)
    a_minus = a_res(T - dT, rho)
    da_dT = (a_plus - a_minus) / (2.0 * dT)
    
    # u_res/(kT) = -T * (∂a_res/∂T)_rho
    u_res_over_kT = -T * da_dT
    
    return u_res_over_kT


def test_energy_identity_finite_difference():
    """Test that u_res matches finite difference of a_res.
    
    Validates the thermodynamic identity:
    u_res/(kT) = tau * (∂a_res/∂tau)_delta = -T * (∂a_res/∂T)_rho
    """
    # Test state points (same as consistency test)
    test_points = [
        (1.3, 0.5),
        (1.0, 0.3),
        (2.0, 0.6),
    ]
    
    # Finite difference step size (small but not too small to avoid numerical issues)
    dT = 1e-5
    
    for T, rho in test_points:
        # Get u_res from EOS module
        u_res_eos = u_res(T, rho)
        
        # Check what units u_res_eos is in by comparing to a_res
        a_res_val = a_res(T, rho)
        
        # Compute u_res/(kT) using finite difference
        u_res_over_kT_fd = finite_difference_u_res(T, rho, dT=dT)
        
        # The EOS u_res should return energy in epsilon units
        # So u_res/(kT) = u_res_eos / T (since kT = T in reduced units where epsilon/k = 1)
        u_res_over_kT_eos = u_res_eos / T
        
        # Compare
        diff = abs(u_res_over_kT_eos - u_res_over_kT_fd)
        rtol = 1e-4  # Allow 0.01% relative error for finite differences
        atol = 1e-6
        
        print(f"\nT={T}, rho={rho}:")
        print(f"  a_res = {a_res_val:.10f}")
        print(f"  u_res (EOS) = {u_res_eos:.10f} (epsilon units)")
        print(f"  u_res/(kT) (EOS) = {u_res_over_kT_eos:.10f}")
        print(f"  u_res/(kT) (finite diff) = {u_res_over_kT_fd:.10f}")
        print(f"  difference = {diff:.10f}")
        
        assert np.allclose(u_res_over_kT_eos, u_res_over_kT_fd, rtol=rtol, atol=atol), (
            f"T={T}, rho={rho}: u_res/(kT) mismatch:\n"
            f"  EOS: {u_res_over_kT_eos:.10f}\n"
            f"  Finite diff: {u_res_over_kT_fd:.10f}\n"
            f"  Difference: {diff:.10f}\n"
            f"  This suggests u_res() implementation may have a scaling/convention error."
        )


def test_energy_identity_direct_tau():
    """Alternative test using direct tau perturbation.
    
    Computes (∂a_res/∂tau)_delta directly by perturbing tau.
    """
    test_points = [
        (1.3, 0.5),
        (1.0, 0.3),
        (2.0, 0.6),
    ]
    
    # Finite difference step in tau
    dtau_frac = 1e-5
    
    for T, rho in test_points:
        tau = TC / T
        
        # Perturb tau
        tau_plus = tau * (1 + dtau_frac)
        tau_minus = tau * (1 - dtau_frac)
        T_plus = TC / tau_plus
        T_minus = TC / tau_minus
        
        # Compute a_res at perturbed temperatures
        a_plus = a_res(T_plus, rho)
        a_minus = a_res(T_minus, rho)
        
        # Finite difference: (∂a_res/∂tau)_delta
        da_dtau = (a_plus - a_minus) / (tau_plus - tau_minus)
        
        # u_res/(kT) = tau * (∂a_res/∂tau)_delta
        u_res_over_kT_fd = tau * da_dtau
        
        # Compare to EOS
        u_res_eos = u_res(T, rho)
        u_res_over_kT_eos = u_res_eos / T
        
        diff = abs(u_res_over_kT_eos - u_res_over_kT_fd)
        rtol = 1e-4
        atol = 1e-6
        
        print(f"\nT={T}, rho={rho} (tau method):")
        print(f"  u_res/(kT) (EOS) = {u_res_over_kT_eos:.10f}")
        print(f"  u_res/(kT) (finite diff tau) = {u_res_over_kT_fd:.10f}")
        print(f"  difference = {diff:.10f}")
        
        assert np.allclose(u_res_over_kT_eos, u_res_over_kT_fd, rtol=rtol, atol=atol), (
            f"T={T}, rho={rho}: u_res/(kT) mismatch (tau method):\n"
            f"  EOS: {u_res_over_kT_eos:.10f}\n"
            f"  Finite diff: {u_res_over_kT_fd:.10f}\n"
            f"  Difference: {diff:.10f}"
        )

