"""Diagnostic test to identify LJTS model convention mismatch.

Compares simulation observables (shifted vs unshifted) against EOS to determine
which convention the Thol EOS uses.

NOTE: This test uses LJTS (shifted-energy) as the primary model, not LJFS.
"""

import numpy as np
import pytest
from src.utils import init_lattice, minimum_image
from src.lj import (
    lj_shifted_energy,
    total_energy,
    delta_energy_particle_move,
    virial_pressure,
    lj_shifted_force_magnitude,
)
from src.lj_kmc import (
    compute_relocation_rates,
    sample_event,
    apply_relocation,
)
from src.eos import u_res, Z, mu_res


def lj_unshifted_energy(r2):
    """Compute unshifted Lennard-Jones energy: u_LJ(r) = 4*((1/r)^12 - (1/r)^6).
    
    Args:
        r2: Squared distance(s)
        
    Returns:
        Unshifted LJ energy
    """
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    return 4.0 * (inv_r12 - inv_r6)


def blocked_standard_error(samples, block_size):
    """Compute blocked standard error accounting for autocorrelation."""
    n_samples = len(samples)
    n_blocks = n_samples // block_size
    
    if n_blocks < 1:
        raise ValueError(f"Block size {block_size} too large for {n_samples} samples")
    
    block_means = []
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        block_means.append(np.mean(samples[start_idx:end_idx]))
    
    block_means = np.array(block_means)
    se_blocked = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
    
    return se_blocked, n_blocks


def compute_total_energy_unshifted(positions, L, rc):
    """Compute total energy using unshifted LJ potential.
    
    Returns: (U_unshift, N_pairs)
    """
    N = positions.shape[0]
    rc2 = rc * rc
    U_unshift = 0.0
    N_pairs = 0
    
    for i in range(N - 1):
        for j in range(i + 1, N):
            dr = minimum_image(positions[j] - positions[i], L)
            r2 = np.dot(dr, dr)
            if r2 < rc2 and r2 > 0:
                U_unshift += lj_unshifted_energy(r2)
                N_pairs += 1
    
    return U_unshift, N_pairs


def compute_widom_mu_ex_single_config(positions, L, rc, beta, n_insertions, rng, use_shifted=True):
    """Compute Widom excess chemical potential for a single configuration.
    
    Args:
        use_shifted: If True, use shifted potential; if False, use unshifted
    
    Returns: mu_ex = -kT * ln(<exp(-beta*dU_ins)>)
    """
    N = positions.shape[0]
    T = 1.0 / beta
    rc2 = rc * rc
    
    weights = []
    for _ in range(n_insertions):
        x_test = rng.random(3) * L
        dU_ins = 0.0
        for j in range(N):
            dr = minimum_image(x_test - positions[j], L)
            r2 = np.dot(dr, dr)
            if r2 < rc2:
                if use_shifted:
                    dU_ins += lj_shifted_energy(r2, rc2)
                else:
                    dU_ins += lj_unshifted_energy(r2)
        weights.append(np.exp(-beta * dU_ins))
    
    mean_weight = np.mean(weights)
    if mean_weight > 1e-300:
        mu_ex = -T * np.log(mean_weight)
    else:
        mu_ex = 1e10  # Large penalty for near-zero weights
    
    return mu_ex


def delta_u_particle_move(positions, i, new_pos, L, rc2):
    """Compute local energy change for moving particle i (O(N)) using shifted potential."""
    N = positions.shape[0]
    old_pos = positions[i]
    dU = 0.0
    
    for j in range(N):
        if j == i:
            continue
        dr_old = minimum_image(old_pos - positions[j], L)
        r2_old = np.dot(dr_old, dr_old)
        if r2_old < rc2:
            dU -= lj_shifted_energy(r2_old, rc2)
        dr_new = minimum_image(new_pos - positions[j], L)
        r2_new = np.dot(dr_new, dr_new)
        if r2_new < rc2:
            dU += lj_shifted_energy(r2_new, rc2)
    
    return dU


def compute_virial_pressure_manual(positions, L, rc, T):
    """Compute virial pressure manually for validation.
    
    Uses LJTS (shifted-energy) potential and force.
    
    P = rho*kT + (1/(3V)) * W
    where W = sum_{i<j} r_ij * F(r_ij)
    and F(r) = -dU/dr is the force magnitude from shifted LJ potential.
    
    Returns: P in reduced units
    """
    N = positions.shape[0]
    V = L**3
    rho = N / V
    rc2 = rc * rc
    
    # Ideal gas contribution
    P_ideal = rho * T
    
    # Virial contribution: W = sum_{i<j} r_ij * F(r_ij)
    # For shifted LJTS, use lj_shifted_force_magnitude
    W = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            dr = minimum_image(positions[j] - positions[i], L)
            r2 = np.dot(dr, dr)
            if r2 < rc2 and r2 > 0:
                r = np.sqrt(r2)
                F = lj_shifted_force_magnitude(r2, rc2)
                W += r * F
    
    # Pressure = ideal + W/(3V)
    P_virial = W / (3.0 * V)
    P_total = P_ideal + P_virial
    
    return P_total


def run_mc_sampler_with_diagnostics(positions, L, rc, beta, max_disp, n_sweeps, rng, sample_stride=1):
    """Metropolis MC sampler that computes shifted and unshifted energies.
    
    Uses LJTS (shifted-energy) for MC moves.
    
    Returns:
        U_shifted_samples, U_unshifted_samples, P_samples, Z_samples,
        mu_shifted_samples, mu_unshifted_samples
    """
    N = positions.shape[0]
    rc2 = rc * rc
    U_shifted_samples = []
    U_unshifted_samples = []
    P_samples = []
    Z_samples = []
    mu_shifted_samples = []
    mu_unshifted_samples = []
    T = 1.0 / beta
    rho = N / (L**3)
    n_widom_insertions = 20
    
    # Precompute u_LJ(rc) for verification
    rc_val = rc
    u_LJ_at_rc = lj_unshifted_energy(rc_val * rc_val)
    
    for sweep in range(n_sweeps):
        for _ in range(N):
            i = rng.integers(N)
            old_pos = positions[i].copy()
            disp = (rng.random(3) - 0.5) * 2 * max_disp
            new_pos = (old_pos + disp) % L
            
            dU = delta_u_particle_move(positions, i, new_pos, L, rc2)
            if dU <= 0.0 or rng.random() < np.exp(-beta * dU):
                positions[i] = new_pos
        
        # Sample observables only every sample_stride sweeps
        if (sweep + 1) % sample_stride == 0:
            # Compute both shifted and unshifted energies
            U_shifted = total_energy(positions, L, rc)
            U_unshifted, N_pairs = compute_total_energy_unshifted(positions, L, rc)
            
            # Verify identity: U_shifted - U_unshifted == -N_pairs * u_LJ(rc)
            expected_diff = -N_pairs * u_LJ_at_rc
            actual_diff = U_shifted - U_unshifted
            assert abs(actual_diff - expected_diff) < 1e-10, (
                f"Energy identity violation: U_shifted - U_unshifted = {actual_diff:.10f}, "
                f"expected -N_pairs*u_LJ(rc) = {expected_diff:.10f}"
            )
            
            U_shifted_samples.append(U_shifted / N)
            U_unshifted_samples.append(U_unshifted / N)
            
            # Virial pressure using LJTS (shifted-energy) potential
            P_inst = virial_pressure(positions, L, rc, T)
            P_samples.append(P_inst)
            
            # Compute Z from P: Z = P / (rho * T)
            Z_inst = P_inst / (rho * T)
            Z_samples.append(Z_inst)
            
            # Validate pressure calculation manually
            P_manual = compute_virial_pressure_manual(positions, L, rc, T)
            assert abs(P_inst - P_manual) < 1e-10, (
                f"Pressure mismatch: virial_pressure={P_inst:.10f}, manual={P_manual:.10f}"
            )
            
            # Widom μ_ex (both conventions)
            mu_ex_shifted = compute_widom_mu_ex_single_config(
                positions, L, rc, beta, n_widom_insertions, rng, use_shifted=True
            )
            mu_ex_unshifted = compute_widom_mu_ex_single_config(
                positions, L, rc, beta, n_widom_insertions, rng, use_shifted=False
            )
            mu_shifted_samples.append(mu_ex_shifted)
            mu_unshifted_samples.append(mu_ex_unshifted)
    
    return (U_shifted_samples, U_unshifted_samples, P_samples, Z_samples,
            mu_shifted_samples, mu_unshifted_samples)


def test_ljts_convention_diagnostics():
    """Diagnostic test to identify which convention the EOS uses.
    
    Primary comparison is LJTS (shifted-energy) model.
    """
    # Single state point
    T, rho = 1.3, 0.5
    
    # Simulation parameters (same as consistency test)
    N = 32
    rc = 2.5
    max_disp = 0.15
    burnin_sweeps = 200
    prod_sweeps = 500
    sample_stride = 5
    block_size = 5
    tolerance_sigma = 3.0
    
    beta = 1.0 / T
    L = (N / rho) ** (1/3)
    
    # Initialize
    rng = np.random.default_rng(12345)
    positions = init_lattice(N, L, rng)
    
    # Burn-in
    for sweep in range(burnin_sweeps):
        for _ in range(N):
            i = rng.integers(N)
            old_pos = positions[i].copy()
            disp = (rng.random(3) - 0.5) * 2 * max_disp
            new_pos = (old_pos + disp) % L
            dU = delta_u_particle_move(positions, i, new_pos, L, rc * rc)
            if dU <= 0.0 or rng.random() < np.exp(-beta * dU):
                positions[i] = new_pos
    
    # Production with diagnostics
    (U_shifted_samples, U_unshifted_samples, P_samples, Z_samples,
     mu_shifted_samples, mu_unshifted_samples) = run_mc_sampler_with_diagnostics(
        positions, L, rc, beta, max_disp, prod_sweeps, rng, sample_stride
    )
    
    # Compute blocked statistics
    se_U_shifted, _ = blocked_standard_error(U_shifted_samples, block_size)
    se_U_unshifted, _ = blocked_standard_error(U_unshifted_samples, block_size)
    se_P, _ = blocked_standard_error(P_samples, block_size)
    se_Z, _ = blocked_standard_error(Z_samples, block_size)
    se_mu_shifted, _ = blocked_standard_error(mu_shifted_samples, block_size)
    se_mu_unshifted, _ = blocked_standard_error(mu_unshifted_samples, block_size)
    
    mean_U_shifted = np.mean(U_shifted_samples)
    mean_U_unshifted = np.mean(U_unshifted_samples)
    mean_P = np.mean(P_samples)
    mean_Z = np.mean(Z_samples)
    mean_mu_shifted = np.mean(mu_shifted_samples)
    mean_mu_unshifted = np.mean(mu_unshifted_samples)
    
    # EOS predictions
    u_res_eos = u_res(T, rho)
    Z_eos = Z(T, rho)
    mu_res_eos = mu_res(T, rho)
    
    # Print diagnostics
    print("\n" + "="*70)
    print(f"Diagnostic Test Results (T={T}, rho={rho})")
    print("="*70)
    print("NOTE: Using LJTS (shifted-energy) potential for MC moves and observables")
    
    print(f"\nSimulation Results (blocked SE, LJTS model):")
    print(f"  U_shifted/N (LJTS) = {mean_U_shifted:10.6f} +/- {se_U_shifted:.6f}")
    print(f"  U_unshifted/N      = {mean_U_unshifted:10.6f} +/- {se_U_unshifted:.6f}")
    print(f"  P (LJTS)           = {mean_P:10.6f} +/- {se_P:.6f}")
    print(f"  Z = P/(rho*T)      = {mean_Z:10.6f} +/- {se_Z:.6f}")
    print(f"  mu_ex (shifted)    = {mean_mu_shifted:10.6f} +/- {se_mu_shifted:.6f}")
    print(f"  mu_ex (unshifted)  = {mean_mu_unshifted:10.6f} +/- {se_mu_unshifted:.6f}")
    
    print(f"\nEOS Predictions:")
    print(f"  u_res = {u_res_eos:10.6f}")
    print(f"  Z     = {Z_eos:10.6f}")
    print(f"  mu_res = {mu_res_eos:10.6f}")
    
    # Compare energy (LJTS shifted vs EOS)
    print(f"\nEnergy Comparison (LJTS shifted vs EOS):")
    diff_U_shifted = abs(mean_U_shifted - u_res_eos)
    diff_U_unshifted = abs(mean_U_unshifted - u_res_eos)
    match_shifted = diff_U_shifted <= tolerance_sigma * se_U_shifted
    match_unshifted = diff_U_unshifted <= tolerance_sigma * se_U_unshifted
    
    print(f"  |U_shifted/N (LJTS) - u_res| = {diff_U_shifted:.6f} (3sigma = {tolerance_sigma * se_U_shifted:.6f}) {'MATCH' if match_shifted else 'NO MATCH'}")
    print(f"  |U_unshifted/N - u_res| = {diff_U_unshifted:.6f} (3sigma = {tolerance_sigma * se_U_unshifted:.6f}) {'MATCH' if match_unshifted else 'NO MATCH'}")
    
    # Compare Z (LJTS vs EOS)
    print(f"\nCompressibility Factor Comparison (LJTS vs EOS):")
    diff_Z = abs(mean_Z - Z_eos)
    match_Z = diff_Z <= tolerance_sigma * se_Z
    print(f"  |Z_sim (LJTS) - Z_eos| = {diff_Z:.6f} (3sigma = {tolerance_sigma * se_Z:.6f}) {'MATCH' if match_Z else 'NO MATCH'}")
    print(f"  Z_sim = {mean_Z:.6f}, Z_eos = {Z_eos:.6f}, difference = {diff_Z:.6f}")
    
    # Compare chemical potential
    print(f"\nChemical Potential Comparison:")
    diff_mu_shifted = abs(mean_mu_shifted - mu_res_eos)
    diff_mu_unshifted = abs(mean_mu_unshifted - mu_res_eos)
    match_mu_shifted = diff_mu_shifted <= tolerance_sigma * se_mu_shifted
    match_mu_unshifted = diff_mu_unshifted <= tolerance_sigma * se_mu_unshifted
    
    print(f"  |mu_ex (shifted) - mu_res| = {diff_mu_shifted:.6f} (3sigma = {tolerance_sigma * se_mu_shifted:.6f}) {'MATCH' if match_mu_shifted else 'NO MATCH'}")
    print(f"  |mu_ex (unshifted) - mu_res| = {diff_mu_unshifted:.6f} (3sigma = {tolerance_sigma * se_mu_unshifted:.6f}) {'MATCH' if match_mu_unshifted else 'NO MATCH'}")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    if match_Z and match_shifted and match_mu_shifted:
        print("  EOS matches LJTS (shifted-energy) convention")
    elif match_Z and match_unshifted and match_mu_unshifted:
        print("  EOS matches UNSHIFTED truncated LJ convention")
    else:
        print("  EOS convention unclear from this diagnostic")
        if not match_Z:
            print(f"  Z mismatch is large: {diff_Z:.6f} (target: < {tolerance_sigma * se_Z:.6f})")
    print("="*70 + "\n")
