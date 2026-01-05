#!/usr/bin/env python3
"""Run LJ simulations and log observables with block-based statistics.

Runs Metropolis MC and/or Do/Ustinov kMC relocation simulations,
sampling observables (energy, pressure, Widom mu_ex) in blocks.
"""

import argparse
import csv
import json
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

# Import from src (not tests)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neighborlist import NeighborListConfig, NeighborList, NUMBA_AVAILABLE
from src.lj_numba import total_energy_numba as total_energy_fast, virial_pressure_numba as virial_pressure_fast
from src.lj import total_energy, virial_pressure, lj_shifted_energy
from src.lj_kmc import compute_relocation_rates, sample_event, apply_relocation
from src.utils import init_lattice, minimum_image
from src.backend import require_numba
from src.mc import advance_mc_sweeps


def compute_widom_mu_ex(positions, L, rc, beta, n_insertions, rng):
    """Compute Widom excess chemical potential via test particle insertion.
    
    Args:
        positions: Particle positions, shape (N, 3) (must be float64, contiguous)
        L: Box length (float64)
        rc: Cutoff distance (float64)
        beta: Inverse temperature (float64)
        n_insertions: Number of test particle insertions
        rng: Random number generator
        
    Returns:
        Excess chemical potential mu_ex = -kT * ln(<exp(-beta*dU_ins)>)
    """
    # Ensure float64, contiguous for consistency
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    L = float(L)
    rc = float(rc)
    beta = float(beta)
    
    N = positions.shape[0]
    T = float(1.0 / beta)
    rc2 = float(rc * rc)
    
    weights = []
    for _ in range(n_insertions):
        x_test = rng.random(3, dtype=np.float64) * L
        dU_ins = 0.0
        for j in range(N):
            dr = minimum_image(x_test - positions[j], L)
            r2 = float(np.dot(dr, dr))
            if r2 < rc2:
                dU_ins += lj_shifted_energy(r2, rc2)
        weights.append(np.exp(-beta * dU_ins))
    
    mean_weight = np.mean(weights)
    if mean_weight > 1e-300:
        mu_ex = -T * np.log(mean_weight)
    else:
        mu_ex = 1e10  # Large penalty for near-zero weights
    
    return mu_ex


def blocked_standard_error(samples, block_size):
    """Compute blocked standard error accounting for autocorrelation.
    
    Args:
        samples: Array of samples
        block_size: Number of samples per block
        
    Returns:
        (standard_error, n_blocks)
    """
    n_samples = len(samples)
    n_blocks = n_samples // block_size
    
    if n_blocks < 1:
        return np.nan, 0
    
    block_means = []
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        block_means.append(np.mean(samples[start_idx:end_idx]))
    
    block_means = np.array(block_means)
    se_blocked = np.std(block_means, ddof=1) / np.sqrt(n_blocks) if n_blocks > 1 else 0.0
    
    return se_blocked, n_blocks


def make_initial_positions(N, rho, seed, method="lattice"):
    """Generate initial positions.
    
    Args:
        N: Number of particles
        rho: Number density
        seed: Random seed
        method: "lattice" or "random"
        
    Returns:
        (positions, L) where positions is float64, contiguous, C-order
    """
    rng = np.random.default_rng(seed)
    L = float((N / rho) ** (1/3))  # Ensure float64
    
    if method == "lattice":
        positions = init_lattice(N, L, rng)
        positions += rng.random((N, 3), dtype=np.float64) * 0.1
    else:
        positions = rng.random((N, 3), dtype=np.float64) * L
    
    positions = positions % L
    # Ensure float64, contiguous, C-order for Numba
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    return positions, L


def sample_observables(positions, L, rc, T):
    """Compute observables for current configuration.
    
    Args:
        positions: Particle positions, shape (N, 3) (must be float64, contiguous)
        L: Box length (float64)
        rc: Cutoff distance (float64)
        T: Temperature (float64)
        
    Returns:
        dict with keys: U, P
    """
    # Ensure float64, contiguous for Numba
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    L = float(L)
    rc = float(rc)
    T = float(T)
    
    U = total_energy_fast(positions, L, rc)
    P = virial_pressure_fast(positions, L, rc, T)
    
    return {"U": U, "P": P}


def run_mc_block(positions, L, rc, T, max_disp, n_sweeps, nl, rng, widom_every, widom_insertions):
    """Run MC for one block using optimized stepping via src.mc.advance_mc_sweeps.
    
    Args:
        positions: Particle positions, shape (N, 3) (modified in place, must be float64, contiguous)
        L: Box length (float64)
        rc: Cutoff distance (float64)
        T: Temperature (float64)
        max_disp: Maximum displacement (float64)
        n_sweeps: Number of sweeps in this block
        nl: NeighborList instance or None
        rng: Random number generator
        widom_every: Perform Widom sampling every N sweeps
        widom_insertions: Number of insertions per Widom estimate
        
    Returns:
        dict with keys: U, P, mu_ex_samples, acceptance, n_events, wall_time_block_s, wall_time_widom_s, wall_time_total_s
    """
    require_numba("MC stepping")
    # Ensure positions are float64, contiguous for Numba
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    # Ensure scalars are float64
    L = float(L)
    rc = float(rc)
    T = float(T)
    max_disp = float(max_disp)
    
    N = positions.shape[0]
    beta = float(1.0 / T)
    mu_ex_samples = []
    total_attempts = 0
    total_accepts = 0
    
    # Timing
    t_block_start = time.perf_counter()
    t_step_total = 0.0
    t_widom_total = 0.0
    
    # Run in chunks to preserve Widom sampling cadence
    remaining = n_sweeps
    sweep_count = 0
    
    while remaining > 0:
        chunk = min(widom_every, remaining)
        
        # Advance MC by chunk sweeps
        t_step_start = time.perf_counter()
        result = advance_mc_sweeps(positions, L, rc, T, max_disp, chunk, nl, rng)
        t_step_end = time.perf_counter()
        t_step_total += (t_step_end - t_step_start)
        total_attempts += result["attempts"]
        total_accepts += result["accepts"]
        sweep_count += chunk
        remaining -= chunk
        
        # Widom sampling after chunk
        if sweep_count % widom_every == 0:
            t_widom_start = time.perf_counter()
            mu_ex = compute_widom_mu_ex(positions, L, rc, beta, widom_insertions, rng)
            t_widom_end = time.perf_counter()
            t_widom_total += (t_widom_end - t_widom_start)
            mu_ex_samples.append(mu_ex)
    
    # Compute observables at end of block
    t_obs_start = time.perf_counter()
    obs = sample_observables(positions, L, rc, T)
    t_obs_end = time.perf_counter()
    
    t_block_end = time.perf_counter()
    wall_time_block_s = t_step_total  # Stepping only
    wall_time_widom_s = t_widom_total
    wall_time_total_s = t_block_end - t_block_start
    
    return {
        "U": obs["U"],
        "P": obs["P"],
        "mu_ex_samples": mu_ex_samples,
        "acceptance": total_accepts / max(total_attempts, 1),
        "n_events": total_attempts,
        "wall_time_block_s": wall_time_block_s,
        "wall_time_widom_s": wall_time_widom_s,
        "wall_time_total_s": wall_time_total_s,
    }


def run_kmc_block(positions, L, rc, T, n_sweeps, nl, rng, widom_every, widom_insertions):
    """Run kMC relocation for one block.
    
    Args:
        positions: Particle positions, shape (N, 3) (modified in place, must be float64, contiguous)
        L: Box length (float64)
        rc: Cutoff distance (float64)
        T: Temperature (float64)
        n_sweeps: Number of sweeps in this block
        nl: NeighborList instance or None
        rng: Random number generator
        widom_every: Perform Widom sampling every N sweeps
        widom_insertions: Number of insertions per Widom estimate
        
    Returns:
        dict with keys: U, P, mu_ex_samples, n_events, wall_time_block_s, wall_time_widom_s, wall_time_total_s
    """
    # Ensure float64, contiguous for Numba
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    L = float(L)
    rc = float(rc)
    T = float(T)
    
    N = positions.shape[0]
    beta = float(1.0 / T)
    mu_ex_samples = []
    
    # Timing
    t_block_start = time.perf_counter()
    t_step_total = 0.0
    t_widom_total = 0.0
    
    for sweep in range(n_sweeps):
        # Compute rates once per sweep
        t_step_start = time.perf_counter()
        kmc_rates = compute_relocation_rates(positions, L, rc, beta, rng, nl=nl)
        
        # Apply N events
        for _ in range(N):
            event = sample_event(kmc_rates, rng)
            apply_relocation(positions, event, L)
            if nl is not None:
                nl.rebuild(positions)  # Force rebuild after relocation
        t_step_end = time.perf_counter()
        t_step_total += (t_step_end - t_step_start)
        
        # Widom sampling
        if (sweep + 1) % widom_every == 0:
            t_widom_start = time.perf_counter()
            mu_ex = compute_widom_mu_ex(positions, L, rc, beta, widom_insertions, rng)
            t_widom_end = time.perf_counter()
            t_widom_total += (t_widom_end - t_widom_start)
            mu_ex_samples.append(mu_ex)
    
    # Compute observables at end of block
    t_obs_start = time.perf_counter()
    obs = sample_observables(positions, L, rc, T)
    t_obs_end = time.perf_counter()
    
    t_block_end = time.perf_counter()
    wall_time_block_s = t_step_total  # Stepping only
    wall_time_widom_s = t_widom_total
    wall_time_total_s = t_block_end - t_block_start
    
    return {
        "U": obs["U"],
        "P": obs["P"],
        "mu_ex_samples": mu_ex_samples,
        "n_events": n_sweeps * N,
        "wall_time_block_s": wall_time_block_s,
        "wall_time_widom_s": wall_time_widom_s,
        "wall_time_total_s": wall_time_total_s,
    }


def get_git_hash(no_git=False):
    """Get git commit hash if available.
    
    Args:
        no_git: If True, skip git hash retrieval and return "SKIPPED"
    
    Returns:
        Git hash string, "SKIPPED" if no_git=True, or None if git unavailable
    """
    if no_git:
        return "SKIPPED"
    
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5.0,  # Timeout after 5 seconds to avoid hangs
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def get_metadata(args, no_git=False):
    """Generate metadata dictionary.
    
    Args:
        args: Parsed arguments
        no_git: If True, skip git hash retrieval
    
    Returns:
        Metadata dictionary (only small scalars/strings/lists, no large arrays)
    """
    # Get git hash with timing
    t_git_start = time.perf_counter()
    git_hash = get_git_hash(no_git=no_git)
    t_git_end = time.perf_counter()
    git_time = t_git_end - t_git_start
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "args": vars(args),
    }
    
    # Git hash
    if git_hash:
        metadata["git_hash"] = git_hash
    
    # Numba version
    if NUMBA_AVAILABLE:
        try:
            import numba
            metadata["numba_version"] = numba.__version__
        except:
            pass
    
    # Ensure metadata doesn't contain large arrays
    # Remove any numpy arrays or large objects
    metadata_clean = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            # Skip numpy arrays
            continue
        elif isinstance(value, dict):
            # Recursively check dict values
            clean_dict = {}
            for k, v in value.items():
                if not isinstance(v, (np.ndarray, list)) or (isinstance(v, list) and len(v) < 1000):
                    # Keep small lists, skip large ones
                    if isinstance(v, list) and len(v) > 0:
                        # Check if list contains arrays
                        if any(isinstance(item, np.ndarray) for item in v):
                            continue
                    clean_dict[k] = v
            metadata_clean[key] = clean_dict
        elif isinstance(value, list) and len(value) > 1000:
            # Skip very large lists
            continue
        else:
            metadata_clean[key] = value
    
    return metadata_clean, git_time


def main():
    # Global timing
    t_script_start = time.perf_counter()
    timestamp_start = datetime.now().isoformat()
    
    # Per-run timing lists (will be populated during runs)
    run_timings = []
    warmup_timings = []
    parser = argparse.ArgumentParser(
        description="Run LJ simulations and log observables with block-based statistics"
    )
    parser.add_argument(
        "--engine", choices=["mc", "kmc", "both"], default="both",
        help="Which engine(s) to use (default: both)"
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory (default: results/run_<timestamp>)"
    )
    parser.add_argument(
        "--N", type=int, nargs="+", default=[256],
        help="Number of particles (default: 256)"
    )
    parser.add_argument(
        "--rho", type=float, nargs="+", default=[0.8],
        help="Number density (default: 0.8)"
    )
    parser.add_argument(
        "--T", type=float, nargs="+", default=[1.0],
        help="Temperature (default: 1.0)"
    )
    parser.add_argument(
        "--rc", type=float, default=2.5,
        help="Cutoff distance (default: 2.5)"
    )
    parser.add_argument(
        "--sweeps", type=int, default=2000,
        help="Number of production sweeps (default: 2000)"
    )
    parser.add_argument(
        "--burnin", type=int, default=500,
        help="Number of burn-in sweeps (default: 500)"
    )
    parser.add_argument(
        "--block", type=int, default=100,
        help="Block size in sweeps (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Base random seed (default: 123)"
    )
    parser.add_argument(
        "--neighborlist-skin", type=float, default=None,
        help="Skin distance for neighbor list (omit for brute-force mode)"
    )
    parser.add_argument(
        "--step", type=float, default=0.1,
        help="MC maximum displacement (default: 0.1)"
    )
    parser.add_argument(
        "--widom-every", type=int, default=10,
        help="Perform Widom sampling every N sweeps (default: 10)"
    )
    parser.add_argument(
        "--widom-insertions", type=int, default=100,
        help="Number of insertions per Widom estimate (default: 100)"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of replicate runs per state point (default: 3)"
    )
    parser.add_argument(
        "--no-git", action="store_true",
        help="Skip git hash retrieval (faster, avoids potential subprocess hangs)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.outdir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path("results") / f"run_{timestamp}"
    else:
        outdir = Path(args.outdir)
    
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {outdir}")
    print(f"Engines: {args.engine}")
    print(f"State points: {len(args.N)} x {len(args.rho)} x {len(args.T)} = {len(args.N) * len(args.rho) * len(args.T)}")
    print(f"Runs per state point: {args.runs}")
    print()
    
    # Check numba availability
    use_nl = args.neighborlist_skin is not None
    if use_nl and not NUMBA_AVAILABLE:
        print("ERROR: Neighbor list mode requested but numba not available")
        print("Install numba: pip install numba")
        sys.exit(1)
    
    # CSV file
    csv_path = outdir / "results.csv"
    csv_tmp_path = outdir / "results.csv.tmp"
    fieldnames = [
        "engine", "use_neighborlist", "N", "rho", "T", "rc", "skin",
        "seed", "run_id", "block_id",
        "U_mean", "P_mean", "mu_ex_mean", "mu_ex_stderr",
        "acceptance", "n_widom_samples", "n_events",
        "wall_time_block_s", "wall_time_widom_s", "wall_time_total_s",
        "steps_per_second", "wall_time_run_s", "effective_steps_per_second"
    ]
    
    # CSV file setup (timing starts after header write)
    rows_written = 0
    
    csvfile = open(csv_tmp_path, "w", newline="")
    try:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        t_csv_header_start = time.perf_counter()
        writer.writeheader()
        t_csv_header_end = time.perf_counter()
        t_csv_write = t_csv_header_end - t_csv_header_start  # Start with header time
        
        # Cartesian product of state points
        engines = []
        if args.engine in ["mc", "both"]:
            engines.append("mc")
        if args.engine in ["kmc", "both"]:
            engines.append("kmc")
        
        config_id = 0
        for N, rho, T in itertools.product(args.N, args.rho, args.T):
            config_id += 1
            print(f"Configuration {config_id}/{len(args.N) * len(args.rho) * len(args.T)}: N={N}, rho={rho}, T={T}")
            
            for engine in engines:
                for run_id in range(1, args.runs + 1):
                    # Per-run timing
                    t_run_start = time.perf_counter()
                    t_warmup = 0.0
                    
                    seed = args.seed + config_id * 1000 + run_id * 100
                    rng = np.random.default_rng(seed)
                    
                    # Print run info: moves per sweep and per block
                    moves_per_sweep = N  # One move per particle per sweep
                    moves_per_block = moves_per_sweep * args.block
                    print(f"  {engine.upper()} run {run_id}/{args.runs}: {moves_per_sweep} moves/sweep, {moves_per_block} moves/block")
                    
                    # Initial positions
                    positions, L = make_initial_positions(N, rho, seed, method="lattice")
                    
                    # Ensure scalars are float64 (for Numba dtype stability)
                    L = float(L)
                    rc_val = float(args.rc)
                    T_val = float(T)
                    step_val = float(args.step)
                    
                    # Initialize neighbor list if requested
                    nl = None
                    if use_nl:
                        nl = NeighborList(positions, L, rc_val, skin=float(args.neighborlist_skin))
                    
                    # Warmup Numba JIT before burn-in (MC only)
                    if engine == "mc":
                        print(f"  {engine.upper()} run {run_id}/{args.runs}: Warming up Numba JIT...", end=" ", flush=True)
                        t_warmup_start = time.perf_counter()
                        # Warmup: run 1 sweep with current positions and same mode
                        warmup_pos = positions.copy()
                        warmup_result = advance_mc_sweeps(
                            warmup_pos, L, rc_val, T_val, step_val, 1, nl, rng
                        )
                        # Warmup Widom if enabled
                        if args.widom_insertions > 0:
                            beta = float(1.0 / T_val)
                            _ = compute_widom_mu_ex(
                                warmup_pos, L, rc_val, beta, 1, rng
                            )
                        t_warmup_end = time.perf_counter()
                        t_warmup = t_warmup_end - t_warmup_start
                        print("done", flush=True)
                    
                    # Burn-in
                    print(f"  {engine.upper()} run {run_id}/{args.runs}: burn-in...", end=" ", flush=True)
                    if engine == "mc":
                        burnin_result = run_mc_block(
                            positions, L, rc_val, T_val, step_val, args.burnin,
                            nl, rng, args.widom_every, args.widom_insertions
                        )
                    else:  # kmc
                        burnin_result = run_kmc_block(
                            positions, L, rc_val, T_val, args.burnin,
                            nl, rng, args.widom_every, args.widom_insertions
                        )
                    print("done")
                    
                    # Production blocks
                    n_blocks = args.sweeps // args.block
                    block_mu_samples = []  # Collect all mu samples for final summary
                    U_block_values = []  # Track U per block (per particle)
                    P_block_values = []  # Track P per block
                    acceptance_blocks = []  # Track acceptance per block (MC only)
                    total_n_events = 0  # Track total events for run
                    
                    print(f"  Production: {n_blocks} blocks...", flush=True)
                    for block_id in range(1, n_blocks + 1):
                        print(f"    Block {block_id}/{n_blocks}...", end=" ", flush=True)
                        t_block_start = time.perf_counter()
                        if engine == "mc":
                            result = run_mc_block(
                                positions, L, rc_val, T_val, step_val, args.block,
                                nl, rng, args.widom_every, args.widom_insertions
                            )
                        else:  # kmc
                            result = run_kmc_block(
                                positions, L, rc_val, T_val, args.block,
                                nl, rng, args.widom_every, args.widom_insertions
                            )
                        t_block_end = time.perf_counter()
                        block_time = t_block_end - t_block_start
                        print(f"{block_time:.1f} seconds  Done", flush=True)
                        
                        # Store block values
                        U_block = result["U"] / N  # Per particle
                        P_block = result["P"]
                        U_block_values.append(U_block)
                        P_block_values.append(P_block)
                        total_n_events += result["n_events"]
                        
                        if "acceptance" in result:
                            acceptance_blocks.append(result["acceptance"])
                        
                        # Block statistics for mu_ex
                        mu_ex_mean_block = np.mean(result["mu_ex_samples"]) if result["mu_ex_samples"] else np.nan
                        mu_ex_stderr_block = np.std(result["mu_ex_samples"], ddof=1) / np.sqrt(len(result["mu_ex_samples"])) if len(result["mu_ex_samples"]) > 1 else np.nan
                        block_mu_samples.extend(result["mu_ex_samples"])
                        
                        # Compute steps_per_second
                        steps_per_second = result["n_events"] / result["wall_time_block_s"] if result["wall_time_block_s"] > 0 else np.nan
                        
                        # Write block row
                        row = {
                            "engine": engine,
                            "use_neighborlist": use_nl,
                            "N": N,
                            "rho": rho,
                            "T": T,
                            "rc": args.rc,
                            "skin": args.neighborlist_skin if use_nl else None,
                            "seed": seed,
                            "run_id": run_id,
                            "block_id": block_id,
                            "U_mean": U_block,
                            "P_mean": P_block,
                            "mu_ex_mean": mu_ex_mean_block,
                            "mu_ex_stderr": mu_ex_stderr_block,
                            "acceptance": result.get("acceptance", np.nan),
                            "n_widom_samples": len(result["mu_ex_samples"]),
                            "n_events": result["n_events"],
                            "wall_time_block_s": result["wall_time_block_s"],
                            "wall_time_widom_s": result["wall_time_widom_s"],
                            "wall_time_total_s": result["wall_time_total_s"],
                            "steps_per_second": steps_per_second,
                            "wall_time_run_s": None,  # Will be filled in summary row
                            "effective_steps_per_second": None,  # Will be filled in summary row
                        }
                        t_row_start = time.perf_counter()
                        writer.writerow(row)
                        t_row_end = time.perf_counter()
                        t_csv_write += (t_row_end - t_row_start)
                        rows_written += 1
                    
                    # Summary row (block_id = -1)
                    # Compute overall statistics using blocking
                    U_block_array = np.array(U_block_values)
                    P_block_array = np.array(P_block_values)
                    
                    U_mean = np.mean(U_block_array)
                    P_mean = np.mean(P_block_array)
                    U_se, _ = blocked_standard_error(U_block_array, 1)  # One sample per block
                    P_se, _ = blocked_standard_error(P_block_array, 1)
                    
                    mu_ex_mean = np.mean(block_mu_samples) if block_mu_samples else np.nan
                    mu_ex_se, _ = blocked_standard_error(block_mu_samples, len(block_mu_samples) // max(n_blocks, 1)) if block_mu_samples else (np.nan, 0)
                    if np.isnan(mu_ex_se) and block_mu_samples:
                        mu_ex_se = np.std(block_mu_samples, ddof=1) / np.sqrt(len(block_mu_samples))
                    
                    acceptance_mean = np.mean(acceptance_blocks) if acceptance_blocks else np.nan
                    
                    # Per-run timing
                    t_run_end = time.perf_counter()
                    wall_time_run_s = t_run_end - t_run_start
                    
                    # Total events (burn-in + production)
                    burnin_events = burnin_result.get("n_events", 0)
                    total_events_run = burnin_events + total_n_events
                    effective_steps_per_second = total_events_run / wall_time_run_s if wall_time_run_s > 0 else np.nan
                    
                    summary_row = {
                        "engine": engine,
                        "use_neighborlist": use_nl,
                        "N": N,
                        "rho": rho,
                        "T": T,
                        "rc": args.rc,
                        "skin": args.neighborlist_skin if use_nl else None,
                        "seed": seed,
                        "run_id": run_id,
                        "block_id": -1,  # Summary row
                        "U_mean": U_mean,
                        "P_mean": P_mean,
                        "mu_ex_mean": mu_ex_mean,
                        "mu_ex_stderr": mu_ex_se,
                        "acceptance": acceptance_mean,
                        "n_widom_samples": len(block_mu_samples),
                        "n_events": total_events_run,
                        "wall_time_block_s": None,  # Not applicable for summary
                        "wall_time_widom_s": None,  # Not applicable for summary
                        "wall_time_total_s": None,  # Not applicable for summary
                        "steps_per_second": None,  # Not applicable for summary
                        "wall_time_run_s": wall_time_run_s,
                        "effective_steps_per_second": effective_steps_per_second,
                    }
                    t_summary_start = time.perf_counter()
                    writer.writerow(summary_row)
                    t_summary_end = time.perf_counter()
                    t_csv_write += (t_summary_end - t_summary_start)
                    
                    # Store per-run timing for metadata
                    run_timings.append(wall_time_run_s)
                    warmup_timings.append(t_warmup)
        
        # Close CSV and perform atomic write
        t_flush_start = time.perf_counter()
        csvfile.flush()
        os.fsync(csvfile.fileno())
        csvfile.close()
        t_flush_end = time.perf_counter()
        t_csv_write += (t_flush_end - t_flush_start)
        
        # Atomic rename for CSV
        t_rename_start = time.perf_counter()
        try:
            os.replace(csv_tmp_path, csv_path)
        except Exception as e:
            # Fallback: copy if replace fails
            import shutil
            shutil.copy2(csv_tmp_path, csv_path)
            csv_tmp_path.unlink()
        t_rename_end = time.perf_counter()
        t_csv_write += (t_rename_end - t_rename_start)
    except Exception as e:
        csvfile.close()
        raise
    
    # Final timing
    t_script_end = time.perf_counter()
    timestamp_end = datetime.now().isoformat()
    wall_time_total_s = t_script_end - t_script_start
    
    # Write metadata (with timing information)
    print("  Building metadata...", end=" ", flush=True)
    t_meta_build_start = time.perf_counter()
    metadata, git_time = get_metadata(args, no_git=args.no_git)
    # Add timing information (ensure these are small scalars/lists)
    metadata["wall_time_total_s"] = float(wall_time_total_s)
    metadata["wall_time_per_run_s"] = [float(t) for t in run_timings]
    metadata["numba_warmup_time_s"] = [float(t) for t in (warmup_timings if warmup_timings else [0.0] * len(run_timings))]
    metadata["timestamp_start"] = timestamp_start
    metadata["timestamp_end"] = timestamp_end
    t_meta_build_end = time.perf_counter()
    t_meta_build = t_meta_build_end - t_meta_build_start
    print(f"done ({t_meta_build:.3f}s)", flush=True)
    
    # Atomic write for metadata
    print("  Writing metadata.json...", end=" ", flush=True)
    t_meta_write_start = time.perf_counter()
    metadata_tmp_path = outdir / "metadata.json.tmp"
    with open(metadata_tmp_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    metadata_path = outdir / "metadata.json"
    try:
        os.replace(metadata_tmp_path, metadata_path)
    except Exception as e:
        # Fallback: copy if replace fails
        import shutil
        shutil.copy2(metadata_tmp_path, metadata_path)
        metadata_tmp_path.unlink()
    t_meta_write_end = time.perf_counter()
    t_meta_write = t_meta_write_end - t_meta_write_start
    print(f"done ({t_meta_write:.3f}s)", flush=True)
    
    print()
    print(f"Results written to: {csv_path}")
    print(f"Metadata written to: {metadata_path}")
    print(f"Output write timings: csv={t_csv_write:.3f}s, json={t_meta_write:.3f}s, git={git_time:.3f}s")
    print("Done!")


if __name__ == "__main__":
    main()

