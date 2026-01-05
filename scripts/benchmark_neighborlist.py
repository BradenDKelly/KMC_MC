#!/usr/bin/env python3
"""Benchmark script comparing runtime scaling with vs without neighbor lists.

Compares Metropolis MC translation and Do/Ustinov-style relocation kMC
with and without neighbor lists.
"""

import argparse
import csv
import time
from pathlib import Path
import numpy as np

# Import from src (not tests)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mc import run_metropolis_mc
from src.neighborlist import NeighborListConfig, NeighborList
from src.lj_kmc import compute_relocation_rates, sample_event, apply_relocation
from src.lj_numba import total_energy_numba as total_energy_fast
from src.utils import init_lattice
from src.backend import require_numba, NUMBA_AVAILABLE


def make_positions(N, rho, seed):
    """Generate initial positions for benchmark.
    
    Args:
        N: Number of particles
        rho: Number density
        seed: Random seed
        
    Returns:
        positions: Particle positions, shape (N, 3)
        L: Box length
    """
    rng = np.random.default_rng(seed)
    L = (N / rho) ** (1/3)
    positions = init_lattice(N, L, rng)
    # Add small random displacement
    positions += rng.random((N, 3)) * 0.1
    positions = positions % L
    return positions, L


def warmup(positions, L, rc, skin):
    """Warm up JIT kernels before timing.
    
    Args:
        positions: Particle positions, shape (N, 3)
        L: Box length
        rc: Cutoff distance
        skin: Skin distance for neighbor list
    """
    try:
        # Warm up neighbor list build
        nl = NeighborList(positions, L, rc, skin=skin)
        
        # Warm up NL delta energy
        i = 0
        new_pos = positions[0] + np.array([0.1, 0.1, 0.1])
        new_pos = new_pos % L
        _ = nl.delta_energy_particle_move(positions, i, new_pos, L, rc)
        
        # Warm up total energy
        _ = nl.total_energy(positions, L, rc)
        
        # Warm up rebuild
        positions_warmup = positions.copy()
        positions_warmup[0] += np.array([0.05, 0.05, 0.05])
        positions_warmup = positions_warmup % L
        nl.rebuild(positions_warmup)
        
    except ImportError:
        # Numba not available, skip warmup
        pass


def run_kmc_relocation_brute(positions, L, rc, beta, n_sweeps, rng):
    """Run Do/Ustinov kMC relocation (brute-force mode).
    
    Args:
        positions: Particle positions, shape (N, 3) (modified in place)
        L: Box length
        rc: Cutoff distance
        beta: Inverse temperature
        n_sweeps: Number of sweeps (N events per sweep)
        rng: Random number generator
        
    Returns:
        Total number of events (n_sweeps * N)
    """
    N = positions.shape[0]
    
    for sweep in range(n_sweeps):
        # Compute rates ONCE per sweep and reuse for all N events
        kmc_rates = compute_relocation_rates(positions, L, rc, beta, rng, nl=None)
        
        # Apply N events using the same rates list
        for _ in range(N):
            event = sample_event(kmc_rates, rng)
            apply_relocation(positions, event, L)
    
    return n_sweeps * N


def run_kmc_relocation_nl(positions, L, rc, beta, n_sweeps, rng, skin):
    """Run Do/Ustinov kMC relocation (neighbor list mode).
    
    Args:
        positions: Particle positions, shape (N, 3) (modified in place)
        L: Box length
        rc: Cutoff distance
        beta: Inverse temperature
        n_sweeps: Number of sweeps (N events per sweep)
        rng: Random number generator
        skin: Skin distance for neighbor list
        
    Returns:
        Total number of events (n_sweeps * N)
    """
    N = positions.shape[0]
    
    # Initialize neighbor list
    nl = NeighborList(positions, L, rc, skin=skin)
    
    for sweep in range(n_sweeps):
        # Compute rates ONCE per sweep and reuse for all N events
        kmc_rates = compute_relocation_rates(positions, L, rc, beta, rng, nl=nl)
        
        # Apply N events using the same rates list
        for _ in range(N):
            event = sample_event(kmc_rates, rng)
            apply_relocation(positions, event, L)
            # Force rebuild after every relocation (relocations are large)
            nl.rebuild(positions)
    
    return n_sweeps * N


def benchmark_mc(N, rho, T, rc, max_disp, n_sweeps, skin, use_nl, seed_base):
    """Benchmark Metropolis MC translation.
    
    Args:
        N: Number of particles
        rho: Number density
        T: Temperature
        rc: Cutoff distance
        max_disp: Maximum displacement
        n_sweeps: Number of sweeps (production steps)
        skin: Skin distance (if use_nl)
        use_nl: If True, use neighbor list mode
        seed_base: Base seed for RNG
        
    Returns:
        dict with keys: seconds, seconds_per_sweep, seconds_per_move, n_events
    """
    positions, L = make_positions(N, rho, seed_base)
    
    # Set up neighbor list config
    neighborlist = NeighborListConfig(skin=skin) if use_nl else None
    
    # Run MC (use n_equil=0 for benchmark, only production)
    t_start = time.perf_counter()
    result = run_metropolis_mc(
        N=N, rho=rho, T=T, rc=rc, max_disp=max_disp,
        n_equil=0, n_prod=n_sweeps * N, sample_every=n_sweeps * N + 1,
        widom_inserts=0, seed=seed_base, neighborlist=neighborlist
    )
    t_end = time.perf_counter()
    
    total_time = t_end - t_start
    n_events = n_sweeps * N  # One move attempt per particle per sweep
    
    return {
        "seconds": total_time,
        "seconds_per_sweep": total_time / n_sweeps,
        "seconds_per_move": total_time / n_events,
        "n_events": n_events,
    }


def benchmark_kmc(N, rho, T, rc, n_sweeps, skin, use_nl, seed_base):
    """Benchmark Do/Ustinov kMC relocation.
    
    Args:
        N: Number of particles
        rho: Number density
        T: Temperature
        rc: Cutoff distance
        n_sweeps: Number of sweeps (N events per sweep)
        skin: Skin distance (if use_nl)
        use_nl: If True, use neighbor list mode
        seed_base: Base seed for RNG
        
    Returns:
        dict with keys: seconds, seconds_per_sweep, seconds_per_move, n_events
    """
    positions, L = make_positions(N, rho, seed_base)
    beta = 1.0 / T
    rng = np.random.default_rng(seed_base)
    
    t_start = time.perf_counter()
    if use_nl:
        n_events = run_kmc_relocation_nl(positions, L, rc, beta, n_sweeps, rng, skin)
    else:
        n_events = run_kmc_relocation_brute(positions, L, rc, beta, n_sweeps, rng)
    t_end = time.perf_counter()
    
    total_time = t_end - t_start
    
    return {
        "seconds": total_time,
        "seconds_per_sweep": total_time / n_sweeps,
        "seconds_per_move": total_time / n_events,
        "n_events": n_events,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark neighbor list performance for MC and kMC"
    )
    parser.add_argument(
        "--mode", choices=["mc", "kmc", "both"], default="both",
        help="Which mode(s) to benchmark (default: both)"
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[256, 512, 1024, 2048, 4096],
        help="System sizes to benchmark (default: 256 512 1024 2048 4096)"
    )
    parser.add_argument(
        "--sweeps", type=int, default=20,
        help="Number of sweeps to run (default: 20)"
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Number of repeats per configuration (default: 3)"
    )
    parser.add_argument(
        "--skin", type=float, default=0.3,
        help="Skin distance for neighbor list (default: 0.3)"
    )
    parser.add_argument(
        "--step", type=float, default=0.1,
        help="MC maximum displacement (default: 0.1)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Output CSV file path (optional)"
    )
    parser.add_argument(
        "--rho", type=float, default=0.8,
        help="Number density (default: 0.8)"
    )
    parser.add_argument(
        "--T", type=float, default=1.0,
        help="Temperature (default: 1.0)"
    )
    parser.add_argument(
        "--rc", type=float, default=2.5,
        help="Cutoff distance (default: 2.5)"
    )
    
    args = parser.parse_args()
    
    # Check numba availability
    numba_available = NUMBA_AVAILABLE
    if not numba_available:
        print("WARNING: Numba not available. Neighbor list mode will be skipped.")
        print("Install numba: pip install numba")
    
    # Print configuration
    print("=" * 70)
    print("Neighbor List Benchmark")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Sizes: {args.sizes}")
    print(f"Sweeps: {args.sweeps}")
    print(f"Repeats: {args.repeats}")
    print(f"Skin: {args.skin}")
    print(f"MC step: {args.step}")
    print(f"Density: {args.rho}, T: {args.T}, rc: {args.rc}")
    if numba_available:
        try:
            import numba
            print(f"Numba threads: {numba.config.NUMBA_NUM_THREADS}")
        except:
            pass
    print("=" * 70)
    print()
    
    # Warmup (once)
    print("Warming up JIT kernels...")
    warmup_pos, warmup_L = make_positions(128, args.rho, seed=999)
    if numba_available:
        try:
            warmup(warmup_pos, warmup_L, args.rc, args.skin)
        except Exception as e:
            print(f"Warning: Warmup failed: {e}")
    print("Warmup complete.\n")
    
    # CSV data
    csv_rows = []
    
    # Benchmark MC
    if args.mode in ["mc", "both"]:
        print("Metropolis MC (brute-force)")
        print("-" * 70)
        print(f"{'N':>8} {'seconds':>12} {'s/sweep':>12} {'us/move':>12}")
        print("-" * 70)
        
        for N in args.sizes:
            times = []
            for repeat in range(args.repeats):
                seed = 1000 + N * 10 + repeat
                try:
                    result = benchmark_mc(
                        N, args.rho, args.T, args.rc, args.step,
                        args.sweeps, args.skin, use_nl=False, seed_base=seed
                    )
                    times.append(result)
                    csv_rows.append({
                        "mode": "mc",
                        "use_neighborlist": False,
                        "N": N,
                        "sweeps": args.sweeps,
                        "repeat": repeat + 1,
                        "seconds": result["seconds"],
                        "seconds_per_sweep": result["seconds_per_sweep"],
                        "seconds_per_move": result["seconds_per_move"] * 1e6,  # microseconds
                        "n_events": result["n_events"],
                    })
                except Exception as e:
                    print(f"ERROR: N={N}, repeat={repeat+1}: {e}")
                    continue
            
            if times:
                avg_time = np.mean([t["seconds"] for t in times])
                avg_sweep = np.mean([t["seconds_per_sweep"] for t in times])
                avg_move = np.mean([t["seconds_per_move"] for t in times]) * 1e6  # microseconds
                print(f"{N:>8} {avg_time:>12.4f} {avg_sweep:>12.6f} {avg_move:>12.2f}")
        
        print()
        
        if numba_available:
            print("Metropolis MC (neighbor list)")
            print("-" * 70)
            print(f"{'N':>8} {'seconds':>12} {'s/sweep':>12} {'us/move':>12}")
            print("-" * 70)
            
            for N in args.sizes:
                times = []
                for repeat in range(args.repeats):
                    seed = 2000 + N * 10 + repeat
                    try:
                        result = benchmark_mc(
                            N, args.rho, args.T, args.rc, args.step,
                            args.sweeps, args.skin, use_nl=True, seed_base=seed
                        )
                        times.append(result)
                        csv_rows.append({
                            "mode": "mc",
                            "use_neighborlist": True,
                            "N": N,
                            "sweeps": args.sweeps,
                            "repeat": repeat + 1,
                            "seconds": result["seconds"],
                            "seconds_per_sweep": result["seconds_per_sweep"],
                            "seconds_per_move": result["seconds_per_move"] * 1e6,  # microseconds
                            "n_events": result["n_events"],
                        })
                    except Exception as e:
                        print(f"ERROR: N={N}, repeat={repeat+1}: {e}")
                        continue
                
                if times:
                    avg_time = np.mean([t["seconds"] for t in times])
                    avg_sweep = np.mean([t["seconds_per_sweep"] for t in times])
                    avg_move = np.mean([t["seconds_per_move"] for t in times]) * 1e6  # microseconds
                    print(f"{N:>8} {avg_time:>12.4f} {avg_sweep:>12.6f} {avg_move:>12.2f}")
        else:
            print("Metropolis MC (neighbor list) - SKIPPED (numba not available)")
        
        print()
    
    # Benchmark kMC
    if args.mode in ["kmc", "both"]:
        print("Do/Ustinov kMC relocation (brute-force)")
        print("-" * 70)
        print(f"{'N':>8} {'seconds':>12} {'s/sweep':>12} {'us/move':>12}")
        print("-" * 70)
        
        for N in args.sizes:
            times = []
            for repeat in range(args.repeats):
                seed = 3000 + N * 10 + repeat
                try:
                    result = benchmark_kmc(
                        N, args.rho, args.T, args.rc,
                        args.sweeps, args.skin, use_nl=False, seed_base=seed
                    )
                    times.append(result)
                    csv_rows.append({
                        "mode": "kmc",
                        "use_neighborlist": False,
                        "N": N,
                        "sweeps": args.sweeps,
                        "repeat": repeat + 1,
                        "seconds": result["seconds"],
                        "seconds_per_sweep": result["seconds_per_sweep"],
                        "seconds_per_move": result["seconds_per_move"] * 1e6,  # microseconds
                        "n_events": result["n_events"],
                    })
                except Exception as e:
                    print(f"ERROR: N={N}, repeat={repeat+1}: {e}")
                    continue
            
            if times:
                avg_time = np.mean([t["seconds"] for t in times])
                avg_sweep = np.mean([t["seconds_per_sweep"] for t in times])
                avg_move = np.mean([t["seconds_per_move"] for t in times]) * 1e6  # microseconds
                print(f"{N:>8} {avg_time:>12.4f} {avg_sweep:>12.6f} {avg_move:>12.2f}")
        
        print()
        
        if numba_available:
            print("Do/Ustinov kMC relocation (neighbor list)")
            print("-" * 70)
            print(f"{'N':>8} {'seconds':>12} {'s/sweep':>12} {'us/move':>12}")
            print("-" * 70)
            
            for N in args.sizes:
                times = []
                for repeat in range(args.repeats):
                    seed = 4000 + N * 10 + repeat
                    try:
                        result = benchmark_kmc(
                            N, args.rho, args.T, args.rc,
                            args.sweeps, args.skin, use_nl=True, seed_base=seed
                        )
                        times.append(result)
                        csv_rows.append({
                            "mode": "kmc",
                            "use_neighborlist": True,
                            "N": N,
                            "sweeps": args.sweeps,
                            "repeat": repeat + 1,
                            "seconds": result["seconds"],
                            "seconds_per_sweep": result["seconds_per_sweep"],
                            "seconds_per_move": result["seconds_per_move"] * 1e6,  # microseconds
                            "n_events": result["n_events"],
                        })
                    except Exception as e:
                        print(f"ERROR: N={N}, repeat={repeat+1}: {e}")
                        continue
                
                if times:
                    avg_time = np.mean([t["seconds"] for t in times])
                    avg_sweep = np.mean([t["seconds_per_sweep"] for t in times])
                    avg_move = np.mean([t["seconds_per_move"] for t in times]) * 1e6  # microseconds
                    print(f"{N:>8} {avg_time:>12.4f} {avg_sweep:>12.6f} {avg_move:>12.2f}")
        else:
            print("Do/Ustinov kMC relocation (neighbor list) - SKIPPED (numba not available)")
        
        print()
    
    # Write CSV
    if args.csv:
        csv_path = Path(args.csv)
        fieldnames = [
            "mode", "use_neighborlist", "N", "sweeps", "repeat",
            "seconds", "seconds_per_sweep", "seconds_per_move", "n_events"
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV written to: {csv_path}")
    
    print("=" * 70)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()

