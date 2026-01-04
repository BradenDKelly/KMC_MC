import numpy as np

# ----------------------------
# Utilities
# ----------------------------
def minimum_image(dr, L):
    return dr - L * np.round(dr / L)

def lj_shifted_energy(r2, rc2):
    # LJ + shift at rc so u(rc)=0
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    u = 4.0 * (inv_r12 - inv_r6)
    # shift
    rc = np.sqrt(rc2)
    inv_rc2 = 1.0 / (rc * rc)
    inv_rc6 = inv_rc2**3
    inv_rc12 = inv_rc6**2
    u_shift = 4.0 * (inv_rc12 - inv_rc6)
    return u - u_shift

def total_energy(x, L, rc):
    N = x.shape[0]
    rc2 = rc * rc
    U = 0.0
    for i in range(N-1):
        dr = minimum_image(x[i+1:] - x[i], L)
        r2 = np.sum(dr*dr, axis=1)
        mask = r2 < rc2
        if np.any(mask):
            U += np.sum(lj_shifted_energy(r2[mask], rc2))
    return U

def delta_energy_particle_move(i, new_xi, x, L, rc):
    N = x.shape[0]
    rc2 = rc * rc
    old_xi = x[i]
    dU = 0.0
    for j in range(N):
        if j == i:
            continue
        # old
        dr_old = minimum_image(old_xi - x[j], L)
        r2_old = np.dot(dr_old, dr_old)
        if r2_old < rc2:
            dU -= lj_shifted_energy(r2_old, rc2)
        # new
        dr_new = minimum_image(new_xi - x[j], L)
        r2_new = np.dot(dr_new, dr_new)
        if r2_new < rc2:
            dU += lj_shifted_energy(r2_new, rc2)
    return dU

def init_lattice(N, L, rng):
    n3 = int(np.ceil(N ** (1/3)))
    grid = np.linspace(0, L, n3, endpoint=False)
    pts = np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T
    x = pts[:N].copy()
    x += 0.01 * (rng.random(x.shape) - 0.5)
    return x % L

# ----------------------------
# Equilibrium Metropolis MC (NVT) + Widom mu_ex
# ----------------------------
def run_metropolis_mc(N=256, rho=0.8, T=1.0, rc=2.5, max_disp=0.12,
                      n_equil=20_000, n_prod=50_000, sample_every=200,
                      widom_inserts=500, seed=123):
    rng = np.random.default_rng(seed)
    beta = 1.0 / T
    V = N / rho
    L = V ** (1/3)

    x = init_lattice(N, L, rng)
    U = total_energy(x, L, rc)

    attempts = 0
    accepts = 0
    mu_samples = []
    U_samples = []

    def widom_mu_ex():
        rc2 = rc*rc
        vals = []
        for _ in range(widom_inserts):
            xt = rng.random(3) * L
            dU = 0.0
            for j in range(N):
                dr = minimum_image(xt - x[j], L)
                r2 = np.dot(dr, dr)
                if r2 < rc2:
                    dU += lj_shifted_energy(r2, rc2)
            vals.append(np.exp(-beta * dU))
        avg = max(float(np.mean(vals)), 1e-300)
        return -T * np.log(avg)

    # Equilibration
    for _ in range(n_equil):
        attempts += 1
        i = rng.integers(N)
        disp = (rng.random(3)*2 - 1) * max_disp
        new_xi = (x[i] + disp) % L
        dU = delta_energy_particle_move(i, new_xi, x, L, rc)
        if dU <= 0.0 or rng.random() < np.exp(-beta*dU):
            x[i] = new_xi
            U += dU
            accepts += 1

    # Production
    for step in range(1, n_prod+1):
        attempts += 1
        i = rng.integers(N)
        disp = (rng.random(3)*2 - 1) * max_disp
        new_xi = (x[i] + disp) % L
        dU = delta_energy_particle_move(i, new_xi, x, L, rc)
        if dU <= 0.0 or rng.random() < np.exp(-beta*dU):
            x[i] = new_xi
            U += dU
            accepts += 1

        if step % sample_every == 0:
            U_samples.append(U / N)
            mu_samples.append(widom_mu_ex())

    return {
        "L": L,
        "acceptance": accepts / max(attempts, 1),
        "U_per_particle_mean": float(np.mean(U_samples)),
        "mu_ex_mean": float(np.mean(mu_samples)),
        "mu_ex_std": float(np.std(mu_samples, ddof=1)),
    }

# ----------------------------
# Rejection-free equilibrium "kMC" sampler (continuous-time Barker rates)
# - Generates K trial moves for each particle each step
# - rate r = exp(-beta*ΔU/2), choose one event proportional to r
# - advances algorithmic time dt = -ln(u)/sum(r)
# ----------------------------
def run_equilibrium_kmc(N=128, rho=0.8, T=1.0, rc=2.5, dmax=0.12, K=6,
                        n_steps=10_000, record_every=200, seed=123):
    rng = np.random.default_rng(seed)
    beta = 1.0 / T
    V = N / rho
    L = V ** (1/3)

    x = init_lattice(N, L, rng)
    U = total_energy(x, L, rc)
    t = 0.0

    U_samples = []
    t_samples = []

    rates = np.empty((N, K), dtype=float)
    trial_pos = np.empty((N, K, 3), dtype=float)

    for step in range(1, n_steps+1):
        disps = (rng.random((N, K, 3))*2 - 1) * dmax
        trial_pos[:] = (x[:, None, :] + disps) % L

        for i in range(N):
            for k in range(K):
                dU = delta_energy_particle_move(i, trial_pos[i, k], x, L, rc)
                z = -0.5 * beta * dU
                z = np.clip(z, -700.0, 700.0)
                rates[i, k] = np.exp(z)

        Rtot = float(np.sum(rates))
        if not np.isfinite(Rtot) or Rtot <= 0:
            raise RuntimeError("Invalid total rate; reduce density/dmax or improve init.")

        dt = -np.log(rng.random()) / Rtot
        t += dt

        flat = rates.ravel()
        thresh = rng.random() * Rtot
        idx = int(np.searchsorted(np.cumsum(flat), thresh))
        i = idx // K
        k = idx % K

        new_xi = trial_pos[i, k].copy()
        dU = delta_energy_particle_move(i, new_xi, x, L, rc)
        x[i] = new_xi
        U += dU

        if step % record_every == 0:
            U_samples.append(U / N)
            t_samples.append(t)

    return {
        "L": L,
        "U_per_particle_mean": float(np.mean(U_samples)),
        "t_last": float(t_samples[-1]) if t_samples else t,
    }

if __name__ == "__main__":
    print("Metropolis MC + Widom mu_ex:")
    res = run_metropolis_mc(N=256, rho=0.8, T=1.0, n_equil=10_000, n_prod=20_000,
                            sample_every=200, widom_inserts=300)
    print(res)

    print("\nEquilibrium kMC sampler (algorithmic time):")
    res2 = run_equilibrium_kmc(N=128, rho=0.8, T=1.0, n_steps=5000, record_every=200)
    print(res2)

