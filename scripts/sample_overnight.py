# Overnight sampling for the Mac. Samples every shape v4 needs (groups A, B, held-outs, strangers)
# PLUS two extra groups (C, D) and extra strangers for later experiments.
# Cached: rerunning skips finished shapes. Ctrl+C any time, rerun to resume.
# When done, zip the folder and upload to Drive:  zip -r moduli_data_v4.zip moduli_data_v4
import os, sys, time, itertools, numpy as np
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator

N_POINTS = 50000
N_NEIGH, N_HELD, N_STRANGE, EPS = 12, 3, 6, 0.05
DATA = 'moduli_data_v4'

def deg3(n): return [m for m in itertools.product(range(4), repeat=n) if sum(m)==3]
MONOMIALS = np.array([a+b for a in deg3(3) for b in deg3(3)], dtype=np.int64)
AMBIENT = np.array([2, 2]); KMODULI = np.ones(2)

def recipe(seed, base=None):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=100) + 1j*rng.normal(size=100); c /= np.linalg.norm(c)
    if base is not None: c = base + EPS*c; c /= np.linalg.norm(c)
    return c

def sample(name, coeffs):
    d = f'{DATA}/{name}'
    if os.path.exists(os.path.join(d, 'basis.pickle')):
        return 'cached'
    t0 = time.time()
    pg = CICYPointGenerator([MONOMIALS], [coeffs], KMODULI, AMBIENT, verbose=0)
    kappa = pg.prepare_dataset(N_POINTS, d); pg.prepare_basis(d, kappa=kappa)
    return f'{time.time()-t0:.0f}s'

def main():
    # same seeds as moduli_net_v4.py so the cache matches
    bases = {'A': recipe(0), 'B': recipe(1), 'C': recipe(2), 'D': recipe(3)}
    jobs = []
    for g, (tag, base) in enumerate(bases.items()):
        jobs += [(f'{tag}_{i}', recipe(100 + 50*g + i, base)) for i in range(N_NEIGH)]
        jobs += [(f'held{tag}_{i}', recipe(200 + 50*g + i, base)) for i in range(N_HELD)]
    jobs += [(f'strange_{i}', recipe(300 + i)) for i in range(N_STRANGE)]
    print(f"{len(jobs)} shapes x {N_POINTS} points -> {DATA}/")
    for k, (name, c) in enumerate(jobs, 1):
        print(f"[{k:2d}/{len(jobs)}] {name:10s} ...", end=' ', flush=True)
        print(sample(name, c), flush=True)
    print("Done. Now:  zip -r moduli_data_v4.zip moduli_data_v4   and upload the zip to Drive/cy_v4")

if __name__ == '__main__':
    main()
