# CONVERGENCE CHECK: does scratch catch up to warm-start when epochs aren't capped?
#   Same near/far radius shapes as radius.py (reuses the cached 50k-point datasets).
#   Arms: scratch / warm (from specialist) / warm_rewarm (shrink-and-perturb first).
#   Each arm trains until sigma stops improving for PATIENCE epochs instead of a
#   fixed budget, so we see the real plateau rather than a budget artifact.
#   LR schedule keeps the note's shape (0.5**(ep/(E_WARM/3)), E_WARM=30) but stretched
#   over E_MAX so the decay doesn't manufacture a plateau at epoch ~40.
#   Needs specialist_{BASE}.weights.h5 in the folder (from moduli_net_v5.py).
import os, sys, csv, itertools, time, numpy as np, tensorflow as tf
tf.get_logger().setLevel('ERROR')
tfk = tf.keras
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.metrics import TotalLoss, SigmaLoss

QUICK = ('--quick' in sys.argv)
BASE = next((a.split('=')[1] for a in sys.argv if a.startswith('--base=')), 'A')

RADII      = [0.2] if QUICK else [0.2, 0.8]
N_PER      = 1 if QUICK else 2
E_MAX      = 20 if QUICK else 300     # ceiling, not a target — early stopping decides real length
PATIENCE   = 5 if QUICK else 15
MIN_DELTA  = 1e-4

N_POINTS       = 50000                # fixed: shares the cache with radius.py
HIDDEN, LAYERS = 256, 4               # fixed: must match the saved specialist
LR, LR_WARM    = 1e-3, 3e-4
SHRINK, PERTURB_STD = 0.5, 0.01

E_WARM_REF = 30                       # the note's budget — defines the decay shape
STRETCH    = E_MAX / E_WARM_REF       # same number of halvings, spread over E_MAX
HALF_LIFE  = STRETCH * E_WARM_REF / 3

DATA = f'moduli_data_radius{BASE}'
LOG  = f'convergence{BASE}_log' + ('_quick' if QUICK else '') + '.csv'
SPEC = f'specialist_{BASE}.weights.h5'
BASE_SEED = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[BASE]
print("BASE", BASE, "RADII", RADII, "E_MAX", E_MAX, "patience", PATIENCE,
      "half-life", f"{HALF_LIFE:.0f} ep", "GPUs:", tf.config.list_physical_devices('GPU'), flush=True)

def deg3(n): return [m for m in itertools.product(range(4), repeat=n) if sum(m)==3]
MONOMIALS = np.array([a+b for a in deg3(3) for b in deg3(3)], dtype=np.int64)
AMBIENT = np.array([2, 2]); KMODULI = np.ones(2)

def recipe(seed, base=None, eps=0.0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=100) + 1j*rng.normal(size=100); c /= np.linalg.norm(c)
    if base is not None: c = base + eps*c; c /= np.linalg.norm(c)
    return c

def make_shape(name, coeffs):
    d = f'{DATA}/{name}'
    if not os.path.exists(os.path.join(d, 'basis.pickle')):
        print(f"   [{name}] not cached in {DATA}/ — sampling {N_POINTS} points (slow)", flush=True)
        pg = CICYPointGenerator([MONOMIALS], [coeffs], KMODULI, AMBIENT, verbose=0)
        kappa = pg.prepare_dataset(N_POINTS, d); pg.prepare_basis(d, kappa=kappa)
    data = np.load(os.path.join(d, 'dataset.npz'))
    BASIS = prepare_tf_basis(np.load(os.path.join(d, 'basis.pickle'), allow_pickle=True))
    return dict(name=name, c=coeffs, d=data, B=BASIS)

def new_core():
    return tfk.Sequential([tfk.Input(shape=(12,))] + [tfk.layers.Dense(HIDDEN, activation='gelu') for _ in range(LAYERS)]
                          + [tfk.layers.Dense(1, use_bias=False)])

def wrap(core, BASIS, lr):
    m = PhiFSModel(core, BASIS, alpha=[1.,1.,1.,1.,1.])
    m.compile(custom_metrics=[TotalLoss(), SigmaLoss()], optimizer=tfk.optimizers.legacy.Adam(lr)); return m

def sigma(m, d): return float(m.evaluate(d['X_val'], d['y_val'], batch_size=2000, verbose=0)[1])

def two_phase_epoch(m, d):
    m.learn_volk = tf.cast(False, tf.bool); m.fit(d['X_train'], d['y_train'], batch_size=64, epochs=1, verbose=0)
    m.learn_volk = tf.cast(True, tf.bool);  m.fit(d['X_train'], d['y_train'], batch_size=min(10000, len(d['X_train'])), epochs=1, verbose=0)

def shrink_perturb(core, shrink=SHRINK, std=PERTURB_STD):
    core.set_weights([w*shrink + np.random.normal(0, std, w.shape).astype(w.dtype) for w in core.get_weights()])
    return core

def recipe_dist(c, base):
    return min(np.linalg.norm(c*np.exp(1j*t) - base) for t in np.linspace(0, 2*np.pi, 64, endpoint=False))

def finetune_to_plateau(s, init_w, lr, rewarm, log_rows, row_prefix):
    k = new_core()
    if init_w is not None:
        k.set_weights(init_w)
        if rewarm: shrink_perturb(k)
    m = wrap(k, s['B'], lr)
    best, since, ep = float('inf'), 0, 0
    val = sigma(m, s['d']); log_rows.append(list(row_prefix) + [ep, val])
    t0 = time.time()
    while ep < E_MAX and since < PATIENCE:
        ep += 1
        m.optimizer.learning_rate.assign(lr * 0.5**(ep / HALF_LIFE))
        two_phase_epoch(m, s['d'])
        val = sigma(m, s['d']); log_rows.append(list(row_prefix) + [ep, val])
        if val < best - MIN_DELTA: best, since = val, 0
        else: since += 1
        if ep in (1, 5) or ep % 10 == 0:
            print(f"      {row_prefix[3]:12s} ep {ep:3d}  sigma {val:.4f}  best {best:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return ep, val, best

def main():
    base = recipe(BASE_SEED)
    core = new_core(); core.load_weights(SPEC); W = core.get_weights()
    print(f"Loaded {SPEC}", flush=True)

    if not os.path.exists(LOG):
        with open(LOG, 'w', newline='') as f:
            csv.writer(f).writerow(['eps', 'shape', 'dist', 'mode', 'epoch', 'sigma'])
    with open(LOG) as f:
        done = {(r[0], r[1], r[3]) for r in csv.reader(f) if r and r[0] != 'eps'}

    print(f"\n{'eps':>5} {'shape':10s} {'mode':12s} | stopped@   final     best", flush=True)
    for eps in RADII:
        for i in range(N_PER):
            name = f'r{eps}_{i}'
            s = make_shape(name, recipe(500 + 10*BASE_SEED + i, base, eps))
            dist = recipe_dist(s['c'], base)
            for mode, init_w, lr, rewarm in [('scratch', None, LR, False),
                                             ('warm', W, LR_WARM, False),
                                             ('warm_rewarm', W, LR_WARM, True)]:
                if (str(eps), name, mode) in done:
                    print(f"   {eps:>5} {name:10s} {mode:12s} already logged, skipping", flush=True); continue
                rows = []
                ep, val, best = finetune_to_plateau(s, init_w, lr, rewarm, rows, (eps, name, dist, mode))
                with open(LOG, 'a', newline='') as f: csv.writer(f).writerows(rows)
                print(f"{eps:>5} {name:10s} {mode:12s} | {ep:8d}  {val:7.4f}  {best:7.4f}", flush=True)

    print(f"\nLog: {LOG}", flush=True)
    print("Read: if scratch's best ties or beats warm's best, the note should claim compute-efficiency,", flush=True)
    print("not 'reaches solutions scratch can't reach'. If warm_rewarm beats plain warm, the gap is the", flush=True)
    print("inherited-basin pathology (Ash & Adams) and shrink-and-perturb is the fix to mention.", flush=True)

if __name__ == '__main__':
    main()
