# RADIUS EXPERIMENT: how far does a village extend?
#   Take specialist A. Make new shapes at increasing distance EPS from base A.
#   For each: zero-shot error under A, then fine-tune from A vs from scratch.
#   Output: error vs distance -> the "radius" of a village, which decides how many villages tile the landscape.
# Needs specialist_A.weights.h5 in the folder. Samples its own shapes (cached). --quick for laptops.
import os, sys, csv, itertools, time, numpy as np, tensorflow as tf
tf.get_logger().setLevel('ERROR')
tfk = tf.keras
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.metrics import TotalLoss, SigmaLoss

QUICK = ('--quick' in sys.argv)
SAMPLE_ONLY = ('--sample-only' in sys.argv)   # Mac: just make the shapes, train later on Colab
BASE = next((a.split('=')[1] for a in sys.argv if a.startswith('--base=')), 'A')   # which village: A/B/C/D
LONG = ('--long' in sys.argv)   # equal long budgets: warm vs scratch for 50 epochs at a few radii
RADII    = ([0.05, 0.2, 0.4] if LONG else [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]) if not QUICK else [0.05, 0.4]
N_PER    = 1 if QUICK else 3          # shapes per radius
N_POINTS = 2000 if QUICK else 50000
E_WARM   = 2 if QUICK else (50 if LONG else 10)
HIDDEN, LAYERS = (64, 3) if QUICK else (256, 4)
LR, LR_WARM = 1e-3, 3e-4
DATA = f'moduli_data_radius{BASE}' + ('_quick' if QUICK else ''); LOG = f'radius{BASE}_log' + ('_long' if LONG else '') + '.csv'
SPEC = f'specialist_{BASE}.weights.h5'
BASE_SEED = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[BASE]
print("BASE", BASE, "LONG", LONG, "RADII", RADII, "GPUs:", tf.config.list_physical_devices('GPU'), flush=True)

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
        pg = CICYPointGenerator([MONOMIALS], [coeffs], KMODULI, AMBIENT, verbose=0)
        kappa = pg.prepare_dataset(N_POINTS, d); pg.prepare_basis(d, kappa=kappa)
    data = np.load(os.path.join(d, 'dataset.npz'))
    BASIS = prepare_tf_basis(np.load(os.path.join(d, 'basis.pickle'), allow_pickle=True))
    return dict(name=name, c=coeffs, d=data, B=BASIS)
def new_core():
    return tfk.Sequential([tfk.Input(shape=(12,))] + [tfk.layers.Dense(HIDDEN, activation='gelu') for _ in range(LAYERS)] + [tfk.layers.Dense(1, use_bias=False)])
def wrap(core, BASIS, lr):
    m = PhiFSModel(core, BASIS, alpha=[1.,1.,1.,1.,1.]); m.compile(custom_metrics=[TotalLoss(), SigmaLoss()], optimizer=tfk.optimizers.legacy.Adam(lr)); return m
def sigma(m, d): return float(m.evaluate(d['X_val'], d['y_val'], batch_size=2000, verbose=0)[1])
def two_phase_epoch(m, d):
    m.learn_volk = tf.cast(False, tf.bool); m.fit(d['X_train'], d['y_train'], batch_size=64, epochs=1, verbose=0)
    m.learn_volk = tf.cast(True, tf.bool);  m.fit(d['X_train'], d['y_train'], batch_size=min(10000, len(d['X_train'])), epochs=1, verbose=0)
def finetune(s, init_w, lr):
    k = new_core()
    if init_w is not None: k.set_weights(init_w)
    m = wrap(k, s['B'], lr); cur = [sigma(m, s['d'])]
    for ep in range(1, E_WARM+1):
        m.optimizer.learning_rate.assign(lr * 0.5**(ep / (E_WARM/3))); two_phase_epoch(m, s['d']); cur.append(sigma(m, s['d']))
    return cur
def recipe_dist(c, base):
    return min(np.linalg.norm(c*np.exp(1j*t) - base) for t in np.linspace(0, 2*np.pi, 64, endpoint=False))

def main():
    base = recipe(BASE_SEED)
    if SAMPLE_ONLY:
        for eps in RADII:
            for i in range(N_PER):
                print(f'sampling r{eps}_{i} ...', flush=True); make_shape(f'r{eps}_{i}', recipe(500 + 10*BASE_SEED + i, base, eps))
        print(f'Done sampling. zip -r {DATA}.zip {DATA}  -> upload to Drive/cy_v4'); return
    core = new_core(); core.load_weights(SPEC); W = core.get_weights()
    done = set()
    if os.path.exists(LOG): done = {r[1] for r in csv.reader(open(LOG)) if r and r[0] != 'eps'}
    else:
        with open(LOG, 'w', newline='') as f: csv.writer(f).writerow(['eps', 'shape', 'recipe_dist', 'mode'] + [f'ep{i}' for i in range(E_WARM+1)])
    print(f"\n{'eps':>5} {'shape':10s} {'dist':>6} | zero-shot(A)  warm-best  scratch-best", flush=True)
    for eps in RADII:
        for i in range(N_PER):
            name = f'r{eps}_{i}'
            if name in done: print(f"   {name} already logged", flush=True); continue
            s = make_shape(name, recipe(500 + 10*BASE_SEED + i, base, eps)); dist = recipe_dist(s['c'], base)
            warm = finetune(s, W, LR_WARM); scr = finetune(s, None, LR)
            with open(LOG, 'a', newline='') as f:
                w = csv.writer(f); w.writerow([eps, name, dist, 'warm'] + warm); w.writerow([eps, name, dist, 'scratch'] + scr)
            print(f"{eps:>5} {name:10s} {dist:6.3f} | {warm[0]:11.3f}  {min(warm):9.3f}  {min(scr):12.3f}", flush=True)
    print(f"\nLog: {LOG}. The radius is where zero-shot stops beating scratch-best, and where warm-best stops beating scratch-best.")

if __name__ == '__main__':
    main()
