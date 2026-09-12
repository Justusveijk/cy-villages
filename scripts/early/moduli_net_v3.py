# Stage-2 prototype v3: GROUPS of shapes.
#   A. Fingerprint every recipe cheaply (no training) and see if it separates
#      "neighbours of the base recipe" from "random strangers".
#   B. Train ONE specialist on the neighbours.
#   C. Warm-start test: does starting from the specialist train a NEW shape faster
#      than starting from scratch? Neighbours vs strangers.
import os, csv, itertools, time, numpy as np, tensorflow as tf
tf.get_logger().setLevel('ERROR')
tfk = tf.keras
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.metrics import TotalLoss, SigmaLoss

N_NEIGH   = 12      # perturbations of the base recipe used for training the specialist
N_HELD    = 3       # perturbations held out (unseen neighbours)
N_STRANGE = 3       # random unrelated recipes
EPS       = 0.05    # perturbation size (relative)
N_POINTS  = 8000
E_SPEC    = 25      # epochs for the group specialist
E_WARM    = 8       # epochs for the warm-start comparison
HIDDEN    = 64
LOG       = 'moduli_v3_log.csv'

def deg3(n): return [m for m in itertools.product(range(4), repeat=n) if sum(m)==3]
MONOMIALS = np.array([a+b for a in deg3(3) for b in deg3(3)], dtype=np.int64)
AMBIENT   = np.array([2, 2]); KMODULI = np.ones(2)

def recipe(seed, base=None):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=100) + 1j*rng.normal(size=100)
    c /= np.linalg.norm(c)
    if base is not None:
        c = base + EPS*c; c /= np.linalg.norm(c)
    return c

def make_shape(name, coeffs):
    d = f'moduli_data/{name}'
    if not os.path.exists(os.path.join(d, 'basis.pickle')):
        pg = CICYPointGenerator([MONOMIALS], [coeffs], KMODULI, AMBIENT, verbose=0)
        kappa = pg.prepare_dataset(N_POINTS, d)
        pg.prepare_basis(d, kappa=kappa)
    data  = np.load(os.path.join(d, 'dataset.npz'))
    BASIS = prepare_tf_basis(np.load(os.path.join(d, 'basis.pickle'), allow_pickle=True))
    return name, coeffs, data, BASIS

def new_core():
    return tfk.Sequential([tfk.Input(shape=(12,)),
                           tfk.layers.Dense(HIDDEN, activation='gelu'),
                           tfk.layers.Dense(HIDDEN, activation='gelu'),
                           tfk.layers.Dense(HIDDEN, activation='gelu'),
                           tfk.layers.Dense(1, use_bias=False)])

def wrap(core, BASIS, lr=1e-3):
    m = PhiFSModel(core, BASIS, alpha=[1.,1.,1.,1.,1.])
    m.compile(custom_metrics=[TotalLoss(), SigmaLoss()], optimizer=tfk.optimizers.legacy.Adam(lr))
    return m

def sigma(m, d): return float(m.evaluate(d['X_val'], d['y_val'], batch_size=1000, verbose=0)[1])

def fingerprint(data, BASIS):
    """Cheap, training-free features of a shape's reference geometry."""
    w = data['y_train'][:, 0]; om = data['y_train'][:, 1]
    zero = tfk.Sequential([tfk.Input(shape=(12,)), tfk.layers.Lambda(lambda x: 0*x[:, :1])])
    s_fs = sigma(wrap(zero, BASIS), data)              # Ricci error of the untrained reference metric
    return np.array([s_fs, np.std(np.log(w)), np.std(np.log(om)), np.max(w)/np.mean(w)])

def main():
    base = recipe(0)
    print("Sampling shapes (cached after first run)...")
    neigh  = [make_shape(f'neigh_{i}',  recipe(100+i, base)) for i in range(N_NEIGH)]
    held   = [make_shape(f'held_{i}',   recipe(200+i, base)) for i in range(N_HELD)]
    strang = [make_shape(f'strange_{i}', recipe(300+i))       for i in range(N_STRANGE)]

    # ---------- A. fingerprints
    print("\nA. FINGERPRINTS   [sigma_FS, std log w, std log |Omega|^2, max/mean w]")
    fps = {}
    for grp, shapes in [('neighbour', neigh), ('held-out', held), ('stranger', strang)]:
        for name, c, d, B in shapes:
            fps[name] = fingerprint(d, B)
            print(f"  {grp:9s} {name:10s} " + " ".join(f"{v:7.3f}" for v in fps[name]))
    F = np.array(list(fps.values())); mu, sd = F.mean(0), F.std(0)+1e-9
    ref = np.mean([(fps[n]-mu)/sd for n,_,_,_ in neigh], axis=0)
    print("  distance to neighbour-group centre (normalised):")
    for grp, shapes in [('held-out', held), ('stranger', strang)]:
        for name,_,_,_ in shapes:
            print(f"    {grp:9s} {name:10s} {np.linalg.norm((fps[name]-mu)/sd - ref):.2f}")

    # ---------- B. group specialist
    print(f"\nB. Training group specialist on {N_NEIGH} neighbours for {E_SPEC} epochs")
    core = new_core()
    spec = [wrap(core, B) for _,_,_,B in neigh]
    t0 = time.time()
    for ep in range(1, E_SPEC+1):
        for m, (_,_,d,_) in zip(spec, neigh):
            m.fit(d['X_train'], d['y_train'], batch_size=64, epochs=1, verbose=0)
        if ep % 5 == 0 or ep == 1:
            print(f"   epoch {ep:2d}  neighbours mean sigma {np.mean([sigma(m,d) for m,(_,_,d,_) in zip(spec,neigh)]):.3f}  ({time.time()-t0:.0f}s)")
    spec_w = core.get_weights()

    # ---------- C. warm start vs scratch
    print(f"\nC. WARM-START TEST ({E_WARM} epochs each)  -- sigma after each epoch")
    rows = []
    for grp, shapes in [('held-out', held), ('stranger', strang)]:
        for name, c, d, B in shapes:
            curves = {}
            for mode in ['scratch', 'warm']:
                k = new_core()
                if mode == 'warm': k.set_weights(spec_w)
                m = wrap(k, B)
                cur = [sigma(m, d)]
                for ep in range(E_WARM):
                    m.fit(d['X_train'], d['y_train'], batch_size=64, epochs=1, verbose=0)
                    cur.append(sigma(m, d))
                curves[mode] = cur
                rows.append([grp, name, mode] + cur)
            print(f"  {grp:9s} {name:10s}  scratch: " + " ".join(f"{v:.3f}" for v in curves['scratch']))
            print(f"  {'':9s} {'':10s}  warm:    " + " ".join(f"{v:.3f}" for v in curves['warm']))
    with open(LOG, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['group','shape','mode'] + [f'ep{i}' for i in range(E_WARM+1)]); w.writerows(rows)
    print(f"\nLog: {LOG}")
    print("Read: if 'warm' beats 'scratch' for held-out but not strangers, groups work and the fingerprint should route.")

if __name__ == '__main__':
    main()
