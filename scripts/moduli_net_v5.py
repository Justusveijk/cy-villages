# v5: GP + specialists + a better router.
#   GP          = one generalist trained on ALL groups (the doctor for every village).
#   Specialists = one per group (loaded from v4 weights if present, else trained).
#   Router      = (1) "try-on": every doctor looks at the new shape, lowest error wins;
#                 (2) fingerprint v2: v4's 4 stats + gradient stats + coefficient-space distance.
#   Test        = every test shape fine-tuned from scratch / GP / routed specialist / best specialist.
# Uses the same cached shapes as v4 (moduli_data_v4). --quick for laptops.
import os, sys, csv, itertools, time, numpy as np, tensorflow as tf
tf.get_logger().setLevel('ERROR')
tfk = tf.keras
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.metrics import TotalLoss, SigmaLoss

QUICK = ('--quick' in sys.argv)
CFG = dict(
    GROUPS    = ['A', 'B'] if QUICK else ['A', 'B', 'C', 'D'],
    N_NEIGH   = 3  if QUICK else 12,
    N_HELD    = 1  if QUICK else 3,
    N_STRANGE = 1  if QUICK else 6,
    EPS       = 0.05,
    N_POINTS  = 2000 if QUICK else 50000,
    E_SPEC    = 2  if QUICK else 40,
    E_GP      = 2  if QUICK else 40,
    E_WARM    = 2  if QUICK else 15,
    HIDDEN    = 64 if QUICK else 256,
    LAYERS    = 3  if QUICK else 4,
    LR        = 1e-3, LR_WARM = 3e-4,
    CKPT      = 1 if QUICK else 5,
)
globals().update(CFG)
DATA = 'moduli_data_v4' if not QUICK else 'moduli_data_v5quick'; LOG = 'moduli_v5_log.csv'
print("CONFIG:", CFG, "\nGPUs:", tf.config.list_physical_devices('GPU'), flush=True)

def deg3(n): return [m for m in itertools.product(range(4), repeat=n) if sum(m)==3]
MONOMIALS = np.array([a+b for a in deg3(3) for b in deg3(3)], dtype=np.int64)
AMBIENT = np.array([2, 2]); KMODULI = np.ones(2)

def recipe(seed, base=None):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=100) + 1j*rng.normal(size=100); c /= np.linalg.norm(c)
    if base is not None: c = base + EPS*c; c /= np.linalg.norm(c)
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
    return tfk.Sequential([tfk.Input(shape=(12,))] + [tfk.layers.Dense(HIDDEN, activation='gelu') for _ in range(LAYERS)]
                          + [tfk.layers.Dense(1, use_bias=False)])
def wrap(core, BASIS, lr):
    m = PhiFSModel(core, BASIS, alpha=[1.,1.,1.,1.,1.])
    m.compile(custom_metrics=[TotalLoss(), SigmaLoss()], optimizer=tfk.optimizers.legacy.Adam(lr)); return m
def sigma(m, d): return float(m.evaluate(d['X_val'], d['y_val'], batch_size=2000, verbose=0)[1])
def two_phase_epoch(m, d):
    m.learn_volk = tf.cast(False, tf.bool); m.fit(d['X_train'], d['y_train'], batch_size=64, epochs=1, verbose=0)
    m.learn_volk = tf.cast(True, tf.bool);  m.fit(d['X_train'], d['y_train'], batch_size=min(10000, len(d['X_train'])), epochs=1, verbose=0)
def set_lr(ms, lr):
    for m in ms: m.optimizer.learning_rate.assign(lr)

# ---------- fingerprint v2
def poly_grad_stats(s):
    """How steep the defining equation is at sampled points — a crude curvature proxy."""
    X = s['d']['X_train'][:4000]; z = X[:, :6] + 1j*X[:, 6:]
    logz = np.log(z + 1e-12); mon = np.exp(logz @ MONOMIALS.T)                    # (n,100)
    grad = np.stack([(MONOMIALS[:, j] * s['c'][None, :] * mon / (z[:, j:j+1] + 1e-12)).sum(1) for j in range(6)], 1)
    g = np.linalg.norm(grad, axis=1)
    return [np.std(np.log(g + 1e-12)), np.mean(np.log(g + 1e-12))]
def fingerprint(s):
    w = s['d']['y_train'][:, 0]; om = s['d']['y_train'][:, 1]
    zero = tfk.Sequential([tfk.Input(shape=(12,)), tfk.layers.Lambda(lambda x: 0*x[:, :1])])
    return np.array([sigma(wrap(zero, s['B'], 1e-3), s['d']), np.std(np.log(w)), np.std(np.log(om)), np.max(w)/np.mean(w)] + poly_grad_stats(s))
def coef_dist(s, group):
    """distance in recipe space to the group's mean recipe (up to overall phase)."""
    mean = np.mean([g['c'] for g in group], 0)
    return min(np.linalg.norm(s['c']*np.exp(1j*t) - mean) for t in np.linspace(0, 2*np.pi, 64, endpoint=False))

def train_shared(group, epochs, tag, weights_file, init_from=None, lr=None):
    """Shared network over a group of shapes. Checkpoints every CKPT epochs; resumes if interrupted.
    init_from: optional weights to start from (the GP starts from a specialist, not from random)."""
    lr = lr or LR
    core = new_core()
    if init_from is not None: core.set_weights(init_from)
    if os.path.exists(weights_file):
        core.load_weights(weights_file); print(f"   [{tag}] loaded {weights_file}", flush=True); return core.get_weights()
    ckpt, prog = f'ckpt_{tag}.weights.h5', f'ckpt_{tag}.epoch'
    start = 0
    if os.path.exists(ckpt) and os.path.exists(prog):
        core.load_weights(ckpt); start = int(open(prog).read()); print(f"   [{tag}] resuming from epoch {start}", flush=True)
    ms = [wrap(core, s['B'], lr) for s in group]; t0 = time.time()
    for ep in range(start+1, epochs+1):
        set_lr(ms, lr * 0.5**(ep / (epochs/3)))
        for m, s in zip(ms, group): two_phase_epoch(m, s['d'])
        if ep in (1, 2, 5) or ep % 10 == 0 or ep == epochs:
            print(f"   [{tag}] epoch {ep:3d}  mean sigma {np.mean([sigma(m, s['d']) for m, s in zip(ms, group)]):.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if ep % CKPT == 0:
            core.save_weights(ckpt); open(prog, 'w').write(str(ep))
    core.save_weights(weights_file); return core.get_weights()

def finetune(s, init_w, lr):
    k = new_core()
    if init_w is not None: k.set_weights(init_w)
    m = wrap(k, s['B'], lr); cur = [sigma(m, s['d'])]
    for ep in range(1, E_WARM+1):
        set_lr([m], lr * 0.5**(ep / (E_WARM/3))); two_phase_epoch(m, s['d']); cur.append(sigma(m, s['d']))
    return cur

def main():
    print("Loading shapes...", flush=True)
    groups, tests = {}, []
    for g, tag in enumerate(GROUPS):
        base = recipe(g)
        groups[tag] = [make_shape(f'{tag}_{i}', recipe(100 + 50*g + i, base)) for i in range(N_NEIGH)]
        tests += [(f'held{tag}', make_shape(f'held{tag}_{i}', recipe(200 + 50*g + i, base))) for i in range(N_HELD)]
    tests += [('stranger', make_shape(f'strange_{i}', recipe(300 + i))) for i in range(N_STRANGE)]
    allshapes = [s for grp in groups.values() for s in grp]

    print(f"\nB. DOCTORS  (specialists: {E_SPEC} ep each, GP: {E_GP} ep on all {len(allshapes)} shapes)", flush=True)
    W = {tag: train_shared(grp, E_SPEC, tag, f'specialist_{tag}.weights.h5') for tag, grp in groups.items()}
    # GP v2: start from specialist A (already ~0.10 on strangers) instead of random, train gently on all villages
    W_GP = train_shared(allshapes, E_GP, 'GP', 'generalist.weights.h5', init_from=W['A'], lr=LR_WARM)

    print("\nA. ROUTER: try-on (zero-shot error under each doctor)  +  fingerprint v2  +  recipe distance", flush=True)
    for s in allshapes + [t for _, t in tests]: s['fp'] = fingerprint(s)
    F = np.array([s['fp'] for s in allshapes]); mu, sd = F.mean(0), F.std(0) + 1e-9
    z = lambda s: (s['fp'] - mu) / sd
    centres = {tag: np.mean([z(s) for s in grp], 0) for tag, grp in groups.items()}
    hdr = f"   {'kind':9s} {'shape':10s} | try-on: " + " ".join(f"{t:>6s}" for t in GROUPS) + f" {'GP':>6s} | fp-route  recipe-route  tryon-route"
    print(hdr, flush=True)
    for kind, s in tests:
        tryon = {tag: sigma(wrap((lambda k: (k.set_weights(W[tag]), k)[1])(new_core()), s['B'], LR), s['d']) for tag in GROUPS}
        tryon_gp = sigma(wrap((lambda k: (k.set_weights(W_GP), k)[1])(new_core()), s['B'], LR), s['d'])
        fp_route  = min(GROUPS, key=lambda t: np.linalg.norm(z(s) - centres[t]))
        rec_route = min(GROUPS, key=lambda t: coef_dist(s, groups[t]))
        try_route = min(GROUPS, key=lambda t: tryon[t])
        s['route'] = try_route; s['tryon'] = tryon; s['tryon_gp'] = tryon_gp
        truth = kind[4:] if kind.startswith('held') else '-'
        print(f"   {kind:9s} {s['name']:10s} | " + " ".join(f"{tryon[t]:6.3f}" for t in GROUPS) + f" {tryon_gp:6.3f} |   {fp_route}   {'✓' if fp_route==truth else ('·' if truth=='-' else '✗')}      {rec_route}   {'✓' if rec_route==truth else ('·' if truth=='-' else '✗')}         {try_route}   {'✓' if try_route==truth else ('·' if truth=='-' else '✗')}", flush=True)

    print(f"\nC. TREATMENT: scratch / GP / routed specialist  ({E_WARM} epochs)  -> start, best", flush=True)
    done = set()
    if os.path.exists(LOG):
        done = {tuple(r[:2]) for r in csv.reader(open(LOG)) if r and r[0] != 'group'}
    else:
        with open(LOG, 'w', newline='') as f:
            csv.writer(f).writerow(['group', 'shape', 'mode'] + [f'ep{i}' for i in range(E_WARM+1)])
    for kind, s in tests:
        if (kind, s['name']) in done:
            print(f"   {kind:9s} {s['name']:10s} already in log, skipping", flush=True); continue
        curves = {'scratch': finetune(s, None, LR), 'GP': finetune(s, W_GP, LR_WARM), f'spec-{s["route"]}': finetune(s, W[s['route']], LR_WARM)}
        print(f"   {kind:9s} {s['name']:10s} " + "  ".join(f"{k}: {c[0]:.3f}->{min(c):.3f}" for k, c in curves.items()), flush=True)
        with open(LOG, 'a', newline='') as f:
            w = csv.writer(f)
            for k, c in curves.items(): w.writerow([kind, s['name'], k] + c)
    print(f"\nLog: {LOG}\nRead: (A) which router column gets the ✓s.  (C) GP vs specialist vs scratch per shape; strangers are the villages without a doctor.")

if __name__ == '__main__':
    main()
