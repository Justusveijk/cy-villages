# Stage-2 prototype v4: TWO groups, a fingerprint ROUTER, and proper training.
#   - Two base recipes A and B, each with a neighbourhood of perturbations.
#   - One specialist per group, trained with cymetric's two-phase scheme + LR decay.
#   - Fingerprint router: a new shape is sent to the nearest group centre.
#   - Test shapes: held-out neighbours of A, of B, and strangers.
#     Each is trained three ways: scratch / warm from ROUTED specialist / warm from the OTHER specialist.
#   Set QUICK=True for a laptop smoke test; QUICK=False for the real (GPU) run.
import os, sys, csv, itertools, time, numpy as np, tensorflow as tf
tf.get_logger().setLevel('ERROR')
tfk = tf.keras
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.metrics import TotalLoss, SigmaLoss

QUICK = ('--quick' in sys.argv)
CFG = dict(
    N_NEIGH   = 4  if QUICK else 12,     # training neighbours per group
    N_HELD    = 1  if QUICK else 3,      # held-out neighbours per group
    N_STRANGE = 1  if QUICK else 3,
    EPS       = 0.05,
    N_POINTS  = 2000 if QUICK else 50000,
    E_SPEC    = 4  if QUICK else 100,    # specialist epochs
    E_WARM    = 3  if QUICK else 30,     # fine-tune / scratch epochs
    HIDDEN    = 64 if QUICK else 256,
    LAYERS    = 3  if QUICK else 4,
    LR        = 1e-3,
    LR_WARM   = 3e-4,                    # gentler LR when fine-tuning from a specialist
    CKPT      = 2 if QUICK else 10,      # checkpoint every N specialist epochs
)
globals().update(CFG)
DATA = 'moduli_data_v4'; LOG = 'moduli_v4_log.csv'
print("CONFIG:", CFG, "\nGPUs:", tf.config.list_physical_devices('GPU'))

def deg3(n): return [m for m in itertools.product(range(4), repeat=n) if sum(m)==3]
MONOMIALS = np.array([a+b for a in deg3(3) for b in deg3(3)], dtype=np.int64)
AMBIENT   = np.array([2, 2]); KMODULI = np.ones(2)

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
    data  = np.load(os.path.join(d, 'dataset.npz'))
    BASIS = prepare_tf_basis(np.load(os.path.join(d, 'basis.pickle'), allow_pickle=True))
    return dict(name=name, c=coeffs, d=data, B=BASIS)

def new_core():
    layers = [tfk.Input(shape=(12,))] + [tfk.layers.Dense(HIDDEN, activation='gelu') for _ in range(LAYERS)]
    return tfk.Sequential(layers + [tfk.layers.Dense(1, use_bias=False)])

def wrap(core, BASIS, lr):
    m = PhiFSModel(core, BASIS, alpha=[1.,1.,1.,1.,1.])
    m.compile(custom_metrics=[TotalLoss(), SigmaLoss()], optimizer=tfk.optimizers.legacy.Adam(lr))
    return m

def sigma(m, d): return float(m.evaluate(d['X_val'], d['y_val'], batch_size=2000, verbose=0)[1])

def two_phase_epoch(m, d):
    """cymetric's recipe: small batches w/o volume loss, then large batches with it."""
    m.learn_volk = tf.cast(False, tf.bool)
    m.fit(d['X_train'], d['y_train'], batch_size=64, epochs=1, verbose=0)
    m.learn_volk = tf.cast(True, tf.bool)
    m.fit(d['X_train'], d['y_train'], batch_size=min(10000, len(d['X_train'])), epochs=1, verbose=0)

def set_lr(models, lr):
    for m in models: m.optimizer.learning_rate.assign(lr)

def fingerprint(s):
    w = s['d']['y_train'][:, 0]; om = s['d']['y_train'][:, 1]
    zero = tfk.Sequential([tfk.Input(shape=(12,)), tfk.layers.Lambda(lambda x: 0*x[:, :1])])
    return np.array([sigma(wrap(zero, s['B'], 1e-3), s['d']), np.std(np.log(w)), np.std(np.log(om)), np.max(w)/np.mean(w)])

def train_specialist(group, tag):
    """Trains with a checkpoint every CKPT epochs; resumes from the last checkpoint if present."""
    core = new_core(); ms = [wrap(core, s['B'], LR) for s in group]
    final, ckpt, prog = f'specialist_{tag}.weights.h5', f'ckpt_{tag}.weights.h5', f'ckpt_{tag}.epoch'
    if os.path.exists(final):
        core.load_weights(final); print(f"   [{tag}] already trained, loaded {final}", flush=True); return core.get_weights()
    start = 0
    if os.path.exists(ckpt) and os.path.exists(prog):
        core.load_weights(ckpt); start = int(open(prog).read()); print(f"   [{tag}] resuming from epoch {start}", flush=True)
    t0 = time.time()
    for ep in range(start+1, E_SPEC+1):
        set_lr(ms, LR * 0.5**(ep / (E_SPEC/3)))
        for m, s in zip(ms, group): two_phase_epoch(m, s['d'])
        if ep in (1, 2, 5) or ep % 10 == 0 or ep == E_SPEC:
            print(f"   [{tag}] epoch {ep:3d}  mean sigma {np.mean([sigma(m, s['d']) for m, s in zip(ms, group)]):.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if ep % CKPT == 0:
            core.save_weights(ckpt); open(prog, 'w').write(str(ep))
    core.save_weights(final)
    return core.get_weights()

def finetune(s, init_w, lr, tag):
    k = new_core()
    if init_w is not None: k.set_weights(init_w)
    m = wrap(k, s['B'], lr); cur = [sigma(m, s['d'])]
    for ep in range(1, E_WARM+1):
        set_lr([m], lr * 0.5**(ep / (E_WARM/3)))
        two_phase_epoch(m, s['d']); cur.append(sigma(m, s['d']))
    return cur

def main():
    print("Sampling shapes (cached after first run)...", flush=True)
    baseA, baseB = recipe(0), recipe(1)
    grpA = [make_shape(f'A_{i}', recipe(100+i, baseA)) for i in range(N_NEIGH)]
    grpB = [make_shape(f'B_{i}', recipe(150+i, baseB)) for i in range(N_NEIGH)]
    tests = ([('heldA', make_shape(f'heldA_{i}', recipe(200+i, baseA))) for i in range(N_HELD)] +
             [('heldB', make_shape(f'heldB_{i}', recipe(250+i, baseB))) for i in range(N_HELD)] +
             [('stranger', make_shape(f'strange_{i}', recipe(300+i))) for i in range(N_STRANGE)])

    print("\nA. FINGERPRINTS + ROUTER", flush=True)
    for s in grpA + grpB + [t for _, t in tests]: s['fp'] = fingerprint(s)
    allfp = np.array([s['fp'] for s in grpA + grpB]); mu, sd = allfp.mean(0), allfp.std(0) + 1e-9
    z = lambda s: (s['fp'] - mu) / sd
    cA, cB = np.mean([z(s) for s in grpA], 0), np.mean([z(s) for s in grpB], 0)
    print(f"   group centres are {np.linalg.norm(cA-cB):.2f} apart (normalised units)")
    for kind, s in tests:
        dA, dB = np.linalg.norm(z(s)-cA), np.linalg.norm(z(s)-cB)
        s['route'] = 'A' if dA < dB else 'B'
        print(f"   {kind:9s} {s['name']:10s} dist A {dA:5.2f}  dist B {dB:5.2f}  -> routed to {s['route']}", flush=True)

    print(f"\nB. SPECIALISTS ({N_NEIGH} shapes each, {E_SPEC} epochs)", flush=True)
    wA = train_specialist(grpA, 'A'); wB = train_specialist(grpB, 'B')
    W = {'A': wA, 'B': wB}

    print(f"\nC. TEST SHAPES: scratch vs warm(routed) vs warm(other), {E_WARM} epochs", flush=True)
    done = set()
    if os.path.exists(LOG):
        done = {tuple(r[:2]) for r in csv.reader(open(LOG)) if r and r[0] != 'group'}
    else:
        with open(LOG, 'w', newline='') as f:
            csv.writer(f).writerow(['group', 'shape', 'mode'] + [f'ep{i}' for i in range(E_WARM+1)])
    for kind, s in tests:
        if (kind, s['name']) in done:
            print(f"   {kind:9s} {s['name']:10s} already in log, skipping", flush=True); continue
        r, o = s['route'], ('B' if s['route'] == 'A' else 'A')
        curves = {'scratch': finetune(s, None, LR, 'scratch'),
                  f'warm-{r} (routed)': finetune(s, W[r], LR_WARM, 'routed'),
                  f'warm-{o} (other)':  finetune(s, W[o], LR_WARM, 'other')}
        with open(LOG, 'a', newline='') as f:
            w = csv.writer(f)
            for mode, cur in curves.items():
                w.writerow([kind, s['name'], mode] + cur)
                print(f"   {kind:9s} {s['name']:10s} {mode:18s} start {cur[0]:.3f}  best {min(cur):.3f}  end {cur[-1]:.3f}", flush=True)
    print(f"\nLog: {LOG}. Read: routed should beat other AND scratch for held-out shapes; strangers show how far the specialists generalise.")

if __name__ == '__main__':
    main()
