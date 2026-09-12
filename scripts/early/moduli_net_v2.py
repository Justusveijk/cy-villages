# Stage-2 prototype v2: one network for MANY bicubic shapes, built to generalise.
#   1. Recipe-point interaction features: each recipe term evaluated at the point.
#   2. FiLM conditioning: the recipe generates scale/shift for every hidden layer.
#   3. Many training shapes, several unseen test shapes, dropout, CSV log.
import os, csv, itertools, time, numpy as np, tensorflow as tf
tf.get_logger().setLevel('ERROR')
tfk = tf.keras
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.metrics import TotalLoss, SigmaLoss

N_TRAIN   = 16       # training shapes
N_TEST    = 2        # unseen shapes (never trained on)
N_POINTS  = 10000    # points per shape
EPOCHS    = 40
HIDDEN    = 128
DROPOUT   = 0.1
LOG       = 'moduli_v2_log.csv'

def deg3(n): return [m for m in itertools.product(range(4), repeat=n) if sum(m)==3]
MONOMIALS = np.array([a+b for a in deg3(3) for b in deg3(3)], dtype=np.int64)   # 100 x 6
AMBIENT   = np.array([2, 2]); KMODULI = np.ones(2)
MON_TF    = tf.constant(MONOMIALS, dtype=tf.complex64)

def make_shape(seed):
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=100) + 1j*rng.normal(size=100)
    coeffs /= np.linalg.norm(coeffs)
    d = f'moduli_data/shape_{seed}'
    if not os.path.exists(os.path.join(d, 'basis.pickle')):
        pg = CICYPointGenerator([MONOMIALS], [coeffs], KMODULI, AMBIENT, verbose=0)
        kappa = pg.prepare_dataset(N_POINTS, d)
        pg.prepare_basis(d, kappa=kappa)
    data  = np.load(os.path.join(d, 'dataset.npz'))
    BASIS = prepare_tf_basis(np.load(os.path.join(d, 'basis.pickle'), allow_pickle=True))
    return coeffs.astype(np.complex64), data, BASIS

class RecipeFeatures(tfk.layers.Layer):
    """For a fixed recipe c: input point x -> [x, c*monomials(x) as re/im, c as re/im]."""
    def __init__(self, coeffs, **kw):
        super().__init__(**kw)
        self.c = tf.constant(coeffs)                                   # (100,) complex
        self.cvec = tf.constant(np.concatenate([coeffs.real, coeffs.imag]).astype(np.float32))
    def call(self, x):
        z = tf.complex(x[:, :6], x[:, 6:])                             # (b,6)
        logz = tf.math.log(z + tf.cast(1e-12, tf.complex64))
        mon = tf.exp(tf.matmul(logz, tf.transpose(MON_TF)))            # (b,100) = prod z_j^m_j
        f = self.c[None, :] * mon                                      # recipe term fired at x
        feat = tf.concat([tf.math.real(f), tf.math.imag(f)], axis=1)   # (b,200)
        cond = tf.tile(self.cvec[None, :], [tf.shape(x)[0], 1])        # (b,200)
        return tf.concat([x, feat], axis=1), cond

class FiLMCore(tfk.Model):
    """Point-net whose hidden layers are scaled & shifted by a recipe-net (hypernetwork-lite)."""
    def __init__(self, n_layers=3, hidden=HIDDEN, **kw):
        super().__init__(**kw)
        self.hyper = tfk.Sequential([tfk.layers.Dense(hidden, activation='gelu'),
                                     tfk.layers.Dense(2*hidden*n_layers)], name='recipe_net')
        self.dense = [tfk.layers.Dense(hidden) for _ in range(n_layers)]
        self.drop  = tfk.layers.Dropout(DROPOUT)
        self.out   = tfk.layers.Dense(1, use_bias=False)
        self.n, self.h = n_layers, hidden
    def call(self, inputs, training=False):
        x, cond = inputs
        gam_bet = tf.reshape(self.hyper(cond), [-1, self.n, 2, self.h])
        for i, layer in enumerate(self.dense):
            x = layer(x)
            x = x * (1.0 + gam_bet[:, i, 0]) + gam_bet[:, i, 1]         # FiLM
            x = tf.nn.gelu(x)
            x = self.drop(x, training=training)
        return self.out(x)

def main():
    core = FiLMCore()
    def wrap(coeffs, BASIS):
        inp = tfk.Input(shape=(12,))
        nn  = tfk.Model(inp, core(RecipeFeatures(coeffs)(inp)))
        m   = PhiFSModel(nn, BASIS, alpha=[1.,1.,1.,1.,1.])
        m.compile(custom_metrics=[TotalLoss(), SigmaLoss()], optimizer=tfk.optimizers.legacy.Adam(1e-3))
        return m

    print(f"Generating {N_TRAIN+N_TEST} shapes (cached after first run)...")
    train = [make_shape(s) for s in range(N_TRAIN)]
    test  = [make_shape(900+s) for s in range(N_TEST)]
    tm = [wrap(c, B) for c, _, B in train]
    sm = [wrap(c, B) for c, _, B in test]
    sig = lambda m, d: float(m.evaluate(d['X_val'], d['y_val'], batch_size=1000, verbose=0)[1])

    def row(ep):
        tr = [sig(m, d) for m, (_, d, _) in zip(tm, train)]
        te = [sig(m, d) for m, (_, d, _) in zip(sm, test)]
        return tr, te
    with open(LOG, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['epoch', 'train_mean', 'train_min', 'train_max'] + [f'unseen{i}' for i in range(N_TEST)])
        print(f"{'ep':>3} | train mean (min–max)   | " + "  ".join(f"UNSEEN{i}" for i in range(N_TEST)))
        t0 = time.time()
        for ep in range(EPOCHS+1):
            if ep > 0:
                for m, (_, d, _) in zip(tm, train):
                    m.fit(d['X_train'], d['y_train'], batch_size=64, epochs=1, verbose=0)
            tr, te = row(ep)
            w.writerow([ep, np.mean(tr), min(tr), max(tr)] + te); f.flush()
            print(f"{ep:>3} | {np.mean(tr):.3f} ({min(tr):.3f}–{max(tr):.3f}) | " + "  ".join(f"{v:.3f}  " for v in te), f"({time.time()-t0:.0f}s)")
    core.save_weights('moduli_v2_core.weights.h5')
    print(f"\nLog: {LOG}   Weights: moduli_v2_core.weights.h5")
    print("Watch UNSEEN vs train mean. Falling together = learning the map. Diverging = memorising.")

if __name__ == '__main__':
    main()
