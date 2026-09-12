# Stage-2 prototype: ONE network that learns the metric for MANY bicubic shapes.
# Input = point on shape (12 numbers) + the shape's recipe (100 complex coeffs = 200 numbers).
# The recipe is glued onto every point, so the net must learn how geometry changes with the recipe.
import os, itertools, time, numpy as np, tensorflow as tf
tf.get_logger().setLevel('ERROR')
tfk = tf.keras
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.metrics import TotalLoss, SigmaLoss

N_SHAPES  = 4        # how many random bicubics to train on (+1 unseen for testing)
N_POINTS  = 20000    # points per shape (raise to 20000 on your Mac)
EPOCHS    = 30
HIDDEN    = 128

def deg3(n):
    return [m for m in itertools.product(range(4), repeat=n) if sum(m)==3]
MONOMIALS = np.array([a+b for a in deg3(3) for b in deg3(3)], dtype=np.int64)
AMBIENT   = np.array([2, 2]); KMODULI = np.ones(2)

def make_shape(seed):
    """One random bicubic. Returns (coefficients, data, BASIS)."""
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=100) + 1j*rng.normal(size=100)
    coeffs /= np.linalg.norm(coeffs)                      # normalise so recipes are comparable
    d = f'moduli_data/shape_{seed}'
    if not os.path.exists(os.path.join(d, 'basis.pickle')):
        pg = CICYPointGenerator([MONOMIALS], [coeffs], KMODULI, AMBIENT, verbose=0)
        kappa = pg.prepare_dataset(N_POINTS, d)
        pg.prepare_basis(d, kappa=kappa)
    data  = np.load(os.path.join(d, 'dataset.npz'))
    BASIS = prepare_tf_basis(np.load(os.path.join(d, 'basis.pickle'), allow_pickle=True))
    vec = np.concatenate([coeffs.real, coeffs.imag]).astype(np.float32)   # 200 numbers
    return vec, data, BASIS

class GlueRecipe(tfk.layers.Layer):
    """Appends this shape's fixed 200-number recipe to every input point."""
    def __init__(self, vec, **kw):
        super().__init__(**kw); self.vec = tf.constant(vec)
    def call(self, x):
        return tf.concat([x, tf.tile(self.vec[None,:], [tf.shape(x)[0], 1])], axis=1)

def main():
    # ---- the ONE shared brain: 212 in -> 1 out
    core = tfk.Sequential([tfk.Input(shape=(12+200,)),
                           tfk.layers.Dense(HIDDEN, activation='gelu'),
                           tfk.layers.Dense(HIDDEN, activation='gelu'),
                           tfk.layers.Dense(HIDDEN, activation='gelu'),
                           tfk.layers.Dense(1, use_bias=False)], name='shared_core')

    def wrap(vec, BASIS):
        inp = tfk.Input(shape=(12,))
        nn  = tfk.Model(inp, core(GlueRecipe(vec)(inp)))
        m   = PhiFSModel(nn, BASIS, alpha=[1.,1.,1.,1.,1.])
        m.compile(custom_metrics=[TotalLoss(), SigmaLoss()], optimizer=tfk.optimizers.legacy.Adam())
        return m

    print("Generating shapes (slow the first time)...")
    shapes = [make_shape(s) for s in range(N_SHAPES)]
    test_vec, test_data, test_BASIS = make_shape(999)         # never trained on
    models = [wrap(v, B) for v, _, B in shapes]
    test_model = wrap(test_vec, test_BASIS)

    def sigma(m, d):
        return float(m.evaluate(d['X_val'], d['y_val'], batch_size=1000, verbose=0)[1])

    print(f"\n{'epoch':>5} | " + " ".join(f"shape{i}" for i in range(N_SHAPES)) + " | UNSEEN")
    row = lambda e: f"{e:>5} | " + " ".join(f"{sigma(m,d):6.3f}" for m,(_,d,_) in zip(models,shapes)) + f" | {sigma(test_model,test_data):6.3f}"
    print(row(0)); t0 = time.time()
    for ep in range(1, EPOCHS+1):
        for m, (_, d, _) in zip(models, shapes):        # one pass over each shape, same core underneath
            m.fit(d['X_train'], d['y_train'], batch_size=64, epochs=1, verbose=0)
        print(row(ep), f"  ({time.time()-t0:.0f}s)")

    print("\nColumns 0..N: shapes it trained on.  UNSEEN: a recipe it never saw.")
    print("If UNSEEN drops too, the net is learning recipe -> geometry, not memorising.")

if __name__ == '__main__':
    main()
