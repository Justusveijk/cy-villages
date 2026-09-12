# Bicubic Calabi-Yau: one equation of bidegree (3,3) in P2 x P2.
# Two "spaces" stitched together -> the first "messy" shape.
import os, itertools, numpy as np, tensorflow as tf
tf.get_logger().setLevel('ERROR')
tfk = tf.keras
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.callbacks import SigmaCallback
from cymetric.models.metrics import TotalLoss, SigmaLoss

def main():
    ambient = np.array([2, 2])                # P2 x P2
    # all monomials z0^a z1^b z2^c * w0^d w1^e w2^f with a+b+c=3 and d+e+f=3
    def deg3(n):
        return [m for m in itertools.product(range(4), repeat=n) if sum(m)==3]
    monomials = np.array([a+b for a in deg3(3) for b in deg3(3)], dtype=np.int64)  # 100 x 6
    rng = np.random.default_rng(0)
    coefficients = rng.normal(size=len(monomials)) + 1j*rng.normal(size=len(monomials))
    kmoduli = np.ones(2)                       # one size dial per P2

    n_p = 50000                                # points on the shape (raise to 100k for real runs)
    dirname = 'bicubic_data'
    pg = CICYPointGenerator([monomials], [coefficients], kmoduli, ambient)
    kappa = pg.prepare_dataset(n_p, dirname)
    pg.prepare_basis(dirname, kappa=kappa)
    data = np.load(os.path.join(dirname, 'dataset.npz'))
    BASIS = prepare_tf_basis(np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True))
    print(f"points: {len(data['X_train'])} train / {len(data['X_val'])} val   monomials: {len(monomials)}")

    # neural network: 12 real inputs (6 complex coords) -> 1 number, the Kahler potential correction phi
    nn = tfk.Sequential([tfk.Input(shape=(12,)),
                         tfk.layers.Dense(64, activation='gelu'),
                         tfk.layers.Dense(64, activation='gelu'),
                         tfk.layers.Dense(64, activation='gelu'),
                         tfk.layers.Dense(1, use_bias=False)])
    model = PhiFSModel(nn, BASIS, alpha=[1.,1.,1.,1.,1.])
    scb = SigmaCallback((data['X_val'], data['y_val']))
    model, hist = train_model(model, data, optimizer=tfk.optimizers.legacy.Adam(), epochs=50,
                              batch_sizes=[64, 5000], verbose=0,
                              custom_metrics=[TotalLoss(), SigmaLoss()], callbacks=[scb])
    print("\nRicci-flatness (sigma) error per epoch:")
    for i, s in enumerate(hist['sigma_val']):
        print(f"  epoch {i+1:2d}: {s:.4f}")

if __name__ == '__main__':
    main()
