import jax, time
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
import cyjax

rns = cyjax.util.PRNGSequence(42)
dwork = cyjax.Dwork(3)            # quintic family: z1^5+...+z5^5 - 5*psi*z1 z2 z3 z4 z5 = 0
psi = jnp.array([10+3j])          # pick a shape in the family
degree = 4

volcy = dwork.compute_vol(next(rns), psi, batch_size=2000)
basis = cyjax.donaldson.MonomialBasisReduced(dwork, degree, psi)
metric = cyjax.donaldson.AlgebraicMetric(dwork, basis)
print(f"psi={psi[0]}  basis size={basis.size}  vol={float(volcy):.3f}")

n_samples = (10*basis.size**2 + 50000)//5
bs=1000; batches=n_samples//bs+1
step = jax.jit(partial(metric.donaldson_step, params=psi, vol_cy=volcy, batches=batches, batch_size=bs))

h = jnp.eye(basis.size, dtype=complex)
print(f"iter 0  sigma error = {metric.sigma_accuracy(next(rns), psi, h, 1000).item():.4f}")
t0=time.time()
for i in range(1,13):
    h = (h + h.conj().T)/2
    h = step(next(rns), h); h = h/jnp.max(jnp.abs(h))
    if i in (1,2,4,8,12):
        print(f"iter {i:2d}  sigma error = {metric.sigma_accuracy(next(rns), psi, h, 1000).item():.4f}   ({time.time()-t0:.0f}s)")
