import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
import cyjax
rns = cyjax.util.PRNGSequence(1)
dwork = cyjax.Dwork(3)
def run(psi_val, degree):
    psi = jnp.array([psi_val])
    volcy = dwork.compute_vol(next(rns), psi, batch_size=2000)
    basis = cyjax.donaldson.MonomialBasisReduced(dwork, degree, psi)
    metric = cyjax.donaldson.AlgebraicMetric(dwork, basis)
    n=(10*basis.size**2+50000)//5; bs=1000; b=n//bs+1
    step = jax.jit(partial(metric.donaldson_step, params=psi, vol_cy=volcy, batches=b, batch_size=bs))
    h = jnp.eye(basis.size, dtype=complex)
    for _ in range(8):
        h=(h+h.conj().T)/2; h=step(next(rns),h); h=h/jnp.max(jnp.abs(h))
    return metric.sigma_accuracy(next(rns), psi, h, 1000).item()
print("shape (psi)   deg3    deg4")
for p in [0+0j, 0.5+0j, 2+0j, 10+3j, 100+0j]:
    print(f"{str(p):>10}  {run(p,3):.3f}  {run(p,4):.3f}")
