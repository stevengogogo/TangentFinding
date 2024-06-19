#%%
import jax
from jaxopt import Bisection


f = jax.jit(lambda x: x**3-100*x**2-x+100)

bisec = Bisection(optimality_fun=f, lower=2, upper=80)
print(bisec.run().params)


# %%
