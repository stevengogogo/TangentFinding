#%%
import jax 
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from jaxopt import Bisection
from jax.scipy import optimize

def plot_fn(fn, x_range, ngrid=1000,**kwargs):
    x = jnp.linspace(x_range[0], x_range[1], 1000)
    y = jax.vmap(fn)(x)
    plt.plot(x, y, **kwargs)

#test function 
def target_fn(x):
    x_ = x + 1e-8
    S = (1-x_)**2 / (x_**2)
    x1 = 1 / (1 + S)
    x2 = S / (1 + S)
    return  x1

dfn = jax.grad(target_fn)

def object_fn(x):
    tan = (target_fn(x) - target_fn(0)) / (x - 0)
    #print(x)
    return jnp.sum(tan - dfn(x))

bisec = Bisection(object_fn,lower=0., upper=1.)

print(bisec.run().params)



plot_fn(target_fn, [0, 1], label='target_fn')
plot_fn(dfn, [0, 1], label='dtarget_fn')
plot_fn(object_fn, [0, 1], label='df - tangent')
plt.legend()

# %%
