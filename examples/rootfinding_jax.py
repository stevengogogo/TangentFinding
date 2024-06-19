#same as Rootfinding example 
#https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter19.05-Root-Finding-in-Python.html
#%%
import jax
import jax.numpy as jnp
import optimistix as optx

def fn(x, args):
    return x**3-100*x**2-x+100


solver = optx.Newton(rtol=1e-8, atol=1e-8)
y0 = jnp.array([2., 80.])
sol = optx.root_find(fn, solver, y0)
print(sol.value)
# %%
