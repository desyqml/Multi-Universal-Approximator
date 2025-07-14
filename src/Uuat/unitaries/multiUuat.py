from jax import vmap

from .Uuat import Uuat

multi_Uuat = vmap(Uuat, in_axes=(None, 0, 0, 0))