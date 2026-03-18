import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
import matplotlib.pyplot as plt
from jax import jit

import time

SEED = 1
WIDTH = 100
HEIGHT = 100
STEPS = 1000
KERNEL_SIZE = 3

key = random.key(SEED)
key, subkey = random.split(key, 2)

grid = random.randint(subkey, (WIDTH, HEIGHT), 0, 2).astype(jnp.float32)

cx, cy = KERNEL_SIZE // 2, KERNEL_SIZE // 2
kernel = jnp.ones((KERNEL_SIZE, KERNEL_SIZE)).astype(jnp.float32)
kernel = kernel.at[cx, cy].set(0)




@jit
def step(grid, kernel):
    neighbours = jsp.signal.convolve(grid, kernel, "same")
    survives = (grid == 1) & ((neighbours == 3.0) | (neighbours == 2.0))
    born = (grid == 0) & (neighbours == 3.0)
    new_grid = jnp.where(survives | born, 1.0, 0)
    return new_grid

plt.ion()  # interactive mode on

for _ in range(STEPS):
    plt.imshow(grid, cmap='binary')
    plt.axis('off')
    plt.pause(0.1)
    plt.clf()
    grid = step(grid, kernel)




