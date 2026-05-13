import os
import jax
import numpy as np
from tqdm import tqdm

from src.utils.perlin_noise import generate_terrain_jax

hfield_data = []
rng = jax.random.PRNGKey(42)
for i in tqdm(range(1024)):
    rng, key_terrain, key_scale, key_octaves, key_persistence, key_lacunarity = jax.random.split(rng, 6)
    scale = jax.random.uniform(key_scale, shape=(), minval=10.0, maxval=16.0)
    octaves = jax.random.randint(key_octaves, shape=(), minval=5, maxval=8)
    persistence = jax.random.uniform(key_persistence, shape=(), minval=0.3, maxval=0.5)
    lacunarity = jax.random.uniform(key_lacunarity, shape=(), minval=2.0, maxval=4.0)
    hfield_data.append(np.asarray(generate_terrain_jax(key_terrain, scale, octaves, persistence, lacunarity).flatten()))

hfield_data = np.stack(hfield_data, axis=0)
os.makedirs("data/hfield", exist_ok=True)
np.savez_compressed("data/hfield/terrain.npz", hfield_data=hfield_data)