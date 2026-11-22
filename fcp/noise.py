"""Reproducible fixed noise for archived Cube windows."""

import math

import numpy as np
import torch


def cube_window_noise(start, count, seed, shape, device):
    size = math.prod(shape)
    ids = np.arange(start, start + count, dtype=np.uint64)
    counter = ids[:, None] * np.uint64(size + 17) + np.arange(size, dtype=np.uint64)

    def uniform(stream):
        with np.errstate(over='ignore'):
            x = counter + np.uint64(seed) + np.uint64(stream) * np.uint64(0x9E3779B97F4A7C15)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x ^= x >> np.uint64(31)
        return ((x >> np.uint64(11)).astype(np.float64) + .5) / 2**53

    noise = np.sqrt(-2 * np.log(uniform(1))) * np.cos(2 * np.pi * uniform(2))
    return torch.from_numpy(noise.astype(np.float32).reshape(count, *shape)).to(device)
