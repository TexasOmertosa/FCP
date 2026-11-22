"""Memory-mapped demonstration rows for action-prior training."""

import json
from pathlib import Path

import numpy as np
import torch

from priors.config import ACTION_BLOCK, ACTION_DIM, ACTION_TOKENS, HISTORY_FRAMES, LIBERO, LOWDIM_DIM
from release import CONFIGS, sha256_file


class PriorRows:
    def __init__(self, directory, environment):
        root = Path(directory)
        spec = CONFIGS[environment]['architecture']
        dim = spec['latent_dim']
        self.array_paths = {name: root / f'{name}.npy' for name in ('history', 'goal', 'actions')}
        lowdim = LOWDIM_DIM[environment]
        if lowdim:
            self.array_paths['lowdim'] = root / 'lowdim.npy'
        if environment not in LIBERO:
            self.array_paths['goal_offset'] = root / 'goal_offset.npy'
        if (root / 'upper_history.npy').exists():
            self.array_paths['upper_history'] = root / 'upper_history.npy'
        self.arrays = {k: np.load(path, mmap_mode='r', allow_pickle=False)
                       for k, path in self.array_paths.items()}
        n = len(self.arrays['history'])
        shapes = dict(history=(n, HISTORY_FRAMES, dim), goal=(n, dim),
                      actions=(n, ACTION_BLOCK * ACTION_TOKENS, ACTION_DIM[environment]),
                      lowdim=(n, lowdim), goal_offset=(n,),
                      upper_history=(n, spec['history_frames'], dim))
        if n < 1 or any(a.shape != shapes[k] for k, a in self.arrays.items()):
            raise ValueError(f'Demonstration prior rows have incorrect shapes in {root}.')
        self.environment = environment
        self.upper_frames = spec['history_frames']
        self.size = n

    def signature(self):
        return {name: sha256_file(path) for name, path in self.array_paths.items()}

    def inputs_for_upper(self, indices, device):
        history = self.arrays.get('upper_history', self.arrays['history'])
        return (as_tensor(history[indices, -self.upper_frames:], device),
                as_tensor(self.arrays['goal'][indices], device))

    def batch(self, local, indices, stats, device):
        history = as_tensor(self.arrays['history'][indices], device)
        far = as_tensor(self.arrays['goal'][indices], device)[:, None]
        near = as_tensor(local[indices], device)[:, None]
        actions = as_tensor(self.arrays['actions'][indices], device)
        scale_key = 'action_scale' if self.environment in LIBERO else 'action_std'
        actions = ((actions - stats['action_mean']) / stats[scale_key]).flatten(1)
        actions = actions.reshape(len(indices), ACTION_TOKENS, -1)
        if LOWDIM_DIM[self.environment]:
            lowdim = as_tensor(self.arrays['lowdim'][indices], device)
            lowdim = (lowdim - stats['lowdim_mean']) / stats['lowdim_std']
            offsets = as_tensor(self.arrays['goal_offset'][indices], device)
        else:
            lowdim = history.new_empty(len(indices), 0)
            offsets = None
        return history, near, far, lowdim, offsets, actions


def as_tensor(array, device):
    return torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(device)


def load_condition(directory, rows, environment, variant):
    directory = Path(directory)
    metadata = json.loads((directory / 'manifest.json').read_text())
    if (metadata['environment'], metadata['variant']) != (environment, variant):
        raise ValueError('Condition cache was made with a different FCP variant.')
    if rows.signature() != metadata['data_sha256']:
        raise ValueError('Condition cache does not match the demonstration rows.')
    path = directory / 'local.npy'
    if sha256_file(path) != metadata['cache_sha256']:
        raise ValueError('Invalid condition cache checksum.')
    local = np.load(path, mmap_mode='r', allow_pickle=False)
    if local.shape != rows.arrays['goal'].shape:
        raise ValueError('Incorrect local-goal shape in the condition cache.')
    return local, metadata


def compute_moments(array):
    width = array.shape[-1]
    count = 0
    total = np.zeros(width)
    squares = np.zeros(width)
    for index in range(0, len(array), 8192):
        block = np.asarray(array[index:index + 8192], dtype=np.float64).reshape(-1, width)
        count += len(block)
        total += block.sum(axis=0)
        squares += np.square(block).sum(axis=0)
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - np.square(mean), 1e-12))
    return mean.tolist(), std.tolist()


def normalization(rows, path=None):
    scale = 'action_scale' if rows.environment in LIBERO else 'action_std'
    if path is None:
        mean, std = compute_moments(rows.arrays['actions'])
        stats = {'action_mean': mean, scale: std}
        if 'lowdim' in rows.arrays:
            stats['lowdim_mean'], stats['lowdim_std'] = compute_moments(rows.arrays['lowdim'])
    else:
        stats = json.loads(Path(path).read_text())
    sizes = {'action_mean': ACTION_DIM[rows.environment], scale: ACTION_DIM[rows.environment]}
    if LOWDIM_DIM[rows.environment]:
        sizes.update(lowdim_mean=LOWDIM_DIM[rows.environment], lowdim_std=LOWDIM_DIM[rows.environment])
    for key, width in sizes.items():
        values = np.asarray(stats[key], dtype=np.float32)
        if (values.shape != (width,) or not np.isfinite(values).all()
                or (key.endswith(('std', 'scale')) and np.any(values <= 0))):
            raise ValueError(f'Incorrect prior normalization for {key}.')
    return {key: stats[key] for key in sizes}
