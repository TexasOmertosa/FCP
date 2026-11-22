"""Latent window preparation and dataset loading."""

import numpy as np
import torch
from torch.utils.data import TensorDataset


def window_indices(n_frames, boundaries, pairs):
    if (boundaries.ndim != 1 or len(boundaries) < 2 or boundaries.dtype.kind not in 'iu'
            or boundaries[0] != 0 or boundaries[-1] != n_frames
            or np.any(boundaries[1:] <= boundaries[:-1])):
        raise ValueError('Provide increasing episode offsets, including zero and the final frame count.')
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
            or pairs.dtype.kind not in 'iu' or np.any(pairs < 0)
            or np.any(pairs >= n_frames)):
        raise ValueError('Provide valid integer start and goal row indices [windows, 2].')
    current, goal = pairs.astype(np.int64).T
    episode = np.searchsorted(boundaries[1:], current, side='right')
    if np.any(goal < current) or np.any(goal >= boundaries[episode + 1]):
        raise ValueError('Each goal must be at or after the start in the same episode.')
    return current, goal, episode


def make_windows(latents, boundaries, pairs, architecture, target_indices=None):
    if latents.ndim != 2 or latents.shape[1] != architecture['latent_dim']:
        raise ValueError('Expected per-frame latents [frames, latent_dim] for this environment.')
    current, goal, episode = window_indices(len(latents), boundaries, pairs)
    offsets = np.array((0,) if architecture['history_frames'] == 1 else (-10, -5, 0))
    history = np.maximum(current[:, None] + offsets, boundaries[episode, None])
    if target_indices is None:
        subgoal = np.minimum(current + 10, goal)
        future = subgoal[:, None] + ((goal - subgoal)[:, None] * np.arange(1, 4) + 2) // 4
    else:
        nodes = architecture['future_nodes'] + 1
        if (target_indices.shape != (len(current), nodes) or target_indices.dtype.kind not in 'iu'
                or np.any(target_indices[:, 0] < current) or np.any(target_indices[:, -1] > goal)
                or np.any(target_indices[:, 1:] < target_indices[:, :-1])):
            raise ValueError('Target indices must be ordered frame positions inside each start-goal window.')
        subgoal, future = target_indices[:, 0], target_indices[:, 1:]
    return {key: latents[index].astype(np.float32, copy=False)
            for key, index in (('history', history), ('goal', goal),
                               ('subgoal', subgoal), ('future', future))}

def read_windows(path):
    """Load frozen latent windows and infer their model dimensions."""
    names = ('history', 'goal', 'subgoal', 'future')
    with np.load(path, allow_pickle=False) as arrays:
        history, goal, subgoal, future = (
            torch.from_numpy(arrays[name].astype(np.float32, copy=False)) for name in names
        )
    if history.ndim == 2:
        history = history[:, None]
    if goal.ndim != 2 or min(goal.shape) < 1 or subgoal.shape != goal.shape:
        raise ValueError('goal and subgoal must have the same nonempty shape [N, D].')
    for name, tensor in (('history', history), ('future', future)):
        if (tensor.ndim != 3 or tensor.shape[0] != goal.shape[0]
                or tensor.shape[1] < 1 or tensor.shape[2] != goal.shape[1]):
            raise ValueError(f'Invalid {name} shape.')
    shape = dict(latent_dim=goal.shape[1], history_frames=history.shape[1],
                 future_nodes=future.shape[1])
    return TensorDataset(history, goal, subgoal, future), shape
