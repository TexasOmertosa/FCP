"""Extract the demonstration action chunks aligned to the FCP prior windows."""

import argparse
from pathlib import Path

import numpy as np

from prepare import window_indices
from priors.config import ACTION_BLOCK, ACTION_DIM, ACTION_TOKENS, HISTORY_FRAMES, LIBERO, LOWDIM_DIM
from release import CONFIGS


def prepare_rows(episodes, pairs, environment, output):
    spec = CONFIGS[environment]['architecture']
    dimensions = spec['latent_dim']
    history_frames = spec['history_frames']
    width = ACTION_BLOCK * ACTION_TOKENS
    latents, boundaries, actions = (episodes[name] for name in ('latents', 'episode_offsets', 'actions'))
    n_frames = len(latents)
    if latents.shape != (n_frames, dimensions) or actions.shape != (n_frames, ACTION_DIM[environment]):
        raise ValueError('Incorrect frame-aligned latent or action input shape.')
    current, goal, episode = window_indices(n_frames, boundaries, pairs)
    if np.any(current + width > goal):
        raise ValueError('Every goal must leave ten supervised actions from the start.')
    low_dims = LOWDIM_DIM[environment]
    lowdim = episodes['lowdim'] if low_dims else None
    if low_dims and lowdim.shape != (n_frames, low_dims):
        raise ValueError('Incorrect frame-aligned low-dimensional input shape.')
    upper_latents = episodes['upper_latents'] if 'upper_latents' in episodes else None
    if upper_latents is not None and upper_latents.shape != latents.shape:
        raise ValueError('Upper latents and prior latents must have the same frame layout.')

    n = len(current)
    shapes = dict(history=(n, HISTORY_FRAMES, dimensions), goal=(n, dimensions),
                  actions=(n, width, ACTION_DIM[environment]))
    if low_dims:
        shapes['lowdim'] = (n, low_dims)
    if environment not in LIBERO:
        shapes['goal_offset'] = (n,)
    if upper_latents is not None:
        shapes['upper_history'] = (n, history_frames, dimensions)
    output.mkdir(parents=True, exist_ok=False)
    arrays = {name: np.lib.format.open_memmap(output / f'{name}.npy', mode='w+',
                                              dtype=np.float32, shape=shape)
              for name, shape in shapes.items()}
    for index in range(0, n, 4096):
        batch = slice(index, index + 4096)
        starts, goals = current[batch], goal[batch]
        history = np.maximum(starts[:, None] + np.array([-10, -5, 0]),
                             boundaries[episode[batch], None])
        arrays['history'][batch] = latents[history]
        arrays['goal'][batch] = latents[goals]
        arrays['actions'][batch] = actions[starts[:, None] + np.arange(width)]
        if low_dims:
            arrays['lowdim'][batch] = lowdim[starts]
        if environment not in LIBERO:
            arrays['goal_offset'][batch] = goals - starts
        if upper_latents is not None:
            arrays['upper_history'][batch] = upper_latents[history[:, -history_frames:]]
    for array in arrays.values():
        array.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', required=True, choices=CONFIGS)
    parser.add_argument('--episodes', required=True, help='NPZ of one frozen demonstration split.')
    parser.add_argument('--pairs', required=True, help='Integer NPY array of start/goal frame pairs.')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.episodes, allow_pickle=False) as episodes:
        prepare_rows(episodes, np.load(args.pairs, allow_pickle=False), args.environment, args.output)


if __name__ == '__main__':
    main()
