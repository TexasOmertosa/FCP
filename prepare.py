"""Extract FCP windows from a single split of frozen demonstration latents."""

import argparse
from pathlib import Path

import numpy as np

from release import CONFIGS


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


def make_windows(latents, boundaries, pairs, architecture):
    if latents.ndim != 2 or latents.shape[1] != architecture['latent_dim']:
        raise ValueError('Expected per-frame latents [frames, latent_dim] for this environment.')
    current, goal, episode = window_indices(len(latents), boundaries, pairs)
    offsets = np.array((0,) if architecture['history_frames'] == 1 else (-10, -5, 0))
    history = np.maximum(current[:, None] + offsets, boundaries[episode, None])
    subgoal = np.minimum(current + 10, goal)
    future = subgoal[:, None] + ((goal - subgoal)[:, None] * np.arange(1, 4) + 2) // 4
    return {key: latents[index].astype(np.float32, copy=False)
            for key, index in (('history', history), ('goal', goal),
                               ('subgoal', subgoal), ('future', future))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', required=True, choices=CONFIGS)
    parser.add_argument('--episodes', required=True, help='NPZ with latents and episode_offsets.')
    parser.add_argument('--pairs', required=True, help='Integer NPY array of start/goal frame pairs.')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.episodes, allow_pickle=False) as data:
        windows = make_windows(data['latents'], data['episode_offsets'],
                               np.load(args.pairs, allow_pickle=False),
                               CONFIGS[args.environment]['architecture'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        np.savez_compressed(stream, **windows)


if __name__ == '__main__':
    main()
