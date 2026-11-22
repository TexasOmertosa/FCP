"""Extract FCP windows from a single split of frozen demonstration latents."""

import argparse
from pathlib import Path

import numpy as np

from .data import make_windows
from .release import CONFIGS


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
