"""Extract FCP windows from a single split of frozen demonstration latents."""

import argparse
from pathlib import Path

import numpy as np

from .data import make_windows
from .release import CONFIGS
from .splits import add_split_arguments, select_window_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', required=True, choices=CONFIGS)
    parser.add_argument('--episodes', required=True, help='NPZ with latents and episode_offsets.')
    parser.add_argument('--pairs', required=True, help='Integer NPY array of start/goal frame pairs.')
    parser.add_argument('--target-indices', help='Optional NPY with the four archived supervision frames per window.')
    parser.add_argument('--output', required=True, type=Path)
    add_split_arguments(parser)
    args = parser.parse_args()
    if args.source_file and args.split is None:
        parser.error('--source-file requires --split.')
    with np.load(args.episodes, allow_pickle=False) as data:
        latents = data['latents']
        pairs = np.load(args.pairs, allow_pickle=False)
        targets = np.load(args.target_indices, allow_pickle=False) if args.target_indices else None
        if args.split is not None:
            selected = select_window_rows(args.environment, args.split, data, len(latents), pairs, args.source_file)
            if targets is not None and (targets.ndim != 2 or len(targets) != len(pairs)):
                parser.error('--target-indices must contain one row per supplied pair.')
            pairs = pairs[selected]
            targets = targets[selected] if targets is not None else None
        windows = make_windows(latents, data['episode_offsets'], pairs,
                               CONFIGS[args.environment]['architecture'], targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        np.savez_compressed(stream, **windows)


if __name__ == '__main__':
    main()
