"""Reconstruct the episode partitions from the original demonstrations."""

import json
from pathlib import Path

import numpy as np

from .data import window_indices


_SHARED_SOURCES = {
    'pusht': (18685, 14948, 1868),
    'cube': (10000, 8000, 1000),
}
_NAVIGATION_SOURCES = {
    ('antmaze', 'train'): ('visual_antmaze_medium_explore_stitch_train.h5', 25000, None, 42),
    ('antmaze', 'val'): ('visual_antmaze_medium_stitch_val.h5', 1250, 201000, 42),
    ('fourroom', 'train'): ('fourroom_tp35_d1.h5', 3750, 1000000, 0),
    ('fourroom', 'val'): ('fourroom_tp35_d1_val.h5', 150, 40000, 0),
}


def add_split_arguments(parser):
    parser.add_argument('--split', choices=('train', 'val', 'test'),
                        help='Select windows from the original episode partition.')
    parser.add_argument('--source-file', type=Path,
                        help='Original navigation HDF5 or extracted LIBERO DATASET_MANIFEST.json.')


def _episode_lengths(path):
    import h5py

    with h5py.File(path, 'r') as source:
        return source['ep_len'][:].astype(np.int64)


def _libero_lengths(environment, split, source_file):
    partition = 'validation' if split == 'val' else split
    if source_file is None or source_file.name != 'DATASET_MANIFEST.json':
        raise ValueError(f'Provide the {environment} {split} archive DATASET_MANIFEST.json with --source-file.')
    manifest = json.loads(source_file.read_text())
    if manifest['suite'] != f'libero_{environment}' or manifest['split'] != partition:
        raise ValueError(f'Expected the original {environment} {split} archive.')
    shards = sorted(name for name in manifest['files']
                    if name.startswith(f'episodes/{partition}/') and name.endswith('.hdf5'))
    if not shards:
        raise ValueError(f'No episode shards found in {source_file}.')
    lengths = np.concatenate([_episode_lengths(source_file.parent / name) for name in shards])
    if len(lengths) != (8000 if split == 'train' else 1000):
        raise ValueError(f'Unexpected number of episodes in {source_file}.')
    return lengths


def _budgeted_members(lengths, budget, seed):
    order = np.random.default_rng(seed).permutation(len(lengths))
    totals = np.cumsum(lengths[order])
    if totals[-1] < budget:
        raise ValueError('The source has fewer transitions than the partition budget.')
    last = int(np.searchsorted(totals, budget))
    if last > 0 and budget - totals[last - 1] < totals[last] - budget:
        last -= 1
    return np.sort(order[:last + 1])


def _partition(environment, split, source_file):
    if environment in _SHARED_SOURCES:
        count, train, val = _SHARED_SOURCES[environment]
        order = np.random.default_rng(42).permutation(count)
        partitions = np.split(order, (train, train + val))
        members = partitions[('train', 'val', 'test').index(split)]
        return np.sort(members) if environment == 'pusht' else members, count, None

    if split not in ('train', 'val'):
        raise ValueError(f'{environment} has no demonstration partition named {split!r}.')
    if environment in ('scene2', 'caddy'):
        lengths = _libero_lengths(environment, split, source_file)
        return np.arange(len(lengths)), len(lengths), lengths

    filename, count, budget, seed = _NAVIGATION_SOURCES[environment, split]
    if source_file is None or source_file.name != filename:
        raise ValueError(f'Provide the path to {filename} with --source-file for {environment} {split}.')
    lengths = _episode_lengths(source_file)
    if lengths.shape != (count,) or np.any(lengths <= 0):
        raise ValueError(f'Unexpected episode layout in {source_file}.')
    members = np.arange(count) if budget is None else _budgeted_members(lengths, budget, seed)
    return members, count, None


def select_window_rows(environment, split, episodes, n_frames, pairs, source_file=None):
    """Return a mask for the requested split, preserving the input row order."""
    members, source_count, source_lengths = _partition(environment, split, source_file)
    boundaries = episodes['episode_offsets']
    _, _, episode = window_indices(n_frames, boundaries, pairs)
    episode_count = len(boundaries) - 1
    if 'episode_ids' in episodes:
        ids = episodes['episode_ids']
        if (ids.shape != (episode_count,) or ids.dtype.kind not in 'iu'
                or np.any(ids < 0) or np.any(ids >= source_count)
                or len(np.unique(ids)) != episode_count):
            raise ValueError('Provide one distinct, valid source episode ID per encoded episode.')
    else:
        if episode_count != source_count:
            raise ValueError('Encode the entire source in episode order or supply source episode_ids.')
        ids = np.arange(episode_count)

    if source_lengths is not None and not np.array_equal(source_lengths[ids], np.diff(boundaries)):
        raise ValueError(f'Encoded episode frame counts differ from the original {environment} {split} archive.')
    selected = np.isin(ids[episode], members)
    if not np.any(selected):
        raise ValueError(f'No candidate windows belong to the {environment} {split} partition.')
    return selected
