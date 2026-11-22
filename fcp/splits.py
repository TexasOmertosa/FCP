"""Apply the published episode partitions to candidate windows."""

import csv
from pathlib import Path

import numpy as np

from .data import window_indices


SPLIT_DIR = Path(__file__).resolve().parents[1] / 'data' / 'splits'
MANIFESTS = {
    'pusht': 'pusht_episode_split.csv',
    'cube': 'cube_episode_split.csv',
    'scene2': 'scene2_train_val_episode_index.csv',
    'caddy': 'caddy_train_val_episode_index.csv',
    'antmaze': 'antmaze_episode_split.csv',
    'fourroom': 'fourroom_episode_split.csv',
}


def add_split_arguments(parser):
    parser.add_argument('--split', choices=('train', 'val', 'test'),
                        help='Select windows using the published episode partition. '
                             'Omit for an NPZ and pairs already prepared for a single split.')
    parser.add_argument('--source-file', type=Path,
                        help='Original HDF5 filename when applying an AntMaze or FourRoom split.')


def select_window_rows(environment, split, episodes, n_frames, pairs, source_file=None):
    """Return the input rows in the requested fixed split, in their original order."""
    with (SPLIT_DIR / MANIFESTS[environment]).open(newline='', encoding='utf-8') as stream:
        manifest = list(csv.DictReader(stream))

    partition = 'validation' if split == 'val' and 'episode_id_in_partition' in manifest[0] else split
    rows = [row for row in manifest if row['partition'] == partition]
    if not rows:
        raise ValueError(f'{environment} has no published demonstration partition named {split!r}.')

    if 'source_file' in rows[0]:
        expected_source = rows[0]['source_file']
        if source_file is None or source_file.name != expected_source:
            raise ValueError(f'{environment} {split} requires --source-file {expected_source}.')

    boundaries = episodes['episode_offsets']
    _, _, episode = window_indices(n_frames, boundaries, pairs)
    episode_count = len(boundaries) - 1
    if 'episode_ids' in episodes:
        ids = episodes['episode_ids']
        if (ids.shape != (episode_count,) or ids.dtype.kind not in 'iu'
                or np.any(ids < 0) or len(np.unique(ids)) != episode_count):
            raise ValueError('Provide one distinct nonnegative source episode ID per encoded episode.')
    else:
        if 'position' in rows[0] and len(manifest) != episode_count:
            raise ValueError('Encode the entire source in episode order or supply source episode_ids.')
        ids = np.arange(episode_count)

    if 'episode_id_in_partition' in rows[0]:
        lengths = {int(row['episode_id_in_partition']): int(row['frame_count']) for row in rows}
        for episode_id, length in zip(ids, np.diff(boundaries)):
            if lengths.get(int(episode_id)) != length:
                raise ValueError(f'Encoded episode {episode_id} does not match the published '
                                 f'{environment} {split} archive frame count.')
        members = list(lengths)
    else:
        members = [int(row['episode_id']) for row in rows]

    selected = np.isin(ids[episode], members)
    if not np.any(selected):
        raise ValueError(f'No candidate windows belong to the published {environment} {split} partition.')
    return selected
