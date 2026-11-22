"""Resolve only the six environment pairs in the published FCP checkpoint set."""

import json
from hashlib import file_digest
from pathlib import Path

from fcp import load_fcp

REGISTRY = json.loads(Path(__file__).with_name('selected_models.json').read_text())
CONFIGS = REGISTRY['environments']


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def load_selected(environment, variant, device, weights_root=None):
    config = CONFIGS[environment]
    if weights_root is None:
        from huggingface_hub import snapshot_download

        files = config['variants'][variant]
        names = ('config.json', files['fcg'], files['fgss'])
        weights_root = snapshot_download(
            REGISTRY['model_repo'], revision=REGISTRY['model_revision'],
            allow_patterns=[f'{environment}/{name}' for name in names],
        )
    folder = Path(weights_root) / environment
    metadata = json.loads((folder / 'config.json').read_text())
    for field in ('architecture', 'variants', 'attention_padding_mask'):
        if metadata[field] != config[field]:
            raise ValueError(f'Weights for {environment} differ from the selected release ({field}).')
    return load_fcp(folder, variant, device)
