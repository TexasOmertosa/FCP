"""Resolve only the six environment pairs in the published FCP checkpoint set."""

import json
from hashlib import sha256
from pathlib import Path

from .model import load_fcp

REGISTRY = json.loads(Path(__file__).with_name('selected_models.json').read_text())
CONFIGS = REGISTRY['environments']


def sha256_file(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_file(filename, *, dataset=False):
    """Fetch one artifact at the pinned model or task revision."""
    from huggingface_hub import hf_hub_download

    kind = 'task' if dataset else 'model'
    return Path(hf_hub_download(
        REGISTRY[f'{kind}_repo'],
        filename,
        revision=REGISTRY[f'{kind}_revision'],
        repo_type='dataset' if dataset else 'model',
    ))


def prior_assets(environment, variant, weights_root=None):
    metadata_name = f'{environment}/action_priors.json'
    if weights_root is None:
        meta_file = resolve_file(metadata_name)
    else:
        meta_file = Path(weights_root) / metadata_name
    metadata = json.loads(meta_file.read_text())[variant]
    filename = f"{environment}/{metadata['file']}"
    if weights_root is None:
        checkpoint = resolve_file(filename)
    else:
        checkpoint = Path(weights_root) / filename
    if sha256_file(checkpoint) != metadata['weight_sha256']:
        raise ValueError(f'The {environment} {variant} action prior has the wrong checksum.')
    return checkpoint, metadata


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
