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


def load_frozen_world_model(device, root, *, repo, revision, checksum):
    from huggingface_hub import snapshot_download
    from hydra.utils import instantiate
    import torch

    folder = Path(root or snapshot_download(repo, revision=revision,
                                            allow_patterns=['weights.pt', 'config.json']))
    checkpoint = folder / 'weights.pt'
    if sha256_file(checkpoint) != checksum:
        raise ValueError(f'Expected the exact frozen world-model checkpoint for {repo}.')
    model = instantiate(json.loads((folder / 'config.json').read_text()))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False)


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
    expected = CONFIGS[environment]['action_priors'][variant]
    if (metadata['file'], metadata['epoch'], metadata['weight_sha256']) != (
            expected['file'], expected['epoch'], expected['weight_sha256']):
        raise ValueError(f'The {environment} {variant} action prior metadata differs from the selected release.')
    if sha256_file(checkpoint) != expected['weight_sha256']:
        raise ValueError(f'The {environment} {variant} action prior has the wrong checksum.')
    return checkpoint, metadata


def load_selected(environment, variant, device, weights_root=None):
    config = CONFIGS[environment]
    files = config['variants'][variant]
    if weights_root is None:
        from huggingface_hub import snapshot_download

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
    for component in ('fcg', 'fgss'):
        filename = files[component]
        if sha256_file(folder / filename) != config['component_sha256'][filename]:
            raise ValueError(f'The {environment} {filename} checkpoint has the wrong checksum.')
    return load_fcp(folder, variant, device)


def load_closed_loop_weights(args, environment, device):
    from .model import load_checkpoint
    from priors.weights import load_prior

    if args.checkpoint:
        if args.prior_root is None:
            raise ValueError('Provide the matching --prior-root for the local FCP checkpoint.')
        upper, checkpoint = load_checkpoint(args.checkpoint, device, 'fgss')
        if (checkpoint['environment'], checkpoint['condition'], checkpoint['config']) != (
                environment, args.variant, CONFIGS[environment]['architecture']):
            raise ValueError('The FCG/FGSS checkpoint belongs to a different configuration.')
    else:
        upper = load_selected(environment, args.variant, device, args.weights_root)
    upper.eval().requires_grad_(False)
    prior, metadata = load_prior(environment, args.variant, device, weights_root=args.weights_root,
                                 prior_root=args.prior_root, upper=upper)
    return upper, prior, metadata
