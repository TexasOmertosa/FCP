"""Load the exact action prior bound to a planner or a locally trained pair."""

import json
from pathlib import Path

from safetensors.torch import load_file

from priors.config import architecture
from priors.model import ActionPrior
from fcp.release import CONFIGS, prior_assets, sha256_file


def load_prior(environment, variant, device, *, weights_root=None, prior_root=None, upper=None):
    if prior_root is None:
        checkpoint, metadata = prior_assets(environment, variant, weights_root)
        expected = CONFIGS[environment]['variants'][variant]
        if any(metadata[key] != expected[key] for key in ('fcg_epoch', 'fgss_epoch')):
            raise ValueError('The action prior does not match the selected planner.')
    else:
        directory = Path(prior_root)
        checkpoint = directory / 'best.safetensors'
        metadata = json.loads((directory / 'best.json').read_text())
        if (metadata['environment'], metadata['variant']) != (environment, variant):
            raise ValueError('The action prior belongs to a different planner.')
        if sha256_file(checkpoint) != metadata['weight_sha256']:
            raise ValueError('Locally trained action prior has the wrong checksum.')
        if upper is not None and metadata['upper_sha256'] != upper_sha256(upper):
            raise ValueError('Locally trained action prior was fitted to different upper weights.')
    model = ActionPrior(**architecture(environment))
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    return model.to(device).eval().requires_grad_(False), metadata


def upper_sha256(model):
    from hashlib import sha256

    digest = sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
