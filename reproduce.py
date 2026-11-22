"""Reproduce the released one-window reference predictions for each FCP pair."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluation import evaluate
from fcp import read_windows
from release import CONFIGS, load_selected, sha256_file


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=CONFIGS, action='append')
    parser.add_argument('--weights-root', type=Path)
    parser.add_argument('--verify-priors', action='store_true',
                        help='Also verify the matching prior checkpoint checksums.')
    args = parser.parse_args()
    environments = args.environment or list(CONFIGS)
    reference = json.loads((ROOT / 'examples' / 'reference.json').read_text())
    device = torch.device('cpu')
    torch.set_num_threads(1)
    rows = []
    for environment in environments:
        fixture = ROOT / 'examples' / f'{environment}_window.npz'
        spec = reference[environment]
        if sha256_file(fixture) != spec['fixture_sha256']:
            raise ValueError(f'Incorrect latent window for {environment}.')
        windows, _ = read_windows(fixture)
        loader = DataLoader(windows, batch_size=1)
        for variant in ('true', 'generated'):
            model = load_selected(environment, variant, device, args.weights_root)
            mse = evaluate(model, loader, device, (1, 8), seed=1209)
            for count, value in mse.items():
                expected = spec[variant][str(count)]
                if not math.isclose(expected, value, rel_tol=1e-4, abs_tol=1e-7):
                    raise ValueError(f'{environment} {variant}, {count} draws: {value} != {expected}')
            row = dict(environment=environment, variant=variant, mse=mse)
            if args.verify_priors:
                from priors.config import ACTION_DIM, LOWDIM_DIM
                from priors.weights import load_prior

                prior, metadata = load_prior(environment, variant, device,
                                             weights_root=args.weights_root)
                history, goal, _, _ = windows.tensors
                history = history[:, -1:].expand(-1, 3, -1)
                low = goal.new_zeros(len(goal), LOWDIM_DIM[environment])
                actions = prior.sample(history, goal[:, None], low, num_samples=4,
                                       generator=torch.Generator().manual_seed(149))
                if actions.shape != (len(goal), 4, 2, 5 * ACTION_DIM[environment]) or not torch.isfinite(actions).all():
                    raise ValueError(f'The {environment} {variant} prior has invalid outputs.')
                row['prior_sha256'] = metadata['weight_sha256']
            rows.append(row)
            del model
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
