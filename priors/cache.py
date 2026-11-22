"""Cache fixed local subgoals from the matching frozen FCP pair."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fcp import load_checkpoint
from priors.config import DRAWS
from priors.data import PriorRows
from priors.weights import upper_sha256
from release import CONFIGS, load_selected, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=CONFIGS, required=True)
    parser.add_argument('--variant', choices=('true', 'generated'), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1209)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--weights-root', type=Path)
    source.add_argument('--checkpoint', type=Path, help='An FGSS checkpoint from train.py.')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Use a positive batch size.')
    device = torch.device(args.device)
    if args.checkpoint:
        model, checkpoint = load_checkpoint(args.checkpoint, device, 'fgss')
        spec = CONFIGS[args.environment]['architecture']
        if (checkpoint['environment'], checkpoint['condition'], checkpoint['config']) != (
                args.environment, args.variant, spec):
            raise ValueError('FGSS checkpoint does not match the selected environment and variant.')
    else:
        model = load_selected(args.environment, args.variant, device, args.weights_root)
    model.eval().requires_grad_(False)
    rows = PriorRows(args.data, args.environment)
    binding = dict(environment=args.environment, variant=args.variant, split=args.split,
                   draws=DRAWS, seed=args.seed, batch_size=args.batch_size,
                   upper_sha256=upper_sha256(model), data_sha256=rows.signature())
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'local.npy'
    values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                      shape=rows.arrays['goal'].shape)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    for start in range(0, rows.size, args.batch_size):
        batch = slice(start, start + args.batch_size)
        history, goal = rows.inputs_for_upper(batch, device)
        predicted = model.sample(history, goal, samples=DRAWS, generator=rng)
        if not torch.isfinite(predicted).all():
            raise FloatingPointError('Nonfinite cached subgoals.')
        values[batch] = predicted.cpu().numpy()
    values.flush()
    del values
    binding['cache_sha256'] = sha256_file(path)
    (args.output / 'manifest.json').write_text(json.dumps(binding, indent=2) + '\n')


if __name__ == '__main__':
    main()
