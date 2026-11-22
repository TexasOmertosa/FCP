"""Cache fixed local subgoals from the matching frozen FCP pair."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fcp.model import load_checkpoint
from fcp.numerics import UPPER_PRIOR_CACHE
from priors.config import DRAWS, LIBERO
from priors.data import PriorRows
from priors.weights import upper_sha256
from fcp.release import CONFIGS, load_selected, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=CONFIGS, required=True)
    parser.add_argument('--variant', choices=('true', 'generated'), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int, help='Base seed for the frozen upper cache.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--weights-root', type=Path)
    source.add_argument('--checkpoint', type=Path, help='An FGSS checkpoint from fcp.train.')
    args = parser.parse_args()
    is_libero = args.environment in LIBERO
    block_seeded = args.environment in ('pusht', 'cube') or is_libero
    if args.batch_size is None:
        args.batch_size = 4096 if args.environment in LIBERO else (2048 if block_seeded else 256)
    if args.seed is None:
        if is_libero:
            args.seed = 9026102900 + (100000000 if args.split == 'val' else 0)
        elif block_seeded:
            args.seed = 149000000 + (1000000 if args.split == 'val' else 0)
        else:
            args.seed = 1209
    if args.batch_size < 1:
        parser.error('Use a positive batch size.')
    device = torch.device(args.device)
    numerics = UPPER_PRIOR_CACHE[args.environment]
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
    goal_offset = rows.arrays.get('goal_offset')
    binding = dict(environment=args.environment, variant=args.variant, split=args.split,
                   draws=DRAWS, seed=args.seed, batch_size=args.batch_size,
                   upper_sha256=upper_sha256(model), data_sha256=rows.signature(),
                   numerics=numerics.metadata(device),
                   draw_seed_stride=1000003 if block_seeded else None,
                   direct_terminal_rows=is_libero and goal_offset is not None)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'local.npy'
    values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                      shape=rows.arrays['goal'].shape)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    with numerics.forward(device, model):
        for start in range(0, rows.size, args.batch_size):
            batch = slice(start, start + args.batch_size)
            history, goal = rows.inputs_for_upper(batch, device)
            if block_seeded:
                base = args.seed + (start // args.batch_size if is_libero else start)
                active = (torch.as_tensor(goal_offset[batch] > 10, device=device)
                          if is_libero and goal_offset is not None else slice(None))
                predicted = goal.clone()
                current, target = history[active], goal[active]
                if len(target):
                    values_by_draw = []
                    for draw in range(DRAWS):
                        rng.manual_seed(base + 1000003 * draw)
                        noise = torch.randn((len(goal), model.future_nodes + 1, goal.shape[-1]),
                                            device=device, generator=rng)
                        values_by_draw.append(model.sample_from_noise(current, target, noise[active]))
                    # The PushT cache averaged a contiguous [window, draw, latent] bank.
                    predicted[active] = (torch.stack(values_by_draw, dim=1).mean(1)
                                         if args.environment == 'pusht' else torch.stack(values_by_draw).mean(0))
            else:
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
