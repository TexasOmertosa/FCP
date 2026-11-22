"""Generate frozen-FCG conditions or FCP subgoals for encoded inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import load_checkpoint
from .noise import cube_window_noise
from .numerics import UPPER_DEPLOYMENT, UPPER_TRAIN
from .release import CONFIGS, load_selected, sha256_file


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=CONFIGS, required=True)
    parser.add_argument('--variant', choices=('true', 'generated'), default='true')
    parser.add_argument('--output-kind', choices=('future', 'subgoal'), default='subgoal')
    parser.add_argument('--data', required=True, help='NPZ with history and goal arrays.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--split', choices=('train', 'val'), default='train',
                        help='Seed schedule for a frozen FCG condition cache.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--weights-root')
    source.add_argument('--checkpoint', help='An output checkpoint from fcp.train.')
    args = parser.parse_args()
    libero_future = args.environment in ('scene2', 'caddy') and args.output_kind == 'future'
    cube_future = args.environment == 'cube' and args.output_kind == 'future'
    full_val_noise = (args.environment in ('pusht', 'fourroom') and args.output_kind == 'future'
                      and args.split == 'val')
    if args.batch_size is None:
        args.batch_size = 2048 if full_val_noise else 1024 if libero_future else 512 if cube_future else 256
    if args.seed is None:
        args.seed = 1209
        if libero_future:
            args.seed = 1020261034 if args.split == 'train' else 20261027
        elif cube_future:
            args.seed = 148 if args.split == 'train' else 77148
        elif full_val_noise:
            args.seed = 77148
    count = args.samples if args.samples is not None else (1 if args.output_kind == 'future' else 8)
    if count < 1 or args.batch_size < 1 or (args.output_kind == 'future' and count != 1):
        parser.error('Use a positive batch size and sample count; generation of future conditions uses one draw.')
    device = torch.device(args.device)
    profiles = UPPER_TRAIN if args.output_kind == 'future' else UPPER_DEPLOYMENT
    numerics = profiles[args.environment]
    architecture = CONFIGS[args.environment]['architecture']
    if args.checkpoint:
        stage = 'fcg' if args.output_kind == 'future' else 'fgss'
        model, checkpoint = load_checkpoint(args.checkpoint, device, stage)
        if (checkpoint['config'] != architecture or checkpoint['environment'] != args.environment
                or (stage == 'fgss' and checkpoint['condition'] != args.variant)):
            raise ValueError('Checkpoint architecture or condition differs from the selected setup.')
        model.eval()
    else:
        model = load_selected(args.environment, args.variant, device, args.weights_root)

    with np.load(args.data, allow_pickle=False) as data:
        history, goal = (torch.from_numpy(data[key].astype(np.float32, copy=False))
                         for key in ('history', 'goal'))
    if history.ndim == 2:
        history = history[:, None]
    dim, frames = architecture['latent_dim'], architecture['history_frames']
    if goal.ndim != 2 or goal.shape[0] == 0 or goal.shape[1] != dim or history.shape != (len(goal), frames, dim):
        raise ValueError('Input sizes differ from the selected environment dimensions.')

    results = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    val_noise = None
    if full_val_noise:
        val_noise = torch.randn((len(goal), model.future_nodes + 1, dim), device=device, generator=generator)
    loader = DataLoader(TensorDataset(history, goal), batch_size=args.batch_size)
    for index, (hist, target) in enumerate(loader):
        hist, target = hist.to(device), target.to(device)
        with numerics.forward(device, model):
            if val_noise is not None:
                start = index * args.batch_size
                prediction = model.generate_future_from_noise(hist, target, val_noise[start:start+len(target), 1:])
            elif cube_future:
                nodes = model.future_nodes + int(args.split == 'val')
                noise = cube_window_noise(index * args.batch_size, len(target), args.seed,
                                          (nodes, target.shape[-1]), device)
                prediction = model.generate_future_from_noise(hist, target, noise[:, -model.future_nodes:])
            elif libero_future:
                generator.manual_seed(args.seed + index)
                noise = torch.randn((len(target), model.future_nodes + 1, target.shape[-1]),
                                    device=device, generator=generator)
                prediction = model.generate_future_from_noise(hist, target, noise[:, 1:])
            elif args.output_kind == 'future':
                if args.split == 'val':
                    noise = torch.randn((len(target), model.future_nodes + 1, target.shape[-1]),
                                        device=device, generator=generator)
                    prediction = model.generate_future_from_noise(hist, target, noise[:, 1:])
                else:
                    prediction = model.generate_future(hist, target, generator=generator)
            else:
                prediction = model.sample(hist, target, samples=count, generator=generator)
        results.append(prediction.cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {args.output_kind: np.concatenate(results, axis=0),
               'numerics': json.dumps(numerics.metadata(device), sort_keys=True),
               'seed': args.seed, 'split': args.split}
    if args.output_kind == 'future' and args.checkpoint:
        payload.update(input_sha256=sha256_file(args.data),
                       fcg_checkpoint_sha256=sha256_file(args.checkpoint))
    with args.output.open('wb') as file:
        np.savez_compressed(file, **payload)


if __name__ == '__main__':
    main()
