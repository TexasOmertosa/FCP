"""Generate frozen-FCG conditions or FCP subgoals for encoded inputs."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import load_checkpoint
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
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1209)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--weights-root')
    source.add_argument('--checkpoint', help='An output checkpoint from fcp.train.')
    args = parser.parse_args()
    count = args.samples if args.samples is not None else (1 if args.output_kind == 'future' else 8)
    if count < 1 or args.batch_size < 1 or (args.output_kind == 'future' and count != 1):
        parser.error('Use a positive batch size and sample count; generation of future conditions uses one draw.')
    device = torch.device(args.device)
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
    for hist, target in DataLoader(TensorDataset(history, goal), batch_size=args.batch_size):
        hist, target = hist.to(device), target.to(device)
        if args.output_kind == 'future':
            prediction = model.generate_future(hist, target, generator=generator)
        else:
            prediction = model.sample(hist, target, samples=count, generator=generator)
        results.append(prediction.cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {args.output_kind: np.concatenate(results, axis=0)}
    if args.output_kind == 'future' and args.checkpoint:
        payload.update(input_sha256=sha256_file(args.data),
                       fcg_checkpoint_sha256=sha256_file(args.checkpoint))
    with args.output.open('wb') as file:
        np.savez_compressed(file, **payload)


if __name__ == '__main__':
    main()
