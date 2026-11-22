"""Evaluate FCP on held-out latents or run a closed-loop environment backend."""

import argparse
from importlib import import_module
import json

import torch
from torch.utils.data import DataLoader

from fcp import load_checkpoint, read_windows
from planner import run_episode
from release import CONFIGS, load_selected


@torch.no_grad()
def evaluate(model, loader, device, counts, seed, future_only=False):
    model.eval()
    noise = torch.Generator(device=device).manual_seed(seed)
    totals = {count: 0.0 for count in counts}
    coordinates = 0
    for history, goal, subgoal, future in loader:
        history, goal = history.to(device), goal.to(device)
        target = (future if future_only else subgoal).to(device)
        predict = model.generate_future if future_only else model.sample
        draws = torch.stack([predict(history, goal, generator=noise)
                             for _ in range(max(counts))])
        for count in counts:
            prediction = draws[:count].mean(dim=0)
            totals[count] += (prediction.double() - target).square().sum().item()
        coordinates += target.numel()
    return {count: total / coordinates for count, total in totals.items()}


def closed_loop(model, args, device, protocol):
    module, separator, name = args.backend.partition(':')
    if not separator:
        raise ValueError('Use --backend module:factory.')
    backend = getattr(import_module(module), name)(args.environment, args.variant,
                                                   args.backend_config, device)
    tasks = list(backend.tasks())
    if not tasks:
        raise ValueError('The backend returned no evaluation tasks.')
    counts = {}
    rows = []
    for draws in args.samples:
        for index, task in enumerate(tasks):
            success, steps = run_episode(model, backend, task, protocol, device,
                                         args.seed + index, draws)
            rows.append(dict(task_index=index, draws=draws, success=success, steps=steps))
        counts[draws] = sum(row['success'] for row in rows if row['draws'] == draws)
    return dict(tasks_per_setting=len(tasks), successes=counts,
                success_percent={draws: 100 * n / len(tasks) for draws, n in counts.items()},
                trials=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=CONFIGS, required=True)
    parser.add_argument('--variant', choices=('true','generated'), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--data', help='Held-out latent windows in NPZ format.')
    mode.add_argument('--backend', help='Native environment integration as module:factory.')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--weights-root')
    source.add_argument('--checkpoint', help='An FGSS checkpoint produced by train.py.')
    parser.add_argument('--backend-config')
    parser.add_argument('--samples', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=1209)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    spec = CONFIGS[args.environment]
    args.samples = args.samples if args.samples is not None else ([1, 8] if args.data
                                                                  else [spec['main_table_complete_draws']])
    if min(args.samples) < 1 or len(set(args.samples)) != len(args.samples) or args.batch_size < 1:
        parser.error('Use distinct positive sample counts and a positive batch size.')
    if args.backend and args.samples != [spec['main_table_complete_draws']]:
        parser.error('The released closed-loop configuration uses eight complete samples.')
    device = torch.device(args.device)
    if args.checkpoint:
        model, cp = load_checkpoint(args.checkpoint, device, 'fgss')
        if (cp['config'] != spec['architecture'] or cp['environment'] != args.environment
                or cp['condition'] != args.variant):
            raise ValueError('The checkpoint does not match the selected environment and variant.')
        model.eval()
    else:
        model = load_selected(args.environment, args.variant, device, args.weights_root)

    if args.data:
        windows, shape = read_windows(args.data)
        if any(spec['architecture'][key] != value for key, value in shape.items()):
            raise ValueError('Evaluation window dimensions do not match the selected environment.')
        loader = DataLoader(windows, batch_size=args.batch_size)
        results = dict(windows=len(windows),
                       per_coordinate_mse=evaluate(model, loader, device, args.samples, args.seed))
    else:
        results = closed_loop(model, args, device, spec['protocol'])
    print(json.dumps(dict(environment=args.environment, variant=args.variant, **results),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
