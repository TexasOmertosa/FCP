"""Train FCG, then fit FGSS using demonstrated or generated future context."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import read_windows
from .metrics import evaluate
from .model import FCP, load_checkpoint
from .release import CONFIGS, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=CONFIGS, required=True)
    parser.add_argument('--stage', choices=('fcg', 'fgss'), required=True)
    parser.add_argument('--condition', choices=('true', 'generated'))
    parser.add_argument('--fcg-checkpoint')
    parser.add_argument('--future-cache', help='Optional output of fcp.generate --output-kind future.')
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    selected = CONFIGS[args.environment]
    protocol = selected['protocol']
    if args.epochs is None:
        args.epochs = protocol.get('generated_epochs', 200) if args.condition == 'generated' else 200
    if args.seed is None:
        args.seed = protocol['training_seed']
    if args.stage == 'fgss' and not (args.condition and args.fcg_checkpoint):
        parser.error('FGSS training needs --condition and --fcg-checkpoint.')
    if args.stage == 'fcg' and (args.condition or args.fcg_checkpoint):
        parser.error('FGC training starts from scratch.')
    if args.future_cache and (args.stage, args.condition) != ('fgss', 'generated'):
        parser.error('Only Generated FGSS training reads a future cache.')
    if min(args.epochs, args.batch_size) < 1 or args.warmup_epochs < 0:
        parser.error('Use positive epochs and batch size and nonnegative warm-up epochs.')

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    train, shape = read_windows(args.train)
    val, val_shape = read_windows(args.val)
    if val_shape != shape:
        raise ValueError('Train and validation window dimensions differ.')
    config = selected['architecture']
    if any(config[key] != value for key, value in shape.items()):
        raise ValueError('Window dimensions do not match the selected environment.')
    if args.stage == 'fcg':
        model = FCP(**config).to(device)
        fitted = model.A
    else:
        if Path(args.fcg_checkpoint).resolve() == args.output.resolve():
            parser.error('Use a separate output path for the FGSS checkpoint.')
        model, checkpoint = load_checkpoint(args.fcg_checkpoint, device, 'fcg')
        if checkpoint['config'] != config or checkpoint['environment'] != args.environment:
            raise ValueError('The FCG checkpoint belongs to a different environment or architecture.')
        model.A.eval().requires_grad_(False)
        fitted = model.B

    if args.future_cache:
        with np.load(args.future_cache, allow_pickle=False) as cache:
            if (cache['input_sha256'].item() != sha256_file(args.train)
                    or cache['fcg_checkpoint_sha256'].item() != sha256_file(args.fcg_checkpoint)):
                raise ValueError('Future cache was generated from different windows or FCG weights.')
            cached = torch.from_numpy(cache['future'].astype(np.float32, copy=False))
        if cached.shape != train.tensors[-1].shape:
            raise ValueError('Cached future nodes do not match the training windows.')
        train = TensorDataset(*train.tensors[:-1], cached)

    optimizer = torch.optim.AdamW(fitted.parameters(), lr=1e-3, weight_decay=1e-4)
    train_batches = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_batches = DataLoader(val, batch_size=args.batch_size)
    warmup = min(args.warmup_epochs, args.epochs) * len(train_batches)
    total = args.epochs * len(train_batches)
    step, best = 0, float('inf')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        fitted.train()
        for history, goal, subgoal, future in train_batches:
            history, goal, subgoal, future = (array.to(device) for array in
                                              (history, goal, subgoal, future))
            if step < warmup:
                lr = 1e-3 * (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, total - warmup - 1)
                lr = 1e-6 + (1e-3 - 1e-6) * (1 + math.cos(math.pi * progress)) / 2
            optimizer.param_groups[0]['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            if args.stage == 'fcg':
                loss = model.fcg_loss(history, goal, future)
            else:
                if args.condition == 'true' or args.future_cache:
                    condition = future
                else:
                    condition = model.generate_future(history, goal)
                loss = model.fgss_loss(history, goal, subgoal, condition)
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite training loss at step {step}.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fitted.parameters(), 1.0)
            optimizer.step()
            step += 1

        mse = evaluate(model, val_batches, device, (1,), seed=1209,
                       future_only=args.stage == 'fcg')[1]
        if not math.isfinite(mse):
            raise ValueError(f'Nonfinite validation MSE at epoch {epoch}.')
        if mse < best:
            best = mse
            temporary = args.output.with_name(args.output.name + '.tmp')
            torch.save(dict(config=config, model=model.state_dict(), stage=args.stage,
                            condition=args.condition, environment=args.environment, epoch=epoch, val_mse=mse), temporary)
            temporary.replace(args.output)
        print(f'epoch={epoch} val_mse={mse:.6f} best={best:.6f}', flush=True)


if __name__ == '__main__':
    main()
