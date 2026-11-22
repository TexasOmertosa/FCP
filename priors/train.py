"""Fit and export the action prior for one environment and FCP variant."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from priors.config import ACTION_BLOCK, ACTION_TOKENS, DRAWS, EPOCHS, LIBERO, architecture
from priors.data import PriorRows, load_condition, normalization
from priors.model import ActionPrior
from fcp.numerics import PRIOR_TRAIN, PRIOR_VAL
from fcp.release import CONFIGS, sha256_file


def forward_args(batch):
    history, near, far, lowdim, offsets, target = batch
    subgoal_steps = None if offsets is None else offsets.new_full(offsets.shape, ACTION_BLOCK * ACTION_TOKENS)
    return (history, near, lowdim), dict(action_horizon=ACTION_TOKENS,
                                        far_goal_latents=far, goal_offset_steps=offsets,
                                        subgoal_offset_steps=subgoal_steps), target


def fit_epoch(network, model, optimizer, rows, local, stats, settings, epoch, rank, world):
    env, batch_size, seed, device = settings
    if env in LIBERO:
        order = torch.randperm(rows.size, generator=torch.Generator().manual_seed(seed + epoch - 1)).numpy()
        order = order[:rows.size // batch_size * batch_size]
    else:
        order = np.random.default_rng(seed + epoch).permutation(rows.size)
    if world > 1:
        # Equal-sized shards keep each distributed optimizer update synchronized.
        order = order[:len(order) // batch_size * batch_size].reshape(-1, world, batch_size // world)
        batches = order[:, rank]
    else:
        batches = (order[k:k + batch_size] for k in range(0, len(order), batch_size))
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    network.train()
    numerics = PRIOR_TRAIN[env]
    with numerics.backend(device):
        for indices in batches:
            args, kwargs, target = forward_args(rows.batch(local, indices, stats, device))
            with numerics.autocast(device):
                prediction = network(*args, **kwargs)
                if env not in LIBERO:
                    nll, best_l1 = model.loss_terms(prediction, target)
                    loss = nll + 0.05 * best_l1
            if env in LIBERO:
                nll, best_l1 = model.loss_terms({key: value.float() for key, value in prediction.items()}, target)
                loss = nll + 0.05 * best_l1
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite action-prior loss.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            totals[0] += loss.detach().double() * len(indices)
            totals[1] += len(indices)
    if world > 1:
        dist.all_reduce(totals)
    if totals[1] == 0:
        raise ValueError('The training split is smaller than the required global batch.')
    return (totals[0] / totals[1]).item()


@torch.no_grad()
def validate(model, rows, local, stats, env, device):
    model.eval()
    use_best_samples = env in LIBERO
    rng = torch.Generator(device=device).manual_seed(1209)
    numerics = PRIOR_VAL[env]
    total = 0.
    size = 256 if use_best_samples else 128
    for start in range(0, rows.size, size):
        indices = np.arange(start, min(rows.size, start + size))
        args, kwargs, target = forward_args(rows.batch(local, indices, stats, device))
        with numerics.forward(device):
            if use_best_samples:
                samples = model.sample(*args, num_samples=128, generator=rng, **kwargs)
                metric = (samples - target[:, None]).abs().mean(dim=(-1, -2)).min(1).values.mean()
            else:
                metric, _ = model.loss_terms(model(*args, **kwargs), target)
        if not torch.isfinite(metric):
            raise FloatingPointError('Nonfinite prior validation metric.')
        total += metric.item() * len(indices)
    return total / rows.size


def save_best(directory, model, metadata):
    pending = directory / 'best.safetensors.tmp'
    weights = directory / 'best.safetensors'
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(pending))
    metadata['weight_sha256'] = sha256_file(pending)
    description = directory / 'best.json.tmp'
    description.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    pending.replace(weights)
    description.replace(directory / 'best.json')


def run(args, device, rank, world):
    env = args.environment
    train = PriorRows(args.train, env)
    val = PriorRows(args.val, env)
    train_local, train_meta = load_condition(args.train_cache, train, env, args.variant)
    val_local, val_meta = load_condition(args.val_cache, val, env, args.variant)
    if ((train_meta['split'], val_meta['split']) != ('train', 'val')
            or train_meta['upper_sha256'] != val_meta['upper_sha256']
            or train_meta['draws'] != DRAWS or val_meta['draws'] != DRAWS
            or train_meta['data_sha256'] == val_meta['data_sha256']):
        raise ValueError('Train and Val must be distinct splits of the same frozen FCP pair.')

    stats = normalization(train, args.normalization)
    config = architecture(env)
    tensors = {key: torch.tensor(value, device=device) for key, value in stats.items()}
    torch.manual_seed(args.seed)
    model = ActionPrior(**config).to(device)
    network = DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    selection = 'best128_action_l1' if env in LIBERO else 'nll'
    best = float('inf')
    settings = env, args.batch_size, args.seed, device
    for epoch in range(1, args.epochs + 1):
        loss = fit_epoch(network, model, optimizer, train, train_local, tensors, settings, epoch, rank, world)
        if rank == 0:
            metric = validate(model, val, val_local, tensors, env, device)
            if metric < best:
                best = metric
                save_best(args.output, model,
                          dict(environment=env, variant=args.variant, epoch=epoch, validation_metric=selection,
                               validation_value=best, model_config=config, normalization=stats,
                               upper_sha256=train_meta['upper_sha256'], train=train_meta, val=val_meta,
                               seed=args.seed, batch_size=args.batch_size, epochs=args.epochs,
                               train_numerics=PRIOR_TRAIN[env].metadata(device),
                               val_numerics=PRIOR_VAL[env].metadata(device)))
            print(json.dumps(dict(epoch=epoch, train_loss=loss, **{f'val_{selection}': metric})), flush=True)
        if world > 1:
            dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=CONFIGS, required=True)
    parser.add_argument('--variant', choices=('true', 'generated'), required=True)
    for name in ('train', 'val', 'train-cache', 'val-cache', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--normalization', type=Path, help='Optional fixed Train-only normalization JSON.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, help='Total batch across all ranks.')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    is_libero = args.environment in LIBERO
    args.batch_size = (1024 if is_libero else 128) if args.batch_size is None else args.batch_size
    args.seed = (218 if is_libero else 149) if args.seed is None else args.seed
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if min(args.epochs, args.batch_size) < 1 or args.batch_size % world:
        parser.error('Use positive epochs and a batch size divisible by the number of ranks.')
    if args.output.exists():
        parser.error('Select a new output directory.')
    if world > 1:
        if not args.device.startswith('cuda'):
            parser.error('Distributed training uses one GPU per rank.')
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
        dist.init_process_group('nccl')
        try:
            run(args, device, dist.get_rank(), world)
        finally:
            dist.destroy_process_group()
    else:
        run(args, torch.device(args.device), rank=0, world=1)


if __name__ == '__main__':
    main()
