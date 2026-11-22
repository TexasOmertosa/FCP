"""Train FCG, then fit FGSS using demonstrated or generated future context."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from .data import read_windows
from .model import FCP, load_checkpoint
from .noise import cube_window_noise
from .numerics import UPPER_TRAIN
from .release import CONFIGS, sha256_file


def with_future_cache(path, data_path, checkpoint, windows, numerics, split):
    with np.load(path, allow_pickle=False) as cache:
        if (cache['input_sha256'].item() != sha256_file(data_path)
                or cache['fcg_checkpoint_sha256'].item() != sha256_file(checkpoint)):
            raise ValueError('Future cache was generated from different windows or FCG weights.')
        if 'numerics' in cache and json.loads(cache['numerics'].item()) != numerics:
            raise ValueError('Future-cache precision differs from the selected FCG training precision.')
        if 'split' in cache and cache['split'].item() != split:
            raise ValueError('Future cache was generated for a different split.')
        cached = torch.from_numpy(cache['future'].astype(np.float32, copy=False))
    if cached.shape != windows.tensors[-1].shape:
        raise ValueError('Cached future nodes do not match the windows.')
    return TensorDataset(*windows.tensors[:-1], cached)


@torch.no_grad()
def validation_mse(model, loader, device, args, cached):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.val_seed)
    environment = args.environment
    shape = (model.future_nodes + 1, model.A.latent_dim)
    fixed_noise = None
    if environment in ('pusht', 'fourroom'):
        fixed_noise = torch.randn((len(loader.dataset), *shape), device=device, generator=generator)
    squared_error, coordinates = 0., 0
    for index, arrays in enumerate(loader):
        history, goal, subgoal, future = (array.to(device) for array in arrays)
        start = index * loader.batch_size
        if environment == 'cube':
            noise = cube_window_noise(start, len(goal), args.val_seed, shape, device)
        elif fixed_noise is not None:
            noise = fixed_noise[start:start+len(goal)]
        else:
            if environment in ('scene2', 'caddy'):
                generator.manual_seed(args.val_seed + index)
            noise = torch.randn((len(goal), *shape), device=device, generator=generator)
        if args.stage == 'fcg':
            prediction = model.generate_future_from_noise(history, goal, noise[:, 1:])
            target = future
        else:
            condition = future if cached else model.generate_future_from_noise(history, goal, noise[:, 1:])
            prediction = model.solve_from_noise(history, goal, noise[:, :1], condition)
            target = subgoal
        squared_error += (prediction - target).square().sum(dtype=torch.float64).item()
        coordinates += target.numel()
    return squared_error / coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=CONFIGS, required=True)
    parser.add_argument('--stage', choices=('fcg', 'fgss'), required=True)
    parser.add_argument('--condition', choices=('true', 'generated'))
    parser.add_argument('--fcg-checkpoint')
    parser.add_argument('--future-cache', help='Optional output of fcp.generate --output-kind future.')
    parser.add_argument('--val-future-cache', help='Fixed generated validation conditions for either solver.')
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=65536, help='Global batch across all ranks.')
    parser.add_argument('--val-batch-size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--init-seed', type=int)
    parser.add_argument('--noise-seed', type=int)
    parser.add_argument('--val-seed', type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    selected = CONFIGS[args.environment]
    protocol = selected['protocol']
    libero = args.environment in ('scene2', 'caddy')
    if args.val_batch_size is None:
        args.val_batch_size = {'pusht': 2048, 'fourroom': 2048, 'cube': 512}.get(args.environment, 1024)
    if args.epochs is None:
        args.epochs = protocol.get('generated_epochs', 200) if args.condition == 'generated' else 200
    if args.seed is None:
        args.seed = 217 if libero and args.stage == 'fcg' else protocol['training_seed']
    if args.init_seed is None:
        args.init_seed = 314160 if args.stage == 'fgss' and (libero or args.environment == 'fourroom') else 314159
    if args.noise_seed is None:
        args.noise_seed = {'pusht': args.seed + 500000003, 'fourroom': 1000003 * args.seed,
                           'cube': args.seed}.get(args.environment, args.seed + 500000003)
        if libero:
            args.noise_seed = 1000220 if args.stage == 'fcg' else 1000221
    if args.val_seed is None:
        args.val_seed = {'pusht': 77148, 'fourroom': 77148, 'cube': 77148}.get(
            args.environment, 20261027 if libero else 1209)
    if args.stage == 'fgss' and not (args.condition and args.fcg_checkpoint):
        parser.error('FGSS training needs --condition and --fcg-checkpoint.')
    if args.stage == 'fcg' and (args.condition or args.fcg_checkpoint):
        parser.error('FGC training starts from scratch.')
    if args.future_cache and (args.stage, args.condition) != ('fgss', 'generated'):
        parser.error('Only Generated FGSS training reads a future cache.')
    if args.val_future_cache and args.stage != 'fgss':
        parser.error('Only FGSS training uses a generated validation cache.')
    if args.condition == 'generated' and not args.future_cache:
        parser.error('Generate fixed Train conditions with fcp.generate and pass --future-cache.')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if (min(args.epochs, args.batch_size, args.val_batch_size) < 1 or args.warmup_epochs < 0
            or args.batch_size % world):
        parser.error('Use positive epochs and batch sizes with the global batch divisible by the number of ranks.')

    device = torch.device(args.device)
    if world > 1:
        if device.type != 'cuda':
            parser.error('Distributed training uses one CUDA device per rank.')
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
        dist.init_process_group('nccl', device_id=device)
    try:
        run(args, selected['architecture'], device, world, libero)
    finally:
        if world > 1:
            dist.destroy_process_group()


def run(args, config, device, world, libero):
    rank = dist.get_rank() if world > 1 else 0
    numerics = UPPER_TRAIN[args.environment]
    settings = numerics.metadata(device)
    torch.manual_seed(314159 if args.stage == 'fgss' else args.init_seed)
    train, shape = read_windows(args.train)
    val, val_shape = read_windows(args.val)
    if val_shape != shape:
        raise ValueError('Train and validation window dimensions differ.')
    if any(config[key] != value for key, value in shape.items()):
        raise ValueError('Window dimensions do not match the selected environment.')
    if args.stage == 'fcg':
        model = FCP(**config).to(device)
        model.B.requires_grad_(False)
        fitted = model.A
    else:
        if Path(args.fcg_checkpoint).resolve() == args.output.resolve():
            parser.error('Use a separate output path for the FGSS checkpoint.')
        frozen, checkpoint = load_checkpoint(args.fcg_checkpoint, 'cpu', 'fcg')
        if checkpoint['config'] != config or checkpoint['environment'] != args.environment:
            raise ValueError('The FCG checkpoint belongs to a different environment or architecture.')
        model = FCP(**config, solver_seed=args.init_seed)
        model.A.load_state_dict(frozen.A.state_dict())
        del frozen, checkpoint
        model.to(device)
        model.A.eval().requires_grad_(False)
        fitted = model.B

    if args.future_cache:
        train = with_future_cache(args.future_cache, args.train, args.fcg_checkpoint, train, settings, 'train')
    if args.val_future_cache:
        val = with_future_cache(args.val_future_cache, args.val, args.fcg_checkpoint, val, settings, 'val')

    network = DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
    optimizer = torch.optim.AdamW(fitted.parameters(), lr=1e-3, weight_decay=1e-4)
    drop_tail = libero or args.environment == 'antmaze'
    steps_per_epoch = len(train) // args.batch_size if drop_tail else math.ceil(len(train) / args.batch_size)
    if not steps_per_epoch:
        raise ValueError('The Train split is smaller than the configured global batch.')
    if not drop_tail and 0 < len(train) % args.batch_size < world:
        raise ValueError('The final global batch must contain at least one window per rank.')
    val_batches = DataLoader(val, batch_size=args.val_batch_size)
    warmup = args.warmup_epochs * (steps_per_epoch if libero else 1)
    total = args.epochs * (steps_per_epoch if libero else 1)
    noise_rng = torch.Generator(device=device).manual_seed(args.noise_seed)
    step, best = 0, float('inf')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    match_pusht = args.environment == 'pusht'
    match_cube = args.environment == 'cube'
    match_fourroom = args.environment == 'fourroom'
    for epoch in range(args.epochs):
        fitted.train()
        if match_cube:
            order = torch.from_numpy(np.random.default_rng(args.seed + epoch).permutation(len(train)))
        else:
            order_device = device if match_pusht or match_fourroom else torch.device('cpu')
            stride = 1009 if match_pusht or match_fourroom else 1
            order_seed = args.seed + stride * epoch
            order = torch.randperm(len(train), device=order_device,
                                   generator=torch.Generator(device=order_device).manual_seed(order_seed)).cpu()
        if match_pusht or match_fourroom:
            noise_rng.manual_seed(args.noise_seed + 104729 * epoch + (1000003 * rank if match_pusht else 0))
        batches = order[:steps_per_epoch * args.batch_size].split(args.batch_size)
        for batch, ids in enumerate(batches):
            shard = ids.tensor_split(world)[rank] if match_cube else ids[rank::world]
            history, goal, subgoal, future = (array[shard].to(device) for array in train.tensors)
            update = step if libero else epoch
            if update < warmup:
                lr = 1e-3 * (update + 1) / warmup
            else:
                progress = (update - warmup) / max(1, total - warmup - 1)
                lr = 1e-6 + (1e-3 - 1e-6) * (1 + math.cos(math.pi * progress)) / 2
            optimizer.param_groups[0]['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            if match_cube:
                noise_rng.manual_seed(args.noise_seed + 100003 * epoch + 1009 * batch + rank)
            local_noise = match_cube or match_pusht
            count = len(shard) if local_noise else len(ids)
            noise = torch.randn((count, model.future_nodes + 1, goal.shape[-1]),
                                device=device, generator=noise_rng)
            time = torch.rand(count, device=device, generator=noise_rng)
            if not local_noise:
                noise = noise[rank::world].contiguous()
                time = time[rank::world].contiguous()
            with numerics.backend(device, model):
                with numerics.autocast(device):
                    if args.stage == 'fcg':
                        loss = network(history, goal, future, noise[:, 1:], time)
                    else:
                        loss = network(history, goal, subgoal[:, None], noise[:, :1], time, future)
                    loss = loss * (world * len(shard) / len(ids))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fitted.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1

        if rank == 0:
            with numerics.forward(device, model):
                mse = validation_mse(model, val_batches, device, args, args.val_future_cache is not None)
            if not math.isfinite(mse):
                raise ValueError(f'Nonfinite validation MSE at epoch {epoch + 1}.')
            if mse < best:
                best = mse
                temporary = args.output.with_name(args.output.name + '.tmp')
                details = dict(seed=args.seed, init_seed=args.init_seed, noise_seed=args.noise_seed,
                               val_seed=args.val_seed, batch_size=args.batch_size,
                               val_batch_size=args.val_batch_size, world_size=world, epochs=args.epochs,
                               numerics=settings)
                torch.save(dict(config=config, model=model.state_dict(), stage=args.stage,
                                condition=args.condition, environment=args.environment,
                                epoch=epoch + 1, val_mse=mse, training=details), temporary)
                temporary.replace(args.output)
            print(f'epoch={epoch + 1} val_mse={mse:.6f} best={best:.6f}', flush=True)
        if world > 1:
            dist.barrier()


if __name__ == '__main__':
    main()
