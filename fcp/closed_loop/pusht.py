"""Native FCP planning for the released PushT full-task benchmark."""

import argparse
import json
import math
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from fcp.numerics import PRIOR_TRAIN, PUSHT_DEPLOYMENT
from fcp.release import load_closed_loop_weights, load_frozen_world_model, resolve_file


LEWM_REPO = 'quentinll/lewm-pusht'
LEWM_REVISION = '22b330c28c27ead4bfd1888615af1340e3fe9052'
LEWM_SHA256 = '48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'
TASK_BATCH_SIZE = 16


class PushTController:
    def __init__(self, fcp, world_model, prior, metadata, device):
        from torchvision.transforms import v2

        self.fcp, self.world, self.device = fcp, world_model, device
        self.prior = prior
        self.stats = {key: torch.tensor(value, device=device)
                      for key, value in metadata['normalization'].items() if key != 'lowdim_keys'}
        self.mean = torch.tensor([-0.007812564379916172, 0.006860687229453032], device=device)
        self.std = torch.tensor([0.20846744284501714, 0.20674862637362224], device=device)
        self.transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def encode(self, images):
        pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
        pixels = self.transform(pixels).to(self.device)
        count = len(pixels)
        if self.device.type == 'cuda' and count < 16:
            pixels = pixels[torch.arange(16, device=self.device) % count]
        return self.world.encode({'pixels': pixels[:, None]})['emb'][:count, 0]

    def subgoal(self, current, goal, seeds):
        with PUSHT_DEPLOYMENT.forward(self.device, self.fcp):
            values = []
            for draw in range(8):
                generators = [torch.Generator(device=self.device).manual_seed(seed + draw * 1_000_003)
                              for seed in seeds]
                noise = torch.stack([torch.randn(4, 192, device=self.device, generator=rng)
                                     for rng in generators])
                values.append(self.fcp.sample_from_noise(current, goal, noise))
        return torch.stack(values).float().mean(dim=0)

    def prior_bank(self, history, target, goal, states, distances, generators):
        stats = self.stats
        lowdim = torch.as_tensor(np.concatenate((states, states[:, :2], states[:, -2:]), axis=1),
                                 dtype=torch.float32, device=self.device)
        lowdim = (lowdim - stats['lowdim_mean']) / stats['lowdim_std'].clamp_min(1e-6)
        with PRIOR_TRAIN['pusht'].forward(self.device):
            samples = torch.cat([
                self.prior.sample(history[i:i+1], target[i:i+1, None], lowdim[i:i+1],
                                  300, generator=rng, action_horizon=2,
                                  far_goal_latents=goal[i:i+1, None],
                                  goal_offset_steps=target.new_tensor([distances[i]]),
                                  subgoal_offset_steps=target.new_tensor([10])).float()
                for i, rng in enumerate(generators)
            ])
        actions = samples.reshape(len(states), 300, 2, 5, 2) * stats['action_std'] + stats['action_mean']
        return ((actions - self.mean) / self.std).reshape(len(states), 300, 2, 10)

    def plan(self, current, target, goal, blocks, history, states, distances, seeds):
        generators = [torch.Generator(device=self.device).manual_seed(seed) for seed in seeds]
        batch = len(seeds)
        mean = torch.zeros((batch, blocks, 10), device=self.device)
        std = torch.ones_like(mean)
        pixels = mean.new_zeros((batch, 1, 1))[:, None].expand(batch, 300, 1, 1)
        initial = current[:, None, None, :].expand(batch, 300, 1, 192)
        rows = torch.arange(batch, device=self.device)[:, None]
        for index in range(30):
            # The native solver consumes the Gaussian random draw before
            # substituting the learned prior in round zero.
            noise = torch.stack([torch.randn(300, blocks, 10, device=self.device, generator=rng)
                                 for rng in generators])
            bank = noise * std[:, None] + mean[:, None]
            bank[:, 0] = mean
            if index == 0 and blocks == 2:
                bank = self.prior_bank(history, target, goal, states, distances, generators)
            predicted = self.world.rollout({'pixels': pixels, 'emb': initial}, bank)
            errors = (predicted['predicted_emb'][:, :, -1].float() - target[:, None]).square().sum(-1)
            elite = bank[rows, errors.topk(30, dim=1, largest=False).indices]
            mean, std = elite.mean(dim=1), elite.std(dim=1).clamp_min(1e-6)
        if not torch.isfinite(mean).all():
            raise FloatingPointError('Nonfinite PushT action.')
        return (mean.reshape(batch, -1, 2) * self.std + self.mean).cpu().numpy()


def reset(initial, goal, seed):
    from stable_worldmodel.envs.pusht import PushT

    env = PushT(resolution=224, with_target=True)
    env.reset(seed=seed, options={'state': initial, 'goal_state': goal})
    env.agent.position = initial[:2].tolist()
    env.agent.velocity = initial[-2:].tolist()
    env.block.angle = float(initial[4])
    env.block.position = initial[2:4].tolist()
    env.block.velocity = (0., 0.)
    env.block.angular_velocity = 0.
    for body in (env.agent, env.block):
        body.force = (0., 0.)
        body.torque = 0.
        env.space.reindex_shapes_for_body(body)
    env._set_goal_state(goal.astype(np.float64))
    delta = env._get_obs() - initial
    delta[4] = np.arctan2(np.sin(delta[4]), np.cos(delta[4]))
    np.testing.assert_allclose(delta, 0., rtol=0, atol=1e-9)
    return env


def goal_polygon():
    import shapely

    vertices = np.array([
        [[-60, 30], [60, 30], [60, 0], [-60, 0]],
        [[-15, 30], [-15, 120], [15, 120], [15, 30]],
    ])
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return shapely.union_all(shapely.polygons(vertices @ rotation.T + [256., 256.]))


def coverage(env, goal):
    import shapely

    polygons = [shapely.Polygon([tuple(env.block.local_to_world(v)) for v in shape.get_vertices()])
                for shape in env.block.shapes]
    return float(shapely.union_all(polygons).intersection(goal).area / goal.area)


def schedule(elapsed, budget):
    remaining = budget - elapsed
    direct = remaining <= 30
    boundary = budget - 30 if not direct else budget - 10 if remaining > 10 else budget
    duration = min(5 if remaining <= 10 else 10, boundary - elapsed)
    return math.ceil(duration / 5), duration, direct


@torch.inference_mode()
def run_tasks(controller, tasks, trace=False):
    episodes = []
    try:
        polygon = goal_polygon()
        for task in tasks:
            seed = int(task.attrs['evaluation_seed'])
            trial = int(task.attrs['trial'])
            env = reset(task['state'][0], task['state'][-1], seed + trial)
            row = dict(trial=trial, episode_id=int(task.attrs['episode_id']), seed=seed,
                       first_hit85=None, first_hit_native=None, executed_steps=0)
            episode = dict(env=env, row=row, horizon=int(task.attrs['H']),
                           budget=int(task.attrs['action_budget']), goal_image=task['pixels'][-1])
            episodes.append(episode)
            image = env.render()
            if np.abs(np.asarray(image, dtype=np.float32) - task['pixels'][0]).mean() > 1:
                raise ValueError('Native reset does not match the benchmark initial observation.')
            episode['history'] = deque([(0, image)], maxlen=3)
            if trace:
                row.update(actions=[], subgoals=[])

        goals = controller.encode([episode['goal_image'] for episode in episodes])
        for episode, goal in zip(episodes, goals):
            episode['goal'] = goal
        while True:
            groups = defaultdict(list)
            for episode in episodes:
                row = episode['row']
                if row['executed_steps'] < episode['budget'] and not finished(row):
                    groups[schedule(row['executed_steps'], episode['budget'])].append(episode)
            if not groups:
                break
            for (blocks, duration, direct), group in groups.items():
                plan_group(controller, group, blocks, duration, direct, polygon, trace)
        return [dict(episode['row'], success=episode['row']['first_hit85'] is not None)
                for episode in episodes]
    finally:
        for episode in episodes:
            episode['env'].close()


def finished(row):
    return row['first_hit85'] is not None and row['first_hit_native'] is not None


def plan_group(controller, group, blocks, duration, direct, polygon, trace):
    current = controller.encode([episode['env'].render() for episode in group])
    goals = torch.stack([episode['goal'] for episode in group])
    elapsed = [episode['row']['executed_steps'] for episode in group]
    horizons = [episode['horizon'] for episode in group]
    seeds = [episode['row']['seed'] + episode['row']['trial'] * 100_003 + step * 1009
             for episode, step in zip(group, elapsed)]
    target = goals if direct else controller.subgoal(current, goals, [seed + 9_000_007 for seed in seeds])
    distances = [max(10, horizon - step) for horizon, step in zip(horizons, elapsed)]
    states = np.stack([episode['env']._get_obs() for episode in group])
    history = None
    if blocks == 2:
        frames = [dict(episode['history']) for episode in group]
        history = torch.stack([controller.encode([source[max(0, step + offset)]
                                                  for source, step in zip(frames, elapsed)])
                               for offset in (-10, -5, 0)], dim=1)
    actions = controller.plan(current, target, goals, blocks, history, states, distances,
                              [seed + 1000 * h for seed, h in zip(seeds, horizons)])
    for index, (episode, proposed) in enumerate(zip(group, actions)):
        env, row = episode['env'], episode['row']
        if trace:
            row['subgoals'].append(target[index].tolist())
        for action in proposed[:duration]:
            _, _, pose, truncated, _ = env.step(action)
            if truncated:
                raise RuntimeError('Unexpected PushT simulator truncation.')
            row['executed_steps'] += 1
            step = row['executed_steps']
            if row['first_hit85'] is None and coverage(env, polygon) > 0.85:
                row['first_hit85'] = step
            if row['first_hit_native'] is None and pose:
                row['first_hit_native'] = step
            if step % 5 == 0:
                episode['history'].append((step, env.render()))
            if trace:
                row['actions'].append(action.tolist())
            if finished(row):
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=('true', 'generated'))
    parser.add_argument('--trials', type=int, nargs='+', help='Zero-based task IDs. Default is all 440.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--weights-root', type=Path)
    source.add_argument('--checkpoint', type=Path, help='Use locally trained FCG/FGSS weights.')
    parser.add_argument('--prior-root', type=Path, help='Use a compatible locally trained prior.')
    parser.add_argument('--lewm-root', type=Path)
    parser.add_argument('--task-file', type=Path)
    parser.add_argument('--output', type=Path, help='Save the per-task results as JSON.')
    parser.add_argument('--trace', action='store_true', help='Include executed actions and subgoal latents.')
    args = parser.parse_args()
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import h5py

    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.mha.set_fastpath_enabled(True)
    fcp, prior, metadata = load_closed_loop_weights(args, 'pusht', device)
    lower = load_frozen_world_model(device, args.lewm_root, repo=LEWM_REPO,
                                   revision=LEWM_REVISION, checksum=LEWM_SHA256)
    controller = PushTController(fcp, lower, prior, metadata, device)
    task_file = args.task_file or resolve_file('pusht/full440.h5', dataset=True)
    with h5py.File(task_file, 'r') as tasks:
        ordered = sorted(map(int, tasks))
        ids = ordered if args.trials is None else args.trials
        if len(set(ids)) != len(ids) or any(f'{key:03d}' not in tasks for key in ids):
            parser.error('Provide unique task IDs present in the PushT full440 file.')
        selected = set(ids)
        by_id = {}
        for start in range(0, len(ordered), TASK_BATCH_SIZE):
            batch = ordered[start:start + TASK_BATCH_SIZE]
            if selected.intersection(batch):
                by_id.update((row['trial'], row)
                             for row in run_tasks(controller, [tasks[f'{key:03d}'] for key in batch], args.trace))
                print(f'completed PushT batch {batch[0]}–{batch[-1]}', file=sys.stderr, flush=True)
        rows = [by_id[key] for key in ids]
    result = dict(environment='pusht', variant=args.variant, tasks=len(rows), draws=8,
                  task_batch_size=TASK_BATCH_SIZE, evaluation_seeds=sorted({row['seed'] for row in rows}),
                  success_percent=100 * sum(row['success'] for row in rows) / len(rows), trials=rows)
    report = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
