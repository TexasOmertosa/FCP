"""Native FCP-only Cube evaluation on the fixed segmented tasks."""

import argparse
from collections import deque
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import hdf5plugin  # Register the demonstration archive's HDF5 filters.
import numpy as np
import stable_worldmodel as swm
import torch
from sklearn.preprocessing import StandardScaler
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal
from torchvision.transforms import v2

from fcp.numerics import UPPER_DEPLOYMENT
from fcp.release import CONFIGS, load_closed_loop_weights, load_frozen_world_model


TASK_BATCH_SIZE = 10
MANIFEST = Path(__file__).with_name('cube_tasks.json')
LEWM_REPO = 'quentinll/lewm-cube'
LEWM_REVISION = 'b0747c5002e86d2ce8f3cd8178004b97524c587d'
LEWM_SHA256 = '2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89'


def task_seed(seed, task, elapsed, stream):
    return int(np.random.SeedSequence([seed, task, elapsed, stream]).generate_state(1, np.uint64)[0])


class FirstBank:
    def __init__(self, world_model):
        self.world_model = world_model
        self.candidates = None

    def __getattr__(self, key):
        return getattr(self.world_model, key)

    def get_cost(self, info, actions):
        if self.candidates is not None:
            actions.copy_(self.candidates)
            self.candidates = None
        return self.world_model.get_cost(info, actions)


class Solver(swm.solver.CEMSolver):
    def __init__(self, lower, upper, prior, metadata, scaler, horizon, seed, device):
        super().__init__(model=FirstBank(lower), batch_size=1, num_samples=300, var_scale=1.,
                         n_steps=30, topk=30, device=device, seed=seed)
        self.upper, self.prior = upper, prior
        self.stats = {key: torch.tensor(value, device=device)
                      for key, value in metadata['normalization'].items() if key != 'lowdim_keys'}
        self.mean = torch.as_tensor(scaler.mean_, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device)
        self.task_horizon, self.seed, self.elapsed = horizon, seed, 0

    def subgoal(self, current, goal, tasks):
        seeds = [task_seed(self.seed, task, self.elapsed, 1) for task in tasks]
        draws = []
        for draw in range(8):
            generators = [torch.Generator(device=self.device).manual_seed(
                (seed + draw * 1_000_003) % 2**64) for seed in seeds]
            noise = torch.stack([torch.randn(4, 192, device=self.device, generator=rng)
                                 for rng in generators])
            draws.append(self.upper.sample_from_noise(current, goal, noise))
        return torch.stack(draws).float().mean(dim=0)

    def prior_candidates(self, history, target, goal, observation, task):
        stats = self.stats
        obs = (observation.to(self.device).float() - stats['lowdim_mean']) / stats['lowdim_std'].clamp_min(1e-6)
        rng = torch.Generator(device=self.device).manual_seed(task_seed(self.seed, task, self.elapsed, 3))
        blocks = self.prior.sample(
            history, target, obs, self.num_samples, generator=rng, action_horizon=2,
            far_goal_latents=goal,
            goal_offset_steps=target.new_full((len(target),), max(10, self.task_horizon - self.elapsed)),
            subgoal_offset_steps=target.new_full((len(target),), 10),
        ).float()
        dense = blocks.reshape(*blocks.shape[:-1], 5, 5)
        raw = dense * stats['action_std'] + stats['action_mean']
        return ((raw - self.mean) / self.std).reshape_as(blocks)

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        ids = info_dict['task_id'].flatten().tolist()
        current = self.model.encode({'pixels': info_dict['pixels'].to(self.device)})['emb']
        goal = self.model.encode({'pixels': info_dict['goal'].to(self.device)})['emb']
        history = None
        if self.horizon == 2:
            history = self.model.encode({'pixels': info_dict['pixels_prior_history'].to(self.device)})['emb']
        target = goal[:, -1] if self.elapsed >= 2 * self.task_horizon - 30 else self.subgoal(
            current[:, -1], goal[:, -1], ids)
        prepared = {key: value for key, value in info_dict.items() if key != 'pixels_prior_history'}
        prepared.update(emb=current, goal_emb=target[:, None])
        actions = []
        for row, task in enumerate(ids):
            self.torch_gen.manual_seed(task_seed(self.seed, task, self.elapsed, 2))
            sliced = {key: value[row:row+1] for key, value in prepared.items()}
            if history is not None:
                self.model.candidates = self.prior_candidates(
                    history[row:row+1], target[row:row+1, None], goal[row:row+1, -1:],
                    sliced['observation'][:, -1], task)
            initial = init_action[row:row+1] if init_action is not None else None
            with redirect_stdout(io.StringIO()):
                actions.append(super().solve(sliced, init_action=initial)['actions'])
            if self.model.candidates is not None:
                raise RuntimeError('CEM did not score the matched prior bank.')
        return {'actions': torch.cat(actions, dim=0)}


class Policy(swm.policy.WorldModelPolicy):
    def __init__(self, solver, scaler, task_offset):
        transform = v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), v2.Resize(224),
        ])
        super().__init__(solver=solver,
                         config=swm.PlanConfig(horizon=2, receding_horizon=2, action_block=5,
                                               history_len=1, warm_start=False),
                         process={'action': scaler},
                         transform={key: transform for key in ('pixels', 'goal', 'pixels_prior_history')})
        self.task_offset = task_offset
        self.executed = 0
        self.history = deque(maxlen=11)
        self.schedule = segment_lengths(solver.task_horizon)

    def get_action(self, info_dict, **kwargs):
        info = dict(info_dict)
        info['task_id'] = (np.arange(self.env.num_envs, dtype=np.int64) + self.task_offset)[:, None]
        self.history.append(np.asarray(info['pixels'])[:, -1].copy())
        history = list(self.history)
        info['pixels_prior_history'] = np.stack(
            [history[max(0, len(history) - 1 - lag)] for lag in (10, 5, 0)], axis=1)
        info['observation'] = np.stack([
            np.asarray(env.unwrapped.compute_observation(), dtype=np.float32)
            for env in self.env.envs])[:, None]
        self.solver.elapsed = self.executed
        length = self.schedule.get(self.executed, 10) // 5
        object.__setattr__(self.cfg, 'horizon', length)
        object.__setattr__(self.cfg, 'receding_horizon', length)
        try:
            action = super().get_action(info, **kwargs)
        finally:
            object.__setattr__(self.cfg, 'horizon', 2)
            object.__setattr__(self.cfg, 'receding_horizon', 2)
        self.executed += 1
        return action


def segment_lengths(horizon):
    budget = 2 * horizon
    boundaries = {budget - 10, budget}
    if budget - 30 < horizon < budget:
        boundaries.add(horizon)
    result = {}
    start = 0
    while start < budget:
        duration = min(10, min(boundary for boundary in boundaries if boundary > start) - start)
        result[start] = duration
        start += duration
    return result


def run_chunk(solver, scaler, dataset, tasks, offset):
    horizon = solver.task_horizon
    budget = 2 * horizon
    policy = Policy(solver, scaler, offset)
    world = swm.World('swm/OGBCube-v0', num_envs=len(tasks), image_shape=(224, 224),
                      max_episode_steps=budget, env_type='single', ob_type='states',
                      multiview=False, width=224, height=224, visualize_info=False,
                      terminate_at_goal=True)
    world.set_policy(policy)
    callables = [
        {'method': 'set_state', 'args': {'qpos': {'value': 'qpos'}, 'qvel': {'value': 'qvel'}}},
        {'method': 'set_target_pos', 'args': {'cube_id': {'value': 0, 'in_dataset': False},
                                               'target_pos': {'value': 'goal_privileged_block_0_pos'},
                                               'target_quat': {'value': 'goal_privileged_block_0_quat'}}},
    ]
    try:
        episodes, starts = map(list, zip(*tasks))
        init, goal, _ = _extract_init_goal(dataset, episodes, starts, horizon)
        world.reset(seed=init.get('seed'))
        merged = {**init, **goal}
        for index, env in enumerate(world.envs.envs):
            _apply_callables(env.unwrapped, callables, {key: value[index] for key, value in merged.items()})
        leading = world.infos['pixels'].shape[:2]
        for source in (init, goal):
            for key, values in source.items():
                if key in world.infos or key in goal:
                    world.infos[key] = np.broadcast_to(values[:, None], leading + values.shape[1:]).copy()
        goal_snapshot = {key: world.infos[key].copy() for key in goal}
        initial = np.linalg.norm(init['privileged_block_0_pos'] - goal['goal_privileged_block_0_pos'],
                                 axis=-1) <= .04
        first_hit = np.where(initial, 0, -1)
        alive = ~initial.copy()
        for _ in range(budget):
            if not alive.any():
                break
            world.infos['terminated'] = (~alive)[:, None]
            actions = world._get_actions()
            _, world.rewards, world.terminateds, world.truncateds, world.infos = world.envs.step(
                actions, mask=alive)
            world.infos.update(deepcopy(goal_snapshot))
            hit = (first_hit < 0) & world.terminateds.astype(bool)
            first_hit[hit] = policy.executed
            alive &= ~(world.terminateds | world.truncateds)
        return [dict(trial=offset + index, episode_id=episode, start_frame=start,
                     first_hit=int(hit), success=bool(hit >= 0))
                for index, ((episode, start), hit) in enumerate(zip(tasks, first_hit))]
    finally:
        for env in world.envs.envs:
            native = env.unwrapped
            renderer = getattr(native, '_renderer', None)
            if renderer is not None:
                renderer.close()
                native._renderer = None
        world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=('true', 'generated'))
    parser.add_argument('--dataset', required=True, type=Path, help='Original cube_single_expert.h5.')
    parser.add_argument('--horizon', type=int, choices=CONFIGS['cube']['protocol']['segmented_horizons'], required=True)
    parser.add_argument('--seed', type=int, choices=CONFIGS['cube']['protocol']['evaluation_seeds'], required=True)
    parser.add_argument('--chunk', type=int, choices=range(10), help='Evaluate one original ten-task batch.')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--weights-root', type=Path)
    source.add_argument('--checkpoint', type=Path, help='Locally trained FGSS checkpoint.')
    parser.add_argument('--prior-root', type=Path, help='Directory containing its matching action prior.')
    parser.add_argument('--lewm-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    device = torch.device('cuda')
    torch.set_num_threads(2)
    data = json.loads(MANIFEST.read_text())
    scaler = StandardScaler()
    stats = data['scaler']
    for key in ('mean', 'scale', 'var'):
        setattr(scaler, key + '_', np.asarray(stats[key], dtype=np.float64))
    scaler.n_features_in_ = 5
    scaler.n_samples_seen_ = stats['n_samples_seen']
    tasks = data['tasks'][str(args.seed)][str(args.horizon)]
    upper, prior, metadata = load_closed_loop_weights(args, 'cube', device)
    lower = load_frozen_world_model(device, args.lewm_root, repo=LEWM_REPO, revision=LEWM_REVISION,
                                   checksum=LEWM_SHA256)
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=['action'])
    rows = []
    with UPPER_DEPLOYMENT['cube'].forward(device, upper):
        for chunk in range(10) if args.chunk is None else [args.chunk]:
            start = chunk * TASK_BATCH_SIZE
            solver = Solver(lower, upper, prior, metadata, scaler, args.horizon, args.seed, device)
            rows.extend(run_chunk(solver, scaler, dataset, tasks[start:start+TASK_BATCH_SIZE], start))
    result = dict(environment='cube', variant=args.variant, H=args.horizon, seed=args.seed, draws=8,
                  tasks=len(rows), success_percent=100 * sum(row['success'] for row in rows) / len(rows),
                  task_batch_size=TASK_BATCH_SIZE, action_budget=2 * args.horizon,
                  trials=rows)
    report = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end='')


if __name__ == '__main__':
    main()
