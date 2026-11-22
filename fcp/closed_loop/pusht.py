"""Native FCP planning for the released PushT full-task benchmark."""

import argparse
import json
import math
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch

from fcp.model import load_checkpoint
from priors.weights import load_prior
from fcp.release import CONFIGS, load_selected, resolve_file, sha256_file


LEWM_REPO = 'quentinll/lewm-pusht'
LEWM_REVISION = '22b330c28c27ead4bfd1888615af1340e3fe9052'
LEWM_SHA256 = '48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'


def load_world_model(device, root):
    from huggingface_hub import snapshot_download
    from hydra.utils import instantiate

    folder = Path(root or snapshot_download(
        LEWM_REPO, revision=LEWM_REVISION, allow_patterns=['weights.pt', 'config.json'],
    ))
    checkpoint = folder / 'weights.pt'
    if sha256_file(checkpoint) != LEWM_SHA256:
        raise ValueError('Expected the frozen LeWM checkpoint used for PushT full440.')
    config = json.loads((folder / 'config.json').read_text())
    model = instantiate(config)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False)


class PushTController:
    def __init__(self, fcp, world_model, variant, weights_root, device, prior_root=None):
        from torchvision.transforms import v2

        self.fcp, self.world, self.device = fcp, world_model, device
        if device.type == 'cuda':
            attention = torch.nn.attention.SDPBackend
            for field in (fcp.A, fcp.B):
                field.frequencies.copy_(torch.exp(-math.log(10000) * torch.arange(64, device=device) / 63))
                for block in field.blocks:
                    block.attention_backends = (attention.EFFICIENT_ATTENTION, attention.FLASH_ATTENTION)
        self.prior, assets = load_prior('pusht', variant, device, weights_root=weights_root,
                                        prior_root=prior_root, upper=fcp)
        self.stats = {key: torch.tensor(value, device=device)
                      for key, value in assets['normalization'].items() if key != 'lowdim_keys'}
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

    def subgoal(self, current, goal, seed):
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            values = []
            for draw in range(8):
                rng = torch.Generator(device=self.device).manual_seed(seed + draw * 1_000_003)
                noise = torch.randn(4, 192, device=self.device, generator=rng)[None]
                values.append(self.fcp.sample_from_noise(current, goal, noise))
        return torch.stack(values).float().mean(dim=0)

    def prior_bank(self, history, target, goal, state, distance, generator):
        stats = self.stats
        lowdim = torch.as_tensor(np.concatenate((state, state[:2], state[-2:])),
                                 dtype=torch.float32, device=self.device)[None]
        lowdim = (lowdim - stats['lowdim_mean']) / stats['lowdim_std'].clamp_min(1e-6)
        fastpath = torch.backends.mha.get_fastpath_enabled()
        tf = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.mha.set_fastpath_enabled(False)
            torch.backends.cudnn.allow_tf32 = False
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                    enabled=self.device.type == 'cuda'):
                    samples = self.prior.sample(
                        history[None], target[:, None], lowdim, 300, generator=generator,
                        action_horizon=2, far_goal_latents=goal[:, None],
                        goal_offset_steps=target.new_tensor([distance]),
                        subgoal_offset_steps=target.new_tensor([10]),
                    ).float()
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)
            torch.backends.cudnn.allow_tf32 = tf
        actions = samples.reshape(300, 2, 5, 2) * stats['action_std'] + stats['action_mean']
        return ((actions - self.mean) / self.std).reshape(300, 2, 10)

    def plan(self, current, target, goal, blocks, history, state, distance, seed):
        rng = torch.Generator(device=self.device).manual_seed(seed)
        mean = torch.zeros((blocks, 10), device=self.device)
        std = torch.ones_like(mean)
        pixels = mean.new_zeros((1, 300, 1, 1))
        initial = current[None, :, None, :].expand(1, 300, 1, 192)
        for index in range(30):
            # The native solver consumes the Gaussian random draw before
            # substituting the learned prior in round zero.
            bank = torch.randn((300, blocks, 10), device=self.device, generator=rng) * std + mean
            bank[0] = mean
            if index == 0 and blocks == 2:
                bank = self.prior_bank(history, target, goal, state, distance, rng)
            predicted = self.world.rollout({'pixels': pixels, 'emb': initial}, bank[None])
            errors = (predicted['predicted_emb'][0, :, -1].float() - target).square().sum(-1)
            elite = bank[errors.topk(30, largest=False).indices]
            mean, std = elite.mean(dim=0), elite.std(dim=0).clamp_min(1e-6)
        if not torch.isfinite(mean).all():
            raise FloatingPointError('Nonfinite PushT action.')
        return (mean.reshape(-1, 2) * self.std + self.mean).cpu().numpy()


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
    np.testing.assert_allclose(env._get_obs(), initial, rtol=0, atol=1e-9)
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


@torch.inference_mode()
def run_task(controller, task):
    seed = int(task.attrs['evaluation_seed'])
    trial = int(task.attrs['trial'])
    horizon = int(task.attrs['H'])
    budget = int(task.attrs['action_budget'])
    env = reset(task['state'][0], task['state'][-1], seed + trial)
    try:
        goal = goal_polygon()
        image = env.render()
        if np.abs(np.asarray(image, dtype=np.float32) - task['pixels'][0]).mean() > 1:
            raise ValueError('Native reset does not match the benchmark initial observation.')
        far = controller.encode([task['pixels'][-1]])
        observations = deque([(0, controller.encode([image])[0])], maxlen=3)
        first_coverage, first_pose = None, None
        elapsed = 0
        while elapsed < budget and (first_coverage is None or first_pose is None):
            remaining = budget - elapsed
            direct = remaining <= 30
            boundary = budget - 30 if not direct else budget - 10 if remaining > 10 else budget
            duration = min(5 if remaining <= 10 else 10, boundary - elapsed)
            blocks = math.ceil(duration / 5) if direct else 2
            current = controller.encode([env.render()])
            target_seed = seed + trial * 100_003 + elapsed * 1009 + 9_000_007
            target = far if direct else controller.subgoal(current, far, target_seed)
            history = dict(observations)
            recent = torch.stack([history[max(0, elapsed + offset)] for offset in (-10, -5, 0)])
            cem_seed = seed + trial * 100_003 + 1000 * horizon + elapsed * 1009
            actions = controller.plan(current, target, far, blocks, recent, env._get_obs(),
                                      max(10, horizon - elapsed), cem_seed)
            for action in actions[:duration]:
                _, _, pose, truncated, _ = env.step(action)
                if truncated:
                    raise RuntimeError('Unexpected PushT simulator truncation.')
                elapsed += 1
                if first_coverage is None and coverage(env, goal) > 0.85:
                    first_coverage = elapsed
                if first_pose is None and pose:
                    first_pose = elapsed
                if elapsed % 5 == 0:
                    observations.append((elapsed, controller.encode([env.render()])[0]))
                if first_coverage is not None and first_pose is not None:
                    break
        return dict(trial=trial, episode_id=int(task.attrs['episode_id']), seed=seed,
                    first_hit85=first_coverage, first_hit_native=first_pose,
                    executed_steps=elapsed, success=first_coverage is not None)
    finally:
        env.close()


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
    args = parser.parse_args()
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import h5py

    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.mha.set_fastpath_enabled(True)
    if args.checkpoint:
        if args.prior_root is None:
            parser.error('Provide the matching --prior-root for a local FCP checkpoint.')
        fcp, metadata = load_checkpoint(args.checkpoint, device, 'fgss')
        if (metadata['environment'], metadata['condition'], metadata['config']) != (
                'pusht', args.variant, CONFIGS['pusht']['architecture']):
            raise ValueError('The FCG/FGSS checkpoint belongs to a different configuration.')
        fcp.eval()
    else:
        fcp = load_selected('pusht', args.variant, device, args.weights_root)
    lower = load_world_model(device, args.lewm_root)
    controller = PushTController(fcp, lower, args.variant, args.weights_root, device, args.prior_root)
    task_file = args.task_file or resolve_file('pusht/full440.h5', dataset=True)
    with h5py.File(task_file, 'r') as tasks:
        ids = sorted(map(int, tasks)) if args.trials is None else args.trials
        if len(set(ids)) != len(ids) or any(f'{key:03d}' not in tasks for key in ids):
            parser.error('Provide unique task IDs present in the PushT full440 file.')
        rows = [run_task(controller, tasks[f'{key:03d}']) for key in ids]
    result = dict(environment='pusht', variant=args.variant, tasks=len(rows), draws=8,
                  success_percent=100 * sum(row['success'] for row in rows) / len(rows), trials=rows)
    report = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
