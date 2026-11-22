"""Native Scene2 and Caddy FCP evaluation on the fixed 150-query suites."""

import argparse
import json
import os
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
from tempfile import TemporaryDirectory


MANIFEST = Path(__file__).with_name('libero_tasks.json')
SAGE_REPO = 'CLTRAY/SAGE'
SAGE_REVISION = 'd81d637d72884d8ef0ad80813d37c583ebfd33c8'
LOWER_SHA256 = {
    'scene2': 'a62b226f1875dfdc31f93ba12ed50e3d4a53ecac47cce34c1db8c68e42fee0ff',
    'caddy': 'c5b6b3cd3384b5e4914942ffb762ad832d38d89436be11ef5446c8abb80c4378',
}
CAMERAS = ('agentview_image', 'robot0_eye_in_hand_image')
PROPOSAL_SEED = {'scene2': 20260829, 'caddy': 42}


def make_controller(args, stats):
    from huggingface_hub import snapshot_download
    import numpy as np
    import torch
    from torchvision.transforms import v2

    from fcp.numerics import UPPER_DEPLOYMENT
    from fcp.release import load_closed_loop_weights, sha256_file
    from sage.native_models import load_world_model

    device = torch.device('cuda')
    suite = 'libero_' + args.environment
    root = args.lewm_root
    if root is None:
        files = f'{suite}/world_model'
        root = Path(snapshot_download(SAGE_REPO, revision=SAGE_REVISION,
                                      allow_patterns=[f'{files}/config.json', f'{files}/weights.pt'])) / files
    lower_path = root / 'weights.pt'
    if sha256_file(lower_path) != LOWER_SHA256[args.environment]:
        raise ValueError('Expected the archived SAGE lower checkpoint for this suite.')
    lower = load_world_model(lower_path, device).eval().requires_grad_(False)
    upper, prior, prior_meta = load_closed_loop_weights(args, args.environment, device)
    prior_stats = prior_meta['normalization']
    prior_mean, prior_scale = [np.asarray(prior_stats[key], np.float32)
                               for key in ('action_mean', 'action_scale')]
    action_mean, action_std = [np.asarray(stats[key], np.float32) for key in ('mean', 'std')]
    preprocess = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]), v2.Resize(112),
    ])

    def images(array):
        return preprocess(torch.from_numpy(np.asarray(array)).permute(0, 3, 1, 2))[None].to(device)

    @torch.inference_mode()
    def plan(agent, wrist, final_agent, final_wrist, recent, elapsed, budget, trial):
        pixels, wrist_pixels = images(agent), images(wrist)
        goal, goal_wrist = images(final_agent[None]), images(final_wrist[None])
        history = lower.encode(dict(pixels=pixels, wrist_pixels=wrist_pixels))['emb'].float()
        final = lower.encode(dict(pixels=goal, wrist_pixels=goal_wrist))['emb'].float()
        proposal_seed = PROPOSAL_SEED[args.environment] + 1009 * trial + 17 * elapsed
        target = final
        if budget - elapsed > 30:
            draws = []
            with UPPER_DEPLOYMENT[args.environment].forward(device, upper):
                for index in range(8):
                    seed = proposal_seed + 1_000_000_007 + index * 1_000_003
                    rng = torch.Generator(device=device).manual_seed(seed)
                    noise = torch.randn(1, 4, 768, device=device, generator=rng)
                    draws.append(upper.sample_from_noise(history, final[:, 0], noise))
            target = torch.stack(draws).float().mean(0)[:, None]
        rng = torch.Generator(device=device).manual_seed(proposal_seed)
        lowdim = history.new_empty(1, 0)
        kwargs = dict(action_horizon=2, far_goal_latents=final)
        top = prior.top_mode(history, target, lowdim, **kwargs)
        sampled = prior.sample(history, target, lowdim, 63, generator=rng, **kwargs)
        normalized = torch.cat([top[:, None], sampled], 1).float().cpu().numpy().reshape(64, 10, 7)
        raw = np.clip(normalized * prior_scale + prior_mean, -1, 1).astype(np.float32)
        raw[..., 6] = np.where(raw[..., 6] >= 0, 1., -1.)
        future = ((raw - action_mean) / action_std).reshape(64, 2, 35)
        known = ((recent - action_mean) / action_std).reshape(2, 35)
        packed = np.concatenate([np.broadcast_to(known[None], (64, 2, 35)), future], 1)
        info = {key: value[:, None].expand(-1, 64, -1, -1, -1, -1) for key, value in
                dict(pixels=pixels, wrist_pixels=wrist_pixels, goal=goal, goal_wrist_pixels=goal_wrist).items()}
        info['goal_emb'] = target
        info['action'] = torch.zeros(1, 64, 3, 35, device=device)
        costs = lower.get_cost(info, torch.from_numpy(packed.astype(np.float32))[None].to(device))[0]
        return raw[int(costs.argmin())]

    return plan


def reset(env, file, query):
    import numpy as np
    from robosuite.utils.binding_utils import MjRenderContextOffscreen

    # Hard resets before the selected one reproduce the collection's seeded history.
    offsets = file['ep_offset'][:]
    for index in range(query['source_episode']):
        env.seed(int(file['episode_seed'][int(offsets[index])]))
        env.reset()
    env.has_offscreen_renderer = True
    env.use_camera_obs = True
    if env.sim._render_context_offscreen is None:
        MjRenderContextOffscreen(env.sim, device_id=env.render_gpu_device_id)
    for name, observable in env._setup_observables().items():
        if name not in env._observables:
            env.add_observable(observable)
    env.seed(query['seed'])
    observed = env.reset()
    for _ in range(query['settle_steps']):
        observed, _, _, _ = env.step(np.zeros(7, np.float32))
    offset = int(offsets[query['source_episode']])
    for action in file['action'][offset:offset + query['start']]:
        observed, _, _, _ = env.step(action)
    return observed, offset


def run_trial(args, data, index, destination):
    from collections import deque
    from hashlib import sha256
    import random

    import h5py
    import hdf5plugin  # Register the demonstration archive's HDF5 filters.
    import numpy as np
    import torch

    sys.path.insert(0, str(args.sage_root.resolve()))
    from sage.runtime.libero import configure

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.mha.set_fastpath_enabled(True)
    query = data['queries'][index]
    with TemporaryDirectory(prefix='fcp_libero_') as folder:
        configure(args.libero_root, Path(folder), args.dataset)
        from sage.eval.libero import make_env, proprio

        plan = make_controller(args, data['scaler'])
        seed = PROPOSAL_SEED[args.environment]
        random.seed(seed)
        np.random.seed(seed)
        env = make_env('libero_' + args.environment, query['task'], render=False)
        try:
            with h5py.File(args.dataset / query['source_shard'], 'r') as file:
                obs, offset = reset(env, file, query)
                start, target = query['start'], query['target']
                initial = np.asarray(env.sim.get_state().flatten(), np.float64)
                if sha256(initial.tobytes()).hexdigest() != query['initial_sha256']:
                    raise ValueError(f'LIBERO initial physics differs from the archived query {index}.')
                history = []
                goals = []
                for key, camera in zip(('pixels', 'wrist_pixels'), CAMERAS):
                    difference = np.abs(np.asarray(obs[camera], np.int16) -
                                        file[key][offset + start].astype(np.int16))
                    if difference.max() > 1 or difference.mean() > .01:
                        raise ValueError(f'LIBERO initial observation differs at query {index}, {camera}.')
                    history.append(deque((file[key][offset + max(0, i)]
                                          for i in range(start - 10, start + 1)), maxlen=11))
                    goals.append(file[key][offset + target])
                if np.abs(proprio(obs) - file['proprio'][offset + start]).max() > 2e-5:
                    raise ValueError(f'LIBERO initial proprioception differs at query {index}.')
                recent = deque((np.zeros(7, np.float32) for _ in range(10)), maxlen=10)
                recent.extend(file['action'][offset + max(0, start - 10):offset + start])
            budget = 10 * ((target - start) // 10) + 120
            elapsed = 0
            if env._check_success():
                raise ValueError(f'LIBERO query {index} is already successful at reset.')
            success = False
            executed = []
            while elapsed < budget and not success:
                agent, wrist = [np.stack([frames[i] for i in (0, 5, 10)]) for frames in history]
                actions = plan(agent, wrist, *goals, np.stack(recent), elapsed, budget, index)
                for action in actions:
                    obs, _, _, _ = env.step(action)
                    elapsed += 1
                    success = bool(env._check_success())
                    recent.append(action.copy())
                    for frames, camera in zip(history, CAMERAS):
                        frames.append(np.asarray(obs[camera]))
                    if args.trace:
                        executed.append(action.tolist())
                    if success:
                        break
            row = dict(trial=index, success=success, executed_steps=elapsed, action_budget=budget,
                       reset_seed=query['seed'], group=query['episode_group'],
                       group_seed=query['evaluation_group_seed'])
            if args.trace:
                row['actions'] = executed
            destination.write_text(json.dumps(row, allow_nan=False))
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=('scene2', 'caddy'), required=True)
    parser.add_argument('--variant', choices=('true', 'generated'), required=True)
    parser.add_argument('--dataset', type=Path, required=True, help='Extracted SAGE suite evaluation archive.')
    parser.add_argument('--sage-root', type=Path, required=True, help='PKU-ML/SAGE source checkout.')
    parser.add_argument('--libero-root', type=Path, required=True, help='LIBERO source checkout.')
    parser.add_argument('--lewm-root', type=Path, help='Directory containing the frozen weights.pt and config.json.')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--weights-root', type=Path)
    source.add_argument('--checkpoint', type=Path, help='Locally trained FGSS checkpoint.')
    parser.add_argument('--prior-root', type=Path, help='Directory containing its matching action prior.')
    parser.add_argument('--trials', nargs='+', type=int)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', nargs=2, metavar=('INDEX', 'RESULT'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    data = json.loads(MANIFEST.read_text())[args.environment]
    if args.worker:
        index, destination = args.worker
        run_trial(args, data, int(index), Path(destination))
        return
    ids = list(range(len(data['queries']))) if args.trials is None else args.trials
    if len(set(ids)) != len(ids) or any(not 0 <= index < len(data['queries']) for index in ids):
        parser.error('Provide unique query IDs between 0 and 149.')
    rows = []
    # OpenBLAS and EGL initialization is process-wide. Apply each archived
    # query's kernel setting before importing the simulator and NumPy.
    with TemporaryDirectory(prefix='fcp_libero_trials_') as directory:
        for index in ids:
            result = Path(directory) / f'{index:03d}.json'
            environ = dict(os.environ, OPENBLAS_CORETYPE=data['queries'][index]['kernel'],
                           MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
            subprocess.run([sys.executable, '-m', 'fcp.closed_loop.libero', *sys.argv[1:],
                            '--worker', str(index), str(result)], env=environ, check=True, stdout=sys.stderr)
            rows.append(json.loads(result.read_text()))
    report = dict(environment=args.environment, variant=args.variant, draws=8, tasks=len(rows),
                  action_seed_base=PROPOSAL_SEED[args.environment],
                  success_percent=100 * mean(row['success'] for row in rows), trials=rows)
    if len(rows) == 150:
        groups = [sum(row['success'] for row in rows if row['group'] == group) for group in range(3)]
        report.update(group_successes=groups, sample_std_pp=stdev(2. * k for k in groups))
    output = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end='')


if __name__ == '__main__':
    main()
